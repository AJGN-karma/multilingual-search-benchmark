"""
Streaming dataset audit for the multilingual Wikipedia
title retrieval benchmark.

This script:
    1. Reads the supplied manifest.
    2. Checks whether dataset files exist.
    3. Verifies CSV column names and order.
    4. Counts logical CSV records without loading the
       entire file into RAM.
    5. Checks the language and Wikipedia identifiers.
    6. Validates title_display.
    7. Validates Unicode NFC normalization.
    8. Validates code-point length.
    9. Validates UTF-8 byte length.
    10. Optionally verifies SHA-256.

It deliberately does not load all titles into a Python list
or set.

Duplicate detection will be handled separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAX_ERROR_SAMPLES = 10


def add_issue(
    result: dict[str, Any],
    issue_type: str,
    message: str
) -> None:
    """Record an error without storing every error message."""

    result["issue_counts"][issue_type] = (
        result["issue_counts"].get(issue_type, 0) + 1
    )

    if len(result["error_samples"]) < MAX_ERROR_SAMPLES:
        result["error_samples"].append(message)


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate SHA-256 using fixed-size blocks.

    Memory usage remains small even for large files.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(8 * 1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def audit_dataset(
    dataset_key: str,
    dataset_config: dict[str, Any],
    manifest_record: dict[str, Any] | None,
    required_columns: list[str],
    verify_sha256: bool
) -> dict[str, Any]:

    file_path = PROJECT_ROOT / dataset_config["relative_path"]

    result: dict[str, Any] = {
        "dataset": dataset_key,
        "filename": dataset_config["filename"],
        "language": dataset_config["language"],
        "status": "PENDING",
        "expected_rows": None,
        "actual_rows": 0,
        "expected_file_size_bytes": None,
        "actual_file_size_bytes": None,
        "row_count_match": False,
        "schema_match": False,
        "sha256_checked": False,
        "sha256_match": None,
        "duplicates_checked": False,
        "issue_counts": {},
        "warnings": [],
        "error_samples": []
    }

    # Check manifest record
    if manifest_record is None:
        add_issue(
            result,
            "manifest_record_missing",
            "Dataset was not found in the manifest."
        )

        result["status"] = "FAIL"
        return result

    expected_rows = manifest_record.get("row_count")
    expected_size = manifest_record.get("file_size_bytes")

    result["expected_rows"] = expected_rows
    result["expected_file_size_bytes"] = expected_size

    if not file_path.is_file():
        add_issue(
            result,
            "file_missing",
            f"Dataset file not found: {file_path}"
        )

        result["status"] = "MISSING"
        return result

    result["actual_file_size_bytes"] = file_path.stat().st_size

    # File-size comparison is reported separately.
    # A different size may indicate a changed line ending
    # or a modified/re-exported CSV.
    if expected_size is not None:
        if result["actual_file_size_bytes"] != expected_size:
            result["warnings"].append(
                "File size differs from the manifest."
            )

    # Read the CSV using streaming iteration.
    try:
        with file_path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            actual_columns = reader.fieldnames or []

            result["schema_match"] = (
                actual_columns == required_columns
            )

            if not result["schema_match"]:
                add_issue(
                    result,
                    "schema_mismatch",
                    f"Expected columns {required_columns}; "
                    f"received {actual_columns}."
                )

            missing_columns = [
                column
                for column in required_columns
                if column not in actual_columns
            ]

            # We can still count rows if some required
            # fields are missing, but cannot validate them.
            can_validate_fields = not missing_columns

            if missing_columns:
                add_issue(
                    result,
                    "required_columns_missing",
                    f"Missing columns: {missing_columns}"
                )

            for row in reader:

                result["actual_rows"] += 1

                row_number = reader.line_num

                # Extra fields are captured under key None.
                if None in row:
                    add_issue(
                        result,
                        "extra_fields",
                        f"Unexpected extra fields near "
                        f"CSV line {row_number}."
                    )

                if not can_validate_fields:
                    continue

                missing_values = [
                    column
                    for column in required_columns
                    if row.get(column) is None
                ]

                if missing_values:
                    add_issue(
                        result,
                        "missing_field_values",
                        f"Missing values near CSV line "
                        f"{row_number}: {missing_values}"
                    )
                    continue

                # Validate non-empty essential title fields.
                essential_fields = [
                    "wiki_id",
                    "language",
                    "title_source",
                    "title_display",
                    "title_nfc",
                    "codepoint_length",
                    "utf8_bytes"
                ]

                empty_fields = [
                    column
                    for column in essential_fields
                    if row.get(column) == ""
                ]

                if empty_fields:
                    add_issue(
                        result,
                        "empty_field_values",
                        f"Empty fields near CSV line "
                        f"{row_number}: {empty_fields}"
                    )
                    continue

                source = row["title_source"]

                # Check language consistency.
                if row["language"] != dataset_config["language"]:
                    add_issue(
                        result,
                        "language_mismatch",
                        f"Unexpected language near CSV line "
                        f"{row_number}."
                    )

                # Check Wikipedia edition identifier.
                if row["wiki_id"] != dataset_config["wiki_id"]:
                    add_issue(
                        result,
                        "wiki_id_mismatch",
                        f"Unexpected wiki_id near CSV line "
                        f"{row_number}."
                    )

                # Check title_display transformation.
                expected_display = source.replace("_", " ")

                if row["title_display"] != expected_display:
                    add_issue(
                        result,
                        "title_display_mismatch",
                        f"title_display inconsistency near "
                        f"CSV line {row_number}."
                    )

                # Check Unicode NFC normalization.
                expected_nfc = unicodedata.normalize(
                    "NFC",
                    source
                )

                if row["title_nfc"] != expected_nfc:
                    add_issue(
                        result,
                        "nfc_mismatch",
                        f"title_nfc inconsistency near "
                        f"CSV line {row_number}."
                    )

                # Check Unicode code-point length.
                try:
                    expected_length = len(source)
                    actual_length = int(row["codepoint_length"])

                    if actual_length != expected_length:
                        add_issue(
                            result,
                            "codepoint_length_mismatch",
                            f"Code-point length inconsistency near "
                            f"CSV line {row_number}."
                        )

                except (ValueError, TypeError):
                    add_issue(
                        result,
                        "invalid_codepoint_length",
                        f"Invalid codepoint_length near "
                        f"CSV line {row_number}."
                    )

                # Check UTF-8 byte length.
                try:
                    expected_bytes = len(source.encode("utf-8"))
                    actual_bytes = int(row["utf8_bytes"])

                    if actual_bytes != expected_bytes:
                        add_issue(
                            result,
                            "utf8_bytes_mismatch",
                            f"UTF-8 byte inconsistency near "
                            f"CSV line {row_number}."
                        )

                except (ValueError, TypeError, UnicodeEncodeError):
                    add_issue(
                        result,
                        "invalid_utf8_bytes",
                        f"Invalid UTF-8 byte count near "
                        f"CSV line {row_number}."
                    )

    except (OSError, UnicodeDecodeError, csv.Error) as error:

        add_issue(
            result,
            "csv_read_error",
            str(error)
        )

    # Compare actual record count with manifest.
    result["row_count_match"] = (
        expected_rows is not None
        and result["actual_rows"] == expected_rows
    )

    if not result["row_count_match"]:
        add_issue(
            result,
            "row_count_mismatch",
            f"Expected {expected_rows} records but read "
            f"{result['actual_rows']} records."
        )

    # Optional checksum verification.
    if verify_sha256 and file_path.is_file():

        expected_hash = manifest_record.get("sha256")

        if expected_hash:

            try:
                actual_hash = calculate_sha256(file_path)

                result["sha256_checked"] = True

                result["sha256_match"] = (
                    actual_hash.lower() == expected_hash.lower()
                )

                if not result["sha256_match"]:
                    add_issue(
                        result,
                        "sha256_mismatch",
                        "Dataset SHA-256 differs from manifest."
                    )

            except OSError as error:
                add_issue(
                    result,
                    "hash_error",
                    str(error)
                )

    # Duplicates are deliberately handled separately.
    result["duplicates_checked"] = False

    # Final status.
    if result["issue_counts"]:
        result["status"] = "FAIL"
    elif result["warnings"]:
        result["status"] = "PASS_WITH_WARNINGS"
    else:
        result["status"] = "PASS"

    return result


def save_reports(
    results: list[dict[str, Any]],
    json_path: Path,
    csv_path: Path
) -> None:

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "verification_method": "streaming_csv_audit",
        "duplicate_detection_performed": False,
        "datasets": results
    }

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    fields = [
        "dataset",
        "filename",
        "language",
        "status",
        "expected_rows",
        "actual_rows",
        "expected_file_size_bytes",
        "actual_file_size_bytes",
        "row_count_match",
        "schema_match",
        "sha256_checked",
        "sha256_match",
        "duplicates_checked"
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                field: result.get(field)
                for field in fields
            })


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Stream and validate Wikipedia title CSV datasets."
    )

    parser.add_argument(
        "--config",
        default="config/experiment.json",
        help="Path to experiment configuration."
    )

    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Calculate and compare complete file SHA-256."
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset key: tulu, kannada or english."
    )

    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config

    try:
        with config_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            config = json.load(file)

    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: Cannot read configuration: {error}")
        return 2

    manifest_path = (
        PROJECT_ROOT / config["paths"]["manifest"]
    )

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            manifest = json.load(file)

    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: Cannot read manifest: {error}")
        print(f"Expected location: {manifest_path}")
        return 2

    manifest_records = {
        item["filename"]: item
        for item in manifest.get("records", [])
    }

    datasets = config["datasets"]

    if args.dataset:
        if args.dataset not in datasets:
            print(f"ERROR: Unknown dataset: {args.dataset}")
            return 2

        datasets = {
            args.dataset: datasets[args.dataset]
        }

    required_columns = config["dataset_schema"][
        "required_columns"
    ]

    results = []

    print("\nWIKIPEDIA DATASET AUDIT")
    print("=" * 60)

    for dataset_key, dataset_config in datasets.items():

        print(f"\nAuditing: {dataset_key}")

        manifest_record = manifest_records.get(
            dataset_config["filename"]
        )

        result = audit_dataset(
            dataset_key=dataset_key,
            dataset_config=dataset_config,
            manifest_record=manifest_record,
            required_columns=required_columns,
            verify_sha256=args.verify_sha256
        )

        results.append(result)

        print(f"Status: {result['status']}")
        print(
            f"Expected rows: {result['expected_rows']}"
        )
        print(
            f"Actual rows:   {result['actual_rows']:,}"
        )

        for issue, count in result["issue_counts"].items():
            print(f"ERROR - {issue}: {count}")

        for warning in result["warnings"]:
            print(f"WARNING - {warning}")

    json_output = (
        PROJECT_ROOT / config["paths"]["audit_report_json"]
    )

    csv_output = (
        PROJECT_ROOT / config["paths"]["audit_report_csv"]
    )

    save_reports(
        results,
        json_output,
        csv_output
    )

    print("\n" + "=" * 60)
    print("DATASET AUDIT SUMMARY")
    print("=" * 60)

    for result in results:
        print(
            f"{result['dataset']:12} : {result['status']}"
        )

    print("\nReports generated:")
    print(json_output)
    print(csv_output)

    all_passed = all(
        result["status"] in ["PASS", "PASS_WITH_WARNINGS"]
        for result in results
    )

    if all_passed:
        print("\nDATASET STRUCTURAL AUDIT COMPLETED.")
        return 0

    print("\nDATASET AUDIT FAILED OR FILES ARE MISSING.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())