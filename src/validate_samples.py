
"""
Generic streaming validator for nested corpus samples.

Supports:
    Tulu
    Kannada
    English

Checks:
    1. Individual sample report exists.
    2. Sample file exists.
    3. Expected CSV schema.
    4. Exact record count.
    5. Non-empty title values.
    6. Unicode NFC consistency.
    7. SHA-256 checksum.
    8. Nested prefix relationship.

Memory:
    Samples are processed one record at a time.
    No corpus-wide list or set is created.

Usage:
    python -m src.validate_samples --dataset tulu
    python -m src.validate_samples --dataset kannada
    python -m src.validate_samples --dataset english
    python -m src.validate_samples --dataset all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "results/raw/corpus_sampling"


def read_json(path: Path) -> dict:

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def calculate_sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def iterate_titles(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != ["title_nfc"]:
            raise ValueError(
                f"Incorrect CSV schema in {path}"
            )

        for row in reader:

            if None in row:
                raise ValueError(
                    f"Unexpected additional CSV fields "
                    f"in {path}"
                )

            title = row.get("title_nfc")

            if title is None or title == "":
                raise ValueError(
                    f"Empty title encountered in {path}"
                )

            if not unicodedata.is_normalized("NFC", title):
                raise ValueError(
                    f"Non-NFC title encountered in {path}"
                )

            yield title


def validate_individual_sample(
    record: dict
) -> None:

    sample_path = ROOT / record["sample_file"]

    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)

    expected_count = int(record["sample_size"])

    if (
        int(record["sample_unique_title_count"])
        != expected_count
    ):
        raise AssertionError(
            "Recorded unique-title count does not "
            "match sample size."
        )

    # Independently verify the actual file hash.
    actual_hash = calculate_sha256(sample_path)

    if actual_hash.lower() != record["sample_sha256"].lower():
        raise AssertionError(
            f"SHA-256 mismatch: {sample_path}"
        )

    # Stream through the CSV to validate its contents.
    actual_count = 0

    for _ in iterate_titles(sample_path):
        actual_count += 1

    if actual_count != expected_count:
        raise AssertionError(
            f"Expected {expected_count:,} records, "
            f"found {actual_count:,} in {sample_path}"
        )

    print(
        f"{sample_path.name}: "
        f"{actual_count:,} records | "
        f"SHA-256 PASS | PASS"
    )


def validate_nested_pair(
    smaller_record: dict,
    larger_record: dict
) -> None:
    """
    Verify that every record in the smaller sample
    is an identical prefix of the larger sample.

    Reads both files incrementally.
    """

    smaller_path = ROOT / smaller_record["sample_file"]
    larger_path = ROOT / larger_record["sample_file"]

    smaller_size = int(smaller_record["sample_size"])
    larger_size = int(larger_record["sample_size"])

    if smaller_size > larger_size:
        raise AssertionError(
            "Smaller sample size exceeds larger sample size."
        )

    smaller_iterator = iterate_titles(smaller_path)
    larger_iterator = iterate_titles(larger_path)

    for index in range(smaller_size):

        try:
            smaller_title = next(smaller_iterator)
            larger_title = next(larger_iterator)

        except StopIteration as error:

            raise AssertionError(
                "Unexpected end of sample during nesting test."
            ) from error

        if smaller_title != larger_title:

            raise AssertionError(
                f"Nested sample mismatch at record {index + 1}: "
                f"{smaller_path.name} versus {larger_path.name}"
            )

    # Confirm the smaller sample itself has no extra records.
    try:
        next(smaller_iterator)
    except StopIteration:
        pass
    else:
        raise AssertionError(
            "Smaller sample contains unexpected extra records."
        )

    print(
        "Nested relationship PASS: "
        f"{smaller_path.name} -> {larger_path.name}"
    )


def validate_dataset(dataset_key: str) -> None:

    report_path = REPORT_DIR / f"{dataset_key}.json"

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Sampling report missing: {report_path}"
        )

    report = read_json(report_path)

    records = sorted(
        report.get("datasets", []),
        key=lambda item: int(item["sample_size"])
    )

    if not records:
        raise AssertionError(
            f"No sample records found for {dataset_key}"
        )

    print("\n" + "=" * 65)
    print(f"SAMPLE VALIDATION: {dataset_key.upper()}")
    print("=" * 65)

    for record in records:

        if record["status"] != "PASS":
            raise AssertionError(
                f"Sampling status is not PASS: {record}"
            )

        validate_individual_sample(record)

    # Verify nested prefix property between every adjacent size.
    for index in range(len(records) - 1):

        validate_nested_pair(
            records[index],
            records[index + 1]
        )

    print(
        f"\nALL {dataset_key.upper()} "
        f"SAMPLE VALIDATION TESTS PASSED"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Validate reproducible multilingual samples."
    )

    parser.add_argument(
        "--dataset",
        choices=["tulu", "kannada", "english", "all"],
        default="all"
    )

    args = parser.parse_args()

    order = ["tulu", "kannada", "english"]

    selected = (
        order if args.dataset == "all"
        else [args.dataset]
    )

    for key in selected:
        validate_dataset(key)

    print("\n" + "=" * 65)
    print("ALL SELECTED SAMPLE VALIDATIONS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
