
"""
Reproducible nested corpus sampling.

Method:
    Reservoir sampling (Algorithm R)
    Fixed random seed
    Deterministic shuffle
    Nested samples using prefixes

Input:
    data/raw/<language>_titles.csv

Search field:
    title_nfc

Output:
    data/processed/<language>/
    results/tables/corpus_sample_manifest.csv
    results/tables/corpus_sample_manifest.json

The original CSV files remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import time
import unicodedata

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "data/metadata/manifest.json"
DUPLICATE_PATH = ROOT / "results/tables/duplicate_audit.json"

INDIVIDUAL_DIR = ROOT / "results/raw/corpus_sampling"
SUMMARY_DIR = ROOT / "results/tables"

FIELDS = [
    "corpus_id",
    "language",
    "source_file",
    "source_sha256",
    "source_record_count",
    "sample_size",
    "sample_unique_title_count",
    "random_seed",
    "shuffle_seed",
    "sampling_algorithm",
    "nested_sample_group",
    "sample_file",
    "sample_file_bytes",
    "sample_sha256",
    "search_column",
    "normalization_policy",
    "status"
]


def read_json(path: Path) -> dict:

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, content: dict) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    """Calculate file checksum using limited memory."""

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_manifest_record(manifest: dict, filename: str) -> dict:

    for record in manifest.get("records", []):

        if record.get("filename") == filename:
            return record

    raise ValueError(
        f"Manifest entry not found for {filename}"
    )


def get_duplicate_records() -> dict:

    if not DUPLICATE_PATH.exists():
        raise FileNotFoundError(
            "Duplicate audit report missing."
        )

    report = read_json(DUPLICATE_PATH)

    records = {
        item["corpus_id"]: item
        for item in report.get("reports", [])
    }

    return records


def validate_duplicate_audit(
    dataset_key: str,
    duplicate_reports: dict,
    expected_rows: int
) -> None:

    if dataset_key not in duplicate_reports:
        raise ValueError(
            f"No duplicate audit exists for {dataset_key}"
        )

    report = duplicate_reports[dataset_key]

    if report["status"] != "PASS":
        raise ValueError(
            f"Duplicate audit failed for {dataset_key}"
        )

    if int(report["total_records"]) != expected_rows:
        raise ValueError(
            "Duplicate audit record count mismatch."
        )

    if int(report["duplicate_occurrences"]) != 0:
        raise ValueError(
            "Duplicate titles exist. Sampling is stopped."
        )

    if int(report["unique_nfc_titles"]) != expected_rows:
        raise ValueError(
            "Unique title count does not match source records."
        )


def reservoir_sampling(
    source_path: Path,
    required_columns: list[str],
    sample_size: int,
    seed: int
) -> tuple[list[str], int]:
    """
    Algorithm R.

    Maintains only sample_size titles in memory.

    For the i-th record after the initial reservoir:
        choose j uniformly from [0, i-1]
        replace reservoir[j] if j < sample_size
    """

    rng = random.Random(seed)

    reservoir = []
    records_read = 0

    started = time.monotonic()

    with source_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != required_columns:
            raise ValueError(
                f"Unexpected CSV columns in {source_path}"
            )

        for row in reader:

            if None in row:
                raise ValueError(
                    f"Unexpected extra CSV fields near "
                    f"line {reader.line_num}"
                )

            title = row.get("title_nfc")

            if title is None or title == "":
                raise ValueError(
                    f"Empty title_nfc at line {reader.line_num}"
                )

            if not unicodedata.is_normalized("NFC", title):
                raise ValueError(
                    f"Non-NFC title at line {reader.line_num}"
                )

            records_read += 1

            if records_read <= sample_size:

                reservoir.append(title)

            else:

                position = rng.randrange(records_read)

                if position < sample_size:
                    reservoir[position] = title

            if records_read % 1_000_000 == 0:

                elapsed = time.monotonic() - started

                print(
                    f"Processed {records_read:,} source records "
                    f"in {elapsed:.1f}s"
                )

    if len(reservoir) != sample_size:
        raise RuntimeError(
            f"Expected {sample_size:,} sample records, "
            f"obtained {len(reservoir):,}"
        )

    return reservoir, records_read


def generate_sample_file(
    dataset_key: str,
    dataset: dict,
    titles: list[str],
    sample_size: int,
    source_record_count: int,
    source_hash: str,
    seed: int,
    shuffle_seed: int
) -> dict:

    output_directory = ROOT / "data/processed" / dataset_key

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"{dataset_key}_n{sample_size}_seed{seed}.csv"
    )

    output_path = output_directory / filename

    print(
        f"\nCreating {dataset['language']} sample "
        f"of {sample_size:,} titles"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["title_nfc"])

        for title in itertools.islice(titles, sample_size):
            writer.writerow([title])

    # Verify generated sample.
    actual_records = 0

    with output_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != ["title_nfc"]:
            raise RuntimeError("Incorrect sample schema.")

        for row in reader:

            title = row["title_nfc"]

            if not title:
                raise RuntimeError("Empty title in sample.")

            if not unicodedata.is_normalized("NFC", title):
                raise RuntimeError("Non-NFC title in sample.")

            actual_records += 1

    if actual_records != sample_size:
        raise RuntimeError(
            f"Sample row count mismatch: {actual_records}"
        )

    result = {
        "corpus_id": dataset_key,
        "language": dataset["language"],
        "source_file": dataset["filename"],
        "source_sha256": source_hash,
        "source_record_count": source_record_count,
        "sample_size": sample_size,
        "sample_unique_title_count": actual_records,
        "random_seed": seed,
        "shuffle_seed": shuffle_seed,
        "sampling_algorithm": (
            "Reservoir sampling Algorithm R without replacement"
        ),
        "nested_sample_group": f"{dataset_key}_seed_{seed}",
        "sample_file": str(output_path.relative_to(ROOT)),
        "sample_file_bytes": output_path.stat().st_size,
        "sample_sha256": sha256_file(output_path),
        "search_column": "title_nfc",
        "normalization_policy": (
            "Unicode NFC; exact case-sensitive equality; "
            "whitespace, punctuation and underscores preserved"
        ),
        "status": "PASS"
    }

    print("Output:", result["sample_file"])
    print("Records:", actual_records)
    print("SHA-256:", result["sample_sha256"])
    print("Status:", result["status"])

    return result


def save_individual_manifest(
    dataset_key: str,
    records: list[dict]
) -> None:

    content = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "sampling_method": (
            "Algorithm R followed by deterministic shuffle"
        ),
        "datasets": records
    }

    write_json(
        INDIVIDUAL_DIR / f"{dataset_key}.json",
        content
    )


def update_combined_manifest() -> None:

    all_records = []

    for key in ["tulu", "kannada", "english"]:

        path = INDIVIDUAL_DIR / f"{key}.json"

        if not path.exists():
            continue

        report = read_json(path)

        all_records.extend(
            report.get("datasets", [])
        )

    all_records.sort(
        key=lambda item: (
            item["corpus_id"],
            item["sample_size"]
        )
    )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    json_report = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "sampling_method": (
            "Reservoir sampling Algorithm R with nested prefixes"
        ),
        "total_sample_files": len(all_records),
        "samples": all_records
    }

    write_json(
        SUMMARY_DIR / "corpus_sample_manifest.json",
        json_report
    )

    with (
        SUMMARY_DIR / "corpus_sample_manifest.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS
        )

        writer.writeheader()

        for item in all_records:

            writer.writerow({
                field: item.get(field)
                for field in FIELDS
            })

    print("\nCombined sample manifest updated.")
    print("Total sample files:", len(all_records))


def sample_dataset(
    dataset_key: str,
    config: dict,
    manifest: dict,
    duplicate_reports: dict
) -> None:

    dataset = config["datasets"][dataset_key]

    source_path = ROOT / dataset["relative_path"]

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_record = get_manifest_record(
        manifest,
        dataset["filename"]
    )

    expected_rows = int(source_record["row_count"])

    validate_duplicate_audit(
        dataset_key,
        duplicate_reports,
        expected_rows
    )

    source_hash = source_record["sha256"]

    requested_sizes = sorted(
        set(
            int(value)
            for value in dataset["candidate_corpus_sizes"]
        )
    )

    sizes = [
        value
        for value in requested_sizes
        if value <= expected_rows
    ]

    skipped = [
        value
        for value in requested_sizes
        if value > expected_rows
    ]

    if skipped:
        print("Skipping sizes above corpus:", skipped)

    if not sizes:
        raise ValueError("No valid corpus sizes configured.")

    maximum_size = max(sizes)

    seed = int(config["sampling"]["seed"])
    shuffle_seed = seed + 1_000_003

    print("\n" + "=" * 60)
    print("CORPUS SAMPLING:", dataset["language"])
    print("=" * 60)

    print("Original records:", expected_rows)
    print("Largest requested sample:", maximum_size)
    print("Random seed:", seed)

    titles, actual_records = reservoir_sampling(
        source_path,
        config["dataset_schema"]["required_columns"],
        maximum_size,
        seed
    )

    if actual_records != expected_rows:
        raise RuntimeError(
            f"Source count changed. Expected {expected_rows}, "
            f"read {actual_records}."
        )

    # One deterministic shuffle establishes the common
    # ordering for every nested sample.
    shuffle_rng = random.Random(shuffle_seed)
    shuffle_rng.shuffle(titles)

    sample_results = []

    for size in sizes:

        result = generate_sample_file(
            dataset_key,
            dataset,
            titles,
            size,
            actual_records,
            source_hash,
            seed,
            shuffle_seed
        )

        sample_results.append(result)

    save_individual_manifest(
        dataset_key,
        sample_results
    )

    update_combined_manifest()

    print("\nSAMPLING SUMMARY")
    print("-" * 60)
    print("Language:", dataset["language"])
    print("Original records:", actual_records)
    print("Created sample sizes:", sizes)
    print("Nested sampling: YES")
    print("Status: PASS")


def main():

    parser = argparse.ArgumentParser(
        description="Generate reproducible nested corpus samples."
    )

    parser.add_argument(
        "--dataset",
        choices=["tulu", "kannada", "english", "all"],
        default="all"
    )

    args = parser.parse_args()

    config = read_json(
        ROOT / "config/experiment.json"
    )

    manifest = read_json(MANIFEST_PATH)

    duplicate_reports = get_duplicate_records()

    order = ["tulu", "kannada", "english"]

    selected = (
        order if args.dataset == "all"
        else [args.dataset]
    )

    for key in selected:

        sample_dataset(
            key,
            config,
            manifest,
            duplicate_reports
        )

    print("\n" + "=" * 60)
    print("CORPUS SAMPLING EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
