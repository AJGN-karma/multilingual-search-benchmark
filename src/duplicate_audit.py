
"""
Independent duplicate detection for multilingual Wikipedia titles.

Search key:
    title_nfc

Comparison:
    Exact, case-sensitive Unicode string equality.
    SQLite BINARY collation.

Memory strategy:
    Disk-backed SQLite database.
    CSV input is processed in batches.
    No corpus-wide Python set is created.

Outputs:
    results/raw/duplicate_audit/<language>.json
    results/raw/duplicate_audit/<language>_examples.csv

    results/tables/duplicate_audit.json
    results/tables/duplicate_audit.csv
    results/tables/duplicate_examples.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import tempfile
import time
import unicodedata

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BATCH_SIZE = 10_000
PROGRESS_INTERVAL = 1_000_000
MAX_EXAMPLES = 100

SUMMARY_FIELDS = [
    "corpus_id",
    "language",
    "source_file",
    "total_records",
    "unique_nfc_titles",
    "duplicate_occurrences",
    "distinct_duplicate_titles",
    "maximum_occurrences_of_one_title",
    "manifest_expected_rows",
    "row_count_match",
    "duplicate_policy",
    "status",
    "processing_seconds"
]

EXAMPLE_FIELDS = [
    "corpus_id",
    "language",
    "title_nfc",
    "occurrences"
]


UPSERT_SQL = """
INSERT INTO title_counts(title, occurrences)
SELECT title, COUNT(*)
FROM batch_titles
WHERE 1
GROUP BY title
ON CONFLICT(title)
DO UPDATE SET
    occurrences =
        title_counts.occurrences + excluded.occurrences
"""


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def flush_batch(
    connection: sqlite3.Connection,
    batch: list[str]
) -> None:
    """
    Aggregate the current batch and update SQLite counts.

    Repeated titles within this batch are counted using
    GROUP BY. Titles already encountered in previous
    batches are handled by the UPSERT operation.
    """

    if not batch:
        return

    connection.execute("DELETE FROM batch_titles")

    connection.executemany(
        "INSERT INTO batch_titles(title) VALUES (?)",
        ((title,) for title in batch)
    )

    connection.execute(UPSERT_SQL)

    connection.commit()

    batch.clear()


def read_manifest_record(
    manifest: dict,
    filename: str
) -> dict:

    for item in manifest.get("records", []):

        if item.get("filename") == filename:
            return item

    raise ValueError(
        f"Dataset {filename} not found in manifest."
    )


def process_dataset(
    dataset_key: str,
    dataset: dict,
    expected_rows: int,
    required_columns: list[str],
    batch_size: int
) -> tuple[dict, list[dict]]:

    source_path = ROOT / dataset["relative_path"]

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    temp_root = ROOT / "data/processed"
    temp_root.mkdir(parents=True, exist_ok=True)

    free_bytes = shutil.disk_usage(ROOT).free

    print("\n" + "=" * 65)
    print(f"DUPLICATE AUDIT: {dataset['language']}")
    print("=" * 65)

    print(
        f"Available disk space: "
        f"{free_bytes / (1024**3):.2f} GiB"
    )

    if free_bytes < 4 * (1024**3):
        raise RuntimeError(
            "Less than 4 GiB of free disk space remains. "
            "Stop before creating the temporary SQLite database."
        )

    print(f"Input: {source_path}")
    print(f"Batch size: {batch_size:,}")

    started = time.monotonic()

    total_records = 0
    last_progress = 0

    batch: list[str] = []

    with tempfile.TemporaryDirectory(
        prefix=f"duplicate_{dataset_key}_",
        dir=temp_root
    ) as temporary_directory:

        database_path = (
            Path(temporary_directory) / "duplicate_audit.sqlite"
        )

        connection = sqlite3.connect(str(database_path))

        try:
            # Restrict SQLite's page cache to approximately 32 MiB.
            connection.execute("PRAGMA cache_size = -32768")

            # Use file-backed temporary structures when needed.
            connection.execute("PRAGMA temp_store = FILE")

            connection.execute("PRAGMA synchronous = NORMAL")

            # Each distinct normalized title has one row.
            connection.execute("""
                CREATE TABLE title_counts (
                    title TEXT COLLATE BINARY PRIMARY KEY,
                    occurrences INTEGER NOT NULL
                        CHECK(occurrences > 0)
                ) WITHOUT ROWID
            """)

            # Holds only the currently processed CSV batch.
            connection.execute("""
                CREATE TABLE batch_titles (
                    title TEXT COLLATE BINARY NOT NULL
                )
            """)

            connection.commit()

            with source_path.open(
                "r",
                encoding="utf-8",
                newline=""
            ) as source_file:

                reader = csv.DictReader(source_file)

                if reader.fieldnames != required_columns:
                    raise ValueError(
                        f"CSV schema mismatch for {dataset_key}."
                    )

                for row in reader:

                    if None in row:
                        raise ValueError(
                            f"Unexpected extra CSV fields near "
                            f"line {reader.line_num}."
                        )

                    title = row.get("title_nfc")

                    if title is None or title == "":
                        raise ValueError(
                            f"Empty or missing title_nfc near "
                            f"line {reader.line_num}."
                        )

                    # Do not silently normalize the input here.
                    # It must already satisfy the established policy.
                    if not unicodedata.is_normalized("NFC", title):
                        raise ValueError(
                            f"Non-NFC title found near "
                            f"line {reader.line_num}."
                        )

                    batch.append(title)
                    total_records += 1

                    if len(batch) >= batch_size:
                        flush_batch(connection, batch)

                    # Print progress at million-record intervals.
                    if total_records - last_progress >= PROGRESS_INTERVAL:

                        flush_batch(connection, batch)

                        elapsed = time.monotonic() - started

                        rate = total_records / elapsed

                        print(
                            f"Processed {total_records:,} records "
                            f"in {elapsed:.1f} seconds "
                            f"({rate:,.0f} records/second)"
                        )

                        last_progress = total_records

            # Process the last partial batch.
            flush_batch(connection, batch)

            # Count unique normalized search keys.
            unique_titles = connection.execute(
                "SELECT COUNT(*) FROM title_counts"
            ).fetchone()[0]

            # Count different titles appearing more than once.
            distinct_duplicates = connection.execute("""
                SELECT COUNT(*)
                FROM title_counts
                WHERE occurrences > 1
            """).fetchone()[0]

            # Count all occurrences beyond the first copy.
            duplicate_occurrences = (
                total_records - unique_titles
            )

            max_occurrences = connection.execute("""
                SELECT COALESCE(MAX(occurrences), 0)
                FROM title_counts
            """).fetchone()[0]

            # Validate the duplicate arithmetic.
            duplicate_sum = connection.execute("""
                SELECT COALESCE(SUM(occurrences - 1), 0)
                FROM title_counts
                WHERE occurrences > 1
            """).fetchone()[0]

            if duplicate_sum != duplicate_occurrences:
                raise RuntimeError(
                    "Duplicate count consistency check failed."
                )

            # Retain at most 100 examples for diagnostics.
            examples = connection.execute("""
                SELECT title, occurrences
                FROM title_counts
                WHERE occurrences > 1
                ORDER BY occurrences DESC, title COLLATE BINARY
                LIMIT ?
            """, (MAX_EXAMPLES,)).fetchall()

        finally:
            connection.close()

    elapsed = time.monotonic() - started

    row_count_match = total_records == expected_rows

    result = {
        "corpus_id": dataset_key,
        "language": dataset["language"],
        "source_file": dataset["filename"],
        "total_records": total_records,
        "unique_nfc_titles": unique_titles,
        "duplicate_occurrences": duplicate_occurrences,
        "distinct_duplicate_titles": distinct_duplicates,
        "maximum_occurrences_of_one_title": max_occurrences,
        "manifest_expected_rows": expected_rows,
        "row_count_match": row_count_match,
        "duplicate_policy": (
            "Exact case-sensitive equality of title_nfc "
            "using SQLite BINARY collation"
        ),
        "status": "PASS" if row_count_match else "FAIL",
        "processing_seconds": round(elapsed, 3)
    }

    example_rows = [
        {
            "corpus_id": dataset_key,
            "language": dataset["language"],
            "title_nfc": title,
            "occurrences": count
        }
        for title, count in examples
    ]

    # Store individual results so that running one language
    # never overwrites the previous languages' results.
    individual_directory = (
        ROOT / "results/raw/duplicate_audit"
    )

    individual_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    write_json(
        individual_directory / f"{dataset_key}.json",
        result
    )

    examples_path = (
        individual_directory /
        f"{dataset_key}_examples.csv"
    )

    with examples_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=EXAMPLE_FIELDS
        )

        writer.writeheader()
        writer.writerows(example_rows)

    print("\nRESULT")
    print("-" * 65)
    print(f"Total records:             {total_records:,}")
    print(f"Unique NFC titles:         {unique_titles:,}")
    print(f"Duplicate occurrences:    {duplicate_occurrences:,}")
    print(f"Distinct duplicate titles:{distinct_duplicates:>11,}")
    print(f"Maximum title frequency:  {max_occurrences:,}")
    print(f"Manifest count match:     {row_count_match}")
    print(f"Processing time:          {elapsed:.2f} seconds")
    print(f"Audit status:             {result['status']}")

    return result, example_rows


def combine_available_reports(
    dataset_order: list[str]
) -> None:

    individual_directory = (
        ROOT / "results/raw/duplicate_audit"
    )

    summary_rows = []
    all_examples = []

    for dataset_key in dataset_order:

        report_path = (
            individual_directory / f"{dataset_key}.json"
        )

        if not report_path.exists():
            continue

        result = load_json(report_path)

        summary_rows.append({
            field: result.get(field)
            for field in SUMMARY_FIELDS
        })

        examples_path = (
            individual_directory /
            f"{dataset_key}_examples.csv"
        )

        if examples_path.exists():

            with examples_path.open(
                "r",
                encoding="utf-8",
                newline=""
            ) as file:

                all_examples.extend(
                    csv.DictReader(file)
                )

    output_directory = ROOT / "results/tables"

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    generated = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "duplicate_policy": (
            "Exact case-sensitive title_nfc equality "
            "after Unicode NFC normalization"
        ),
        "method": "Disk-backed SQLite aggregation",
        "datasets_audited": len(summary_rows),
        "reports": summary_rows
    }

    write_json(
        output_directory / "duplicate_audit.json",
        generated
    )

    with (
        output_directory / "duplicate_audit.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    with (
        output_directory / "duplicate_examples.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=EXAMPLE_FIELDS
        )

        writer.writeheader()
        writer.writerows(all_examples)

    print("\nCOMBINED REPORTS UPDATED")
    print("Datasets audited:", len(summary_rows))
    print(output_directory / "duplicate_audit.csv")
    print(output_directory / "duplicate_audit.json")
    print(output_directory / "duplicate_examples.csv")


def main():

    parser = argparse.ArgumentParser(
        description="Disk-backed multilingual title duplicate audit."
    )

    parser.add_argument(
        "--dataset",
        choices=["tulu", "kannada", "english", "all"],
        default="all"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )

    args = parser.parse_args()

    if args.batch_size < 1000:
        parser.error("Batch size must be at least 1000.")

    config = load_json(
        ROOT / "config/experiment.json"
    )

    manifest = load_json(
        ROOT / config["paths"]["manifest"]
    )

    required_columns = config["dataset_schema"][
        "required_columns"
    ]

    datasets = config["datasets"]

    dataset_order = ["tulu", "kannada", "english"]

    selected = (
        dataset_order
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset_key in selected:

        dataset = datasets[dataset_key]

        manifest_record = read_manifest_record(
            manifest,
            dataset["filename"]
        )

        expected_rows = manifest_record["row_count"]

        result, _ = process_dataset(
            dataset_key,
            dataset,
            expected_rows,
            required_columns,
            args.batch_size
        )

        combine_available_reports(dataset_order)

        if result["status"] != "PASS":
            print(
                "\nAUDIT FAILED: Actual record count "
                "does not match manifest."
            )
            return 1

    print("\nDUPLICATE AUDIT EXECUTION COMPLETED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
