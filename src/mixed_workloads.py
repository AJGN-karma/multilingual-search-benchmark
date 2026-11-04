
import csv
import random
import zlib

from pathlib import Path

from src.dataset_loader import calculate_sha256
from src.query_workloads import (
    load_experiment_config,
    load_sample_manifest,
    update_combined_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORKLOAD_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "query_workload_manifest.csv"
)

OUTPUT_COLUMNS = [
    "query_id",
    "query",
    "expected_found",
]


def derive_mixed_seed(
    base_seed: int,
    corpus_id: str,
    sample_size: int,
    query_count: int,
) -> int:
    """Create a reproducible seed for each mixed workload."""

    identity = (
        f"{corpus_id}|{sample_size}|{query_count}|mixed"
    )

    checksum = zlib.crc32(identity.encode("utf-8"))

    return (base_seed + checksum) % (2 ** 32)


def load_workload_manifest() -> list[dict]:
    """Read existing successful and unsuccessful workload entries."""

    if not WORKLOAD_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Query workload manifest does not exist."
        )

    with WORKLOAD_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def find_source_workload(
    manifest_rows: list[dict],
    corpus_id: str,
    sample_size: int,
    query_count: int,
    workload_type: str,
) -> dict:
    """Find exactly one matching source workload."""

    matches = [
        row for row in manifest_rows
        if row["corpus_id"] == corpus_id
        and int(row["sample_size"]) == sample_size
        and int(row["query_count"]) == query_count
        and row["workload_type"] == workload_type
        and row["status"] == "PASS"
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {workload_type} workload for "
            f"{corpus_id}, n={sample_size}, q={query_count}; "
            f"found {len(matches)}."
        )

    return matches[0]


def read_query_prefix(
    workload: dict,
    expected_result: str,
    prefix_count: int,
) -> list[tuple[str, str]]:
    """
    Read the first prefix_count queries from a validated workload.

    Verify its checksum and CSV structure before using its records.
    """

    path = PROJECT_ROOT / workload["workload_file"]

    if not path.is_file():
        raise FileNotFoundError(
            f"Source workload missing: {path}"
        )

    actual_hash = calculate_sha256(path)

    if actual_hash != workload["workload_sha256"]:
        raise ValueError(
            f"SHA-256 mismatch: {workload['workload_file']}"
        )

    if int(workload["query_count"]) < prefix_count:
        raise ValueError(
            "Source workload does not contain enough records."
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != OUTPUT_COLUMNS:
            raise ValueError(
                f"Unexpected CSV columns: {path.name}"
            )

        for expected_id, row in enumerate(
            reader,
            start=1,
        ):

            if expected_id > prefix_count:
                break

            if int(row["query_id"]) != expected_id:
                raise ValueError(
                    f"Invalid query ID in {path.name}"
                )

            if row["expected_found"] != expected_result:
                raise ValueError(
                    f"Unexpected search result label in {path.name}"
                )

            if not row["query"]:
                raise ValueError(
                    f"Empty query found in {path.name}"
                )

            records.append((
                row["query"],
                expected_result,
            ))

    if len(records) != prefix_count:
        raise ValueError(
            f"Insufficient records in {path.name}"
        )

    return records


def write_mixed_workload(
    records: list[tuple[str, str]],
    corpus_id: str,
    sample_size: int,
    query_count: int,
    seed: int,
) -> dict:
    """Write the mixed workload and calculate its checksum."""

    output_directory = (
        PROJECT_ROOT
        / "data"
        / "workloads"
        / "mixed"
        / f"{corpus_id}_n{sample_size}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"mixed_q{query_count}_seed{seed}.csv"
    )

    output_path = output_directory / filename

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file,
            lineterminator="\n",
        )

        writer.writerow(OUTPUT_COLUMNS)

        for query_id, (query, expected_result) in enumerate(
            records,
            start=1,
        ):

            writer.writerow([
                query_id,
                query,
                expected_result,
            ])

    unique_count = len({
        query for query, _ in records
    })

    return {
        "workload_file": str(
            output_path.relative_to(PROJECT_ROOT)
        ),
        "workload_sha256": calculate_sha256(output_path),
        "query_count": len(records),
        "unique_query_count": unique_count,
        "repeated_query_occurrences": (
            len(records) - unique_count
        ),
    }


def main():
    """Generate balanced mixed workloads for every corpus sample."""

    config = load_experiment_config()

    base_seed = int(config["query_workloads"]["seed"])

    query_counts = [
        int(value)
        for value in config["query_workloads"]["query_counts"]
    ]

    mixed_ratio = float(
        config["query_workloads"]["mixed_success_ratio"]
    )

    if mixed_ratio != 0.5:
        raise ValueError(
            "This implementation requires a 50% success ratio."
        )

    if not config["query_workloads"]["mixed_queries"]:
        raise ValueError(
            "Mixed query workloads are disabled in the configuration."
        )

    sample_rows = load_sample_manifest()
    source_rows = load_workload_manifest()

    new_manifest_rows = []

    total_records = 0

    print("=" * 65)
    print("MIXED QUERY WORKLOAD GENERATION")
    print("=" * 65)

    print("Base seed:", base_seed)
    print("Success ratio:", "50%")
    print("Unsuccessful ratio:", "50%")
    print("Query counts:", query_counts)
    print("Corpus samples:", len(sample_rows))

    for sample in sample_rows:

        corpus_id = sample["corpus_id"]
        language = sample["language"]
        sample_size = int(sample["sample_size"])

        print("\n" + "-" * 65)
        print(f"Language: {language}")
        print(f"Corpus ID: {corpus_id}")
        print(f"Sample size: {sample_size:,}")

        for query_count in query_counts:

            if query_count % 2 != 0:
                raise ValueError(
                    "Mixed query counts must be even."
                )

            half_count = query_count // 2

            # Locate the already validated source workloads.
            positive_source = find_source_workload(
                source_rows,
                corpus_id,
                sample_size,
                query_count,
                "successful",
            )

            negative_source = find_source_workload(
                source_rows,
                corpus_id,
                sample_size,
                query_count,
                "unsuccessful",
            )

            # Select equal numbers from both source workloads.
            positive_records = read_query_prefix(
                positive_source,
                "true",
                half_count,
            )

            negative_records = read_query_prefix(
                negative_source,
                "false",
                half_count,
            )

            mixed_records = (
                positive_records + negative_records
            )

            # Randomize the order using a separate deterministic seed.
            derived_seed = derive_mixed_seed(
                base_seed,
                corpus_id,
                sample_size,
                query_count,
            )

            rng = random.Random(derived_seed)
            rng.shuffle(mixed_records)

            # Check the exact 50:50 distribution.
            success_count = sum(
                result == "true"
                for _, result in mixed_records
            )

            unsuccessful_count = sum(
                result == "false"
                for _, result in mixed_records
            )

            if success_count != half_count:
                raise RuntimeError(
                    "Incorrect successful-query count."
                )

            if unsuccessful_count != half_count:
                raise RuntimeError(
                    "Incorrect unsuccessful-query count."
                )

            result = write_mixed_workload(
                records=mixed_records,
                corpus_id=corpus_id,
                sample_size=sample_size,
                query_count=query_count,
                seed=derived_seed,
            )

            workload_id = (
                f"{corpus_id}_n{sample_size}"
                f"_mixed_q{query_count}"
            )

            manifest_row = {
                "workload_id": workload_id,
                "corpus_id": corpus_id,
                "language": language,
                "sample_size": sample_size,
                "source_sample_file": sample["sample_file"],
                "source_sample_sha256": sample["sample_sha256"],
                "search_column": config["dataset_schema"]["search_column"],
                "workload_type": "mixed",
                "query_count": result["query_count"],
                "unique_query_count": result["unique_query_count"],
                "repeated_query_occurrences": (
                    result["repeated_query_occurrences"]
                ),
                "base_seed": base_seed,
                "derived_seed": derived_seed,
                "query_sampling_method": (
                    "balanced_merge_of_validated_successful_and_"
                    "unsuccessful_workloads_then_deterministic_shuffle"
                ),
                "expected_found": "50%_true_50%_false",
                "workload_file": result["workload_file"],
                "workload_sha256": result["workload_sha256"],
                "status": "PASS",
            }

            new_manifest_rows.append(manifest_row)

            total_records += query_count

            print(
                f"PASS: {query_count:,} total | "
                f"Successful: {success_count:,} | "
                f"Unsuccessful: {unsuccessful_count:,} | "
                f"Unique: {result['unique_query_count']:,} | "
                f"Repeated: {result['repeated_query_occurrences']:,}"
            )

    # Preserve the existing successful and unsuccessful manifest entries.
    update_combined_manifest(new_manifest_rows)

    print("\n" + "=" * 65)
    print("MIXED QUERY GENERATION SUMMARY")
    print("=" * 65)

    print("Corpus samples:", len(sample_rows))
    print("New workload files:", len(new_manifest_rows))
    print("Total mixed query records:", f"{total_records:,}")
    print("Successful share:", "50%")
    print("Unsuccessful share:", "50%")

    print("\nCombined workload manifest updated.")
    print("\nMIXED QUERY WORKLOAD GENERATION COMPLETED")


if __name__ == "__main__":
    main()
