
import csv
import json
import random
import zlib

from pathlib import Path

from src.dataset_loader import load_corpus, calculate_sha256


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.json"

SAMPLE_MANIFEST_PATH = (
    PROJECT_ROOT / "results" / "tables"
    / "corpus_sample_manifest.csv"
)

WORKLOAD_MANIFEST_PATH = (
    PROJECT_ROOT / "results" / "tables"
    / "query_workload_manifest.csv"
)

WORKLOAD_COLUMNS = [
    "query_id",
    "query",
    "expected_found"
]


def read_csv_records(path):
    """Read CSV records using UTF-8 encoding."""

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def calculate_expected_seed(
    base_seed,
    corpus_id,
    sample_size,
    query_count
):
    """Recalculate the deterministic workload seed independently."""

    identity = (
        f"{corpus_id}|{sample_size}|{query_count}|successful"
    )

    checksum = zlib.crc32(identity.encode("utf-8"))

    return (base_seed + checksum) % (2 ** 32)


def validate_one_workload(
    workload,
    sample,
    titles,
    title_set,
    search_column
):
    """Validate one successful-query workload."""

    workload_id = workload["workload_id"]

    query_count = int(workload["query_count"])
    base_seed = int(workload["base_seed"])
    sample_size = int(workload["sample_size"])

    # Verify source corpus metadata.
    if workload["source_sample_file"] != sample["sample_file"]:
        raise ValueError(
            f"{workload_id}: Source sample file mismatch"
        )

    if workload["source_sample_sha256"] != sample["sample_sha256"]:
        raise ValueError(
            f"{workload_id}: Source sample checksum mismatch"
        )

    if workload["search_column"] != search_column:
        raise ValueError(
            f"{workload_id}: Search column mismatch"
        )

    if workload["workload_type"] != "successful":
        raise ValueError(
            f"{workload_id}: Incorrect workload type"
        )

    if workload["query_sampling_method"] != (
        "uniform_random_sampling_with_replacement"
    ):
        raise ValueError(
            f"{workload_id}: Incorrect sampling method"
        )

    # Verify the deterministic random seed.
    expected_seed = calculate_expected_seed(
        base_seed,
        sample["corpus_id"],
        sample_size,
        query_count
    )

    actual_seed = int(workload["derived_seed"])

    if actual_seed != expected_seed:
        raise ValueError(
            f"{workload_id}: Random seed mismatch"
        )

    # Verify workload file and checksum.
    workload_path = PROJECT_ROOT / workload["workload_file"]

    if not workload_path.is_file():
        raise ValueError(
            f"{workload_id}: Workload file not found"
        )

    actual_sha256 = calculate_sha256(workload_path)

    if actual_sha256 != workload["workload_sha256"]:
        raise ValueError(
            f"{workload_id}: Workload SHA-256 mismatch"
        )

    # Recreate the expected query sequence.
    # This checks reproducibility using the recorded seed.
    rng = random.Random(expected_seed)

    expected_queries = iter(
        rng.choices(titles, k=query_count)
    )

    unique_queries = set()
    records_checked = 0

    with workload_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != WORKLOAD_COLUMNS:
            raise ValueError(
                f"{workload_id}: Invalid CSV header"
            )

        for expected_id, row in enumerate(reader, start=1):

            records_checked += 1

            if int(row["query_id"]) != expected_id:
                raise ValueError(
                    f"{workload_id}: Invalid query ID "
                    f"at record {expected_id}"
                )

            query = row["query"]

            if not query:
                raise ValueError(
                    f"{workload_id}: Empty query found"
                )

            if row["expected_found"] != "true":
                raise ValueError(
                    f"{workload_id}: Expected result is not true"
                )

            # Every successful query must belong to the corpus.
            if query not in title_set:
                raise ValueError(
                    f"{workload_id}: Query not found in corpus: "
                    f"{query!r}"
                )

            # Verify exact reproducibility.
            expected_query = next(expected_queries, None)

            if query != expected_query:
                raise ValueError(
                    f"{workload_id}: Query sequence is not "
                    f"reproducible at record {expected_id}"
                )

            unique_queries.add(query)

    if records_checked != query_count:
        raise ValueError(
            f"{workload_id}: Record count mismatch. "
            f"Expected {query_count}, got {records_checked}"
        )

    unique_count = len(unique_queries)

    repeated_count = records_checked - unique_count

    if unique_count != int(workload["unique_query_count"]):
        raise ValueError(
            f"{workload_id}: Unique query count mismatch"
        )

    if repeated_count != int(
        workload["repeated_query_occurrences"]
    ):
        raise ValueError(
            f"{workload_id}: Repeated query count mismatch"
        )

    return {
        "records": records_checked,
        "unique": unique_count,
        "repeated": repeated_count
    }


def main():
    """Validate every generated successful-query workload."""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    search_column = config["dataset_schema"]["search_column"]

    query_counts = [
        int(value)
        for value in config["query_workloads"]["query_counts"]
    ]

    base_seed = int(config["query_workloads"]["seed"])

    sample_rows = read_csv_records(SAMPLE_MANIFEST_PATH)
    workload_rows = read_csv_records(WORKLOAD_MANIFEST_PATH)

    expected_workload_count = len(sample_rows) * len(query_counts)

    print("=" * 65)
    print("SUCCESSFUL QUERY WORKLOAD VALIDATION")
    print("=" * 65)

    print("Corpus samples:", len(sample_rows))
    print("Expected workload files:", expected_workload_count)
    print("Actual workload entries:", len(workload_rows))

    if len(workload_rows) != expected_workload_count:
        raise ValueError("Unexpected number of workload manifest entries")

    workload_ids = [
        row["workload_id"] for row in workload_rows
    ]

    if len(workload_ids) != len(set(workload_ids)):
        raise ValueError("Duplicate workload IDs found")

    total_records = 0
    validated_workloads = 0

    for sample in sample_rows:

        corpus_id = sample["corpus_id"]
        sample_size = int(sample["sample_size"])

        if sample["status"] != "PASS":
            raise ValueError(
                f"Source sample failed validation: "
                f"{sample['sample_file']}"
            )

        titles = load_corpus(
            sample_path=PROJECT_ROOT / sample["sample_file"],
            expected_count=sample_size,
            expected_sha256=sample["sample_sha256"],
            search_column=search_column
        )

        title_set = set(titles)

        if len(title_set) != sample_size:
            raise ValueError(
                f"{corpus_id} n={sample_size}: "
                "Source corpus is not unique"
            )

        matching_workloads = [
            row for row in workload_rows
            if row["corpus_id"] == corpus_id
            and int(row["sample_size"]) == sample_size
        ]

        if len(matching_workloads) != len(query_counts):
            raise ValueError(
                f"{corpus_id} n={sample_size}: "
                "Incorrect number of workloads"
            )

        observed_counts = sorted(
            int(row["query_count"])
            for row in matching_workloads
        )

        if observed_counts != sorted(query_counts):
            raise ValueError(
                f"{corpus_id} n={sample_size}: "
                "Query-count configuration mismatch"
            )

        for workload in matching_workloads:

            if int(workload["base_seed"]) != base_seed:
                raise ValueError(
                    f"{workload['workload_id']}: Base seed mismatch"
                )

            result = validate_one_workload(
                workload=workload,
                sample=sample,
                titles=titles,
                title_set=title_set,
                search_column=search_column
            )

            validated_workloads += 1
            total_records += result["records"]

            print(
                f"PASS: {workload['workload_id']} | "
                f"Records: {result['records']:,} | "
                f"Unique: {result['unique']:,} | "
                f"Repeated: {result['repeated']:,} | "
                f"Membership: PASS | "
                f"SHA-256: PASS | "
                f"Reproducibility: PASS"
            )

        del title_set
        del titles

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print("Corpus samples:", len(sample_rows))
    print("Validated workload files:", validated_workloads)
    print("Validated query records:", f"{total_records:,}")

    print("\nMembership validation: PASS")
    print("Record-count validation: PASS")
    print("SHA-256 validation: PASS")
    print("Seed validation: PASS")
    print("Reproducibility validation: PASS")

    print("\nALL SUCCESSFUL QUERY WORKLOADS VALIDATED")


if __name__ == "__main__":
    main()
