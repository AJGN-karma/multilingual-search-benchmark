
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

EXPECTED_COLUMNS = [
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


def derive_expected_seed(
    base_seed,
    corpus_id,
    sample_size,
    query_count
):
    """Independently reconstruct the negative workload seed."""

    identity = (
        f"{corpus_id}|{sample_size}|"
        f"{query_count}|unsuccessful"
    )

    checksum = zlib.crc32(identity.encode("utf-8"))

    return (base_seed + checksum) % (2 ** 32)


def recreate_expected_queries(
    titles,
    title_set,
    query_count,
    seed
):
    """
    Recreate the exact deterministic unsuccessful-query sequence.

    This independently reproduces the generation procedure
    so the saved query sequence can be checked.
    """

    rng = random.Random(seed)

    base_titles = rng.choices(
        titles,
        k=query_count
    )

    generated_set = set()

    for index, base_title in enumerate(
        base_titles,
        start=1
    ):

        marker = f"~x{index:08x}"

        candidate = base_title + marker

        collision_number = 0

        while (
            candidate in title_set
            or candidate in generated_set
        ):

            collision_number += 1

            candidate = (
                base_title
                + marker
                + f"~c{collision_number}"
            )

        generated_set.add(candidate)

        yield candidate


def validate_one_workload(
    workload,
    sample,
    titles,
    title_set,
    search_column,
    configured_base_seed
):
    """Validate one unsuccessful-query workload."""

    workload_id = workload["workload_id"]

    query_count = int(workload["query_count"])
    sample_size = int(workload["sample_size"])

    # Verify source corpus identity.
    if workload["corpus_id"] != sample["corpus_id"]:
        raise ValueError(
            f"{workload_id}: Corpus ID mismatch"
        )

    if workload["source_sample_file"] != sample["sample_file"]:
        raise ValueError(
            f"{workload_id}: Source sample file mismatch"
        )

    if workload["source_sample_sha256"] != sample["sample_sha256"]:
        raise ValueError(
            f"{workload_id}: Source sample SHA-256 mismatch"
        )

    if sample_size != int(sample["sample_size"]):
        raise ValueError(
            f"{workload_id}: Sample size mismatch"
        )

    if workload["search_column"] != search_column:
        raise ValueError(
            f"{workload_id}: Search column mismatch"
        )

    if workload["workload_type"] != "unsuccessful":
        raise ValueError(
            f"{workload_id}: Incorrect workload type"
        )

    if workload["expected_found"] != "false":
        raise ValueError(
            f"{workload_id}: Incorrect expected result"
        )

    if workload["query_sampling_method"] != (
        "random_corpus_title_plus_unique_suffix"
    ):
        raise ValueError(
            f"{workload_id}: Incorrect generation method"
        )

    # Verify base seed and derived seed.
    base_seed = int(workload["base_seed"])

    if base_seed != configured_base_seed:
        raise ValueError(
            f"{workload_id}: Base seed mismatch"
        )

    expected_seed = derive_expected_seed(
        base_seed,
        sample["corpus_id"],
        sample_size,
        query_count
    )

    if int(workload["derived_seed"]) != expected_seed:
        raise ValueError(
            f"{workload_id}: Derived seed mismatch"
        )

    # Verify workload file and checksum.
    workload_path = PROJECT_ROOT / workload["workload_file"]

    if not workload_path.is_file():
        raise ValueError(
            f"{workload_id}: Workload file missing"
        )

    actual_sha256 = calculate_sha256(workload_path)

    if actual_sha256 != workload["workload_sha256"]:
        raise ValueError(
            f"{workload_id}: SHA-256 mismatch"
        )

    expected_queries = iter(
        recreate_expected_queries(
            titles=titles,
            title_set=title_set,
            query_count=query_count,
            seed=expected_seed
        )
    )

    unique_queries = set()
    records_checked = 0

    with workload_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"{workload_id}: Invalid CSV columns"
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
                    f"{workload_id}: Empty query"
                )

            if row["expected_found"] != "false":
                raise ValueError(
                    f"{workload_id}: Expected result is not false"
                )

            # Actual corpus-membership check.
            if query in title_set:
                raise ValueError(
                    f"{workload_id}: Negative query exists "
                    f"in corpus: {query!r}"
                )

            # Reproducibility check.
            expected_query = next(expected_queries, None)

            if query != expected_query:
                raise ValueError(
                    f"{workload_id}: Query sequence mismatch "
                    f"at record {expected_id}"
                )

            # Negative queries must be unique.
            if query in unique_queries:
                raise ValueError(
                    f"{workload_id}: Duplicate negative query"
                )

            unique_queries.add(query)

    if records_checked != query_count:
        raise ValueError(
            f"{workload_id}: Record count mismatch. "
            f"Expected {query_count}, got {records_checked}"
        )

    unique_count = len(unique_queries)

    repeated_count = records_checked - unique_count

    if unique_count != query_count:
        raise ValueError(
            f"{workload_id}: Negative queries are not unique"
        )

    if repeated_count != 0:
        raise ValueError(
            f"{workload_id}: Repeated negative queries detected"
        )

    if unique_count != int(workload["unique_query_count"]):
        raise ValueError(
            f"{workload_id}: Unique query count mismatch"
        )

    if int(workload["repeated_query_occurrences"]) != 0:
        raise ValueError(
            f"{workload_id}: Manifest reports repetitions"
        )

    return records_checked


def main():
    """Validate all unsuccessful-query workload files."""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    search_column = config["dataset_schema"]["search_column"]

    base_seed = int(config["query_workloads"]["seed"])

    query_counts = [
        int(value)
        for value in config["query_workloads"]["query_counts"]
    ]

    sample_rows = read_csv_records(SAMPLE_MANIFEST_PATH)

    workload_rows = read_csv_records(WORKLOAD_MANIFEST_PATH)

    expected_per_type = len(sample_rows) * len(query_counts)

    unsuccessful_rows = [
        row for row in workload_rows
        if row["workload_type"] == "unsuccessful"
    ]

    successful_rows = [
        row for row in workload_rows
        if row["workload_type"] == "successful"
    ]

    print("=" * 65)
    print("UNSUCCESSFUL QUERY WORKLOAD VALIDATION")
    print("=" * 65)

    print("Corpus samples:", len(sample_rows))
    print("Total combined manifest entries:", len(workload_rows))
    print("Expected unsuccessful workloads:", expected_per_type)

    # Confirm that successful entries were preserved.
    if len(successful_rows) != expected_per_type:
        raise ValueError(
            "Successful workload entries are missing "
            "from the combined manifest."
        )

    if len(unsuccessful_rows) != expected_per_type:
        raise ValueError(
            "Unexpected number of unsuccessful workload entries."
        )

    # No duplicate workload IDs.
    workload_ids = [
        row["workload_id"] for row in workload_rows
    ]

    if len(workload_ids) != len(set(workload_ids)):
        raise ValueError("Duplicate workload IDs in manifest")

    print("Successful workload entries: PASS")
    print("Unsuccessful workload entries: PASS")
    print("Manifest workload IDs: PASS")

    validated_workloads = 0
    total_records = 0

    for sample in sample_rows:

        corpus_id = sample["corpus_id"]
        sample_size = int(sample["sample_size"])

        if sample["status"] != "PASS":
            raise ValueError(
                f"Source sample is not validated: "
                f"{sample['sample_file']}"
            )

        # Load the verified source sample.
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
                "Source corpus contains duplicates"
            )

        matching_workloads = [
            row for row in unsuccessful_rows
            if row["corpus_id"] == corpus_id
            and int(row["sample_size"]) == sample_size
        ]

        if len(matching_workloads) != len(query_counts):
            raise ValueError(
                f"{corpus_id} n={sample_size}: "
                "Incorrect number of negative workloads"
            )

        observed_counts = sorted(
            int(row["query_count"])
            for row in matching_workloads
        )

        if observed_counts != sorted(query_counts):
            raise ValueError(
                f"{corpus_id} n={sample_size}: "
                "Configured query counts do not match"
            )

        for workload in matching_workloads:

            records = validate_one_workload(
                workload=workload,
                sample=sample,
                titles=titles,
                title_set=title_set,
                search_column=search_column,
                configured_base_seed=base_seed
            )

            validated_workloads += 1
            total_records += records

            print(
                f"PASS: {workload['workload_id']} | "
                f"Records: {records:,} | "
                f"Unique: {records:,} | "
                f"Absent from corpus: PASS | "
                f"SHA-256: PASS | "
                f"Reproducibility: PASS"
            )

        del title_set
        del titles

    print("\n" + "=" * 65)
    print("UNSUCCESSFUL WORKLOAD VALIDATION SUMMARY")
    print("=" * 65)

    print("Corpus samples:", len(sample_rows))
    print("Validated workload files:", validated_workloads)
    print("Validated query records:", f"{total_records:,}")

    print("\nCorpus absence validation: PASS")
    print("Query uniqueness validation: PASS")
    print("Record-count validation: PASS")
    print("SHA-256 validation: PASS")
    print("Seed validation: PASS")
    print("Reproducibility validation: PASS")
    print("Successful manifest preservation: PASS")

    print("\nALL UNSUCCESSFUL QUERY WORKLOADS VALIDATED")


if __name__ == "__main__":
    main()
