
import csv
import json
import random
import zlib
from collections import Counter
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
    """Read a CSV file using UTF-8 encoding."""

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
    """Reconstruct the mixed-workload random seed."""

    identity = (
        f"{corpus_id}|{sample_size}|{query_count}|mixed"
    )

    checksum = zlib.crc32(identity.encode("utf-8"))

    return (base_seed + checksum) % (2 ** 32)


def read_verified_prefix(
    workload,
    expected_result,
    prefix_count
):
    """
    Read a prefix of a previously validated source workload.

    Verify source checksum, column structure, query IDs,
    and expected-result labels.
    """

    path = PROJECT_ROOT / workload["workload_file"]

    if not path.is_file():
        raise ValueError(
            f"Source workload missing: {path}"
        )

    actual_hash = calculate_sha256(path)

    if actual_hash != workload["workload_sha256"]:
        raise ValueError(
            f"Source workload SHA-256 mismatch: {path.name}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Invalid source CSV columns: {path.name}"
            )

        for expected_id, row in enumerate(
            reader,
            start=1
        ):

            if expected_id > prefix_count:
                break

            if int(row["query_id"]) != expected_id:
                raise ValueError(
                    f"Incorrect source query ID: {path.name}"
                )

            if row["expected_found"] != expected_result:
                raise ValueError(
                    f"Incorrect source query label: {path.name}"
                )

            records.append((
                row["query"],
                expected_result
            ))

    if len(records) != prefix_count:
        raise ValueError(
            f"Insufficient source records: {path.name}"
        )

    return records


def find_exact_source(
    workload_rows,
    corpus_id,
    sample_size,
    query_count,
    workload_type
):
    """Find exactly one matching successful or unsuccessful source."""

    matches = [
        row for row in workload_rows
        if row["corpus_id"] == corpus_id
        and int(row["sample_size"]) == sample_size
        and int(row["query_count"]) == query_count
        and row["workload_type"] == workload_type
        and row["status"] == "PASS"
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {workload_type} source for "
            f"{corpus_id}, n={sample_size}, q={query_count}; "
            f"found {len(matches)}."
        )

    return matches[0]


def validate_one_mixed_workload(
    workload,
    sample,
    titles,
    title_set,
    workload_rows,
    search_column,
    configured_base_seed,
):
    """Independently validate one mixed-query workload."""

    corpus_id = sample["corpus_id"]
    sample_size = int(sample["sample_size"])
    query_count = int(workload["query_count"])

    workload_id = workload["workload_id"]

    if query_count % 2 != 0:
        raise ValueError(
            f"{workload_id}: Query count must be even."
        )

    half_count = query_count // 2

    # Verify corpus identity.
    if workload["corpus_id"] != corpus_id:
        raise ValueError(
            f"{workload_id}: Corpus ID mismatch."
        )

    if int(workload["sample_size"]) != sample_size:
        raise ValueError(
            f"{workload_id}: Sample size mismatch."
        )

    if workload["source_sample_file"] != sample["sample_file"]:
        raise ValueError(
            f"{workload_id}: Source sample path mismatch."
        )

    if workload["source_sample_sha256"] != sample["sample_sha256"]:
        raise ValueError(
            f"{workload_id}: Source sample SHA-256 mismatch."
        )

    if workload["search_column"] != search_column:
        raise ValueError(
            f"{workload_id}: Search column mismatch."
        )

    if workload["workload_type"] != "mixed":
        raise ValueError(
            f"{workload_id}: Incorrect workload type."
        )

    if workload["expected_found"] != "50%_true_50%_false":
        raise ValueError(
            f"{workload_id}: Incorrect expected-result metadata."
        )

    expected_method = (
        "balanced_merge_of_validated_successful_and_"
        "unsuccessful_workloads_then_deterministic_shuffle"
    )

    if workload["query_sampling_method"] != expected_method:
        raise ValueError(
            f"{workload_id}: Incorrect generation method."
        )

    # Verify random seeds.
    base_seed = int(workload["base_seed"])

    if base_seed != configured_base_seed:
        raise ValueError(
            f"{workload_id}: Base seed mismatch."
        )

    expected_seed = derive_expected_seed(
        base_seed,
        corpus_id,
        sample_size,
        query_count
    )

    if int(workload["derived_seed"]) != expected_seed:
        raise ValueError(
            f"{workload_id}: Derived seed mismatch."
        )

    # Locate the validated successful and unsuccessful sources.
    positive_source = find_exact_source(
        workload_rows,
        corpus_id,
        sample_size,
        query_count,
        "successful"
    )

    negative_source = find_exact_source(
        workload_rows,
        corpus_id,
        sample_size,
        query_count,
        "unsuccessful"
    )

    # Reconstruct the exact source prefixes used by the generator.
    positive_records = read_verified_prefix(
        positive_source,
        "true",
        half_count
    )

    negative_records = read_verified_prefix(
        negative_source,
        "false",
        half_count
    )

    if positive_source["source_sample_sha256"] != sample["sample_sha256"]:
        raise ValueError(
            f"{workload_id}: Positive source corpus mismatch."
        )

    if negative_source["source_sample_sha256"] != sample["sample_sha256"]:
        raise ValueError(
            f"{workload_id}: Negative source corpus mismatch."
        )

    # Independently reconstruct the mixed order.
    expected_records = positive_records + negative_records

    rng = random.Random(expected_seed)
    rng.shuffle(expected_records)

    # Verify the mixed workload file and checksum.
    workload_path = PROJECT_ROOT / workload["workload_file"]

    if not workload_path.is_file():
        raise ValueError(
            f"{workload_id}: Mixed workload file missing."
        )

    actual_hash = calculate_sha256(workload_path)

    if actual_hash != workload["workload_sha256"]:
        raise ValueError(
            f"{workload_id}: Mixed workload SHA-256 mismatch."
        )

    result_counts = Counter()
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
                f"{workload_id}: Incorrect CSV columns."
            )

        for expected_id, row in enumerate(
            reader,
            start=1
        ):

            records_checked += 1

            if int(row["query_id"]) != expected_id:
                raise ValueError(
                    f"{workload_id}: Invalid query ID."
                )

            query = row["query"]
            result = row["expected_found"]

            if not query:
                raise ValueError(
                    f"{workload_id}: Empty query."
                )

            if result not in ("true", "false"):
                raise ValueError(
                    f"{workload_id}: Invalid expected result."
                )

            expected_query, expected_result = expected_records[
                expected_id - 1
            ]

            if query != expected_query:
                raise ValueError(
                    f"{workload_id}: Query sequence mismatch "
                    f"at record {expected_id}."
                )

            if result != expected_result:
                raise ValueError(
                    f"{workload_id}: Result-label sequence mismatch "
                    f"at record {expected_id}."
                )

            # Independent actual membership check.
            exists = query in title_set

            if result == "true" and not exists:
                raise ValueError(
                    f"{workload_id}: Positive query is absent: "
                    f"{query!r}"
                )

            if result == "false" and exists:
                raise ValueError(
                    f"{workload_id}: Negative query exists: "
                    f"{query!r}"
                )

            result_counts[result] += 1
            unique_queries.add(query)

    if records_checked != query_count:
        raise ValueError(
            f"{workload_id}: Record count mismatch."
        )

    if result_counts["true"] != half_count:
        raise ValueError(
            f"{workload_id}: Incorrect successful-query ratio."
        )

    if result_counts["false"] != half_count:
        raise ValueError(
            f"{workload_id}: Incorrect unsuccessful-query ratio."
        )

    unique_count = len(unique_queries)
    repeated_count = records_checked - unique_count

    if unique_count != int(workload["unique_query_count"]):
        raise ValueError(
            f"{workload_id}: Unique query count mismatch."
        )

    if repeated_count != int(
        workload["repeated_query_occurrences"]
    ):
        raise ValueError(
            f"{workload_id}: Repeated query count mismatch."
        )

    return {
        "records": records_checked,
        "successful": result_counts["true"],
        "unsuccessful": result_counts["false"],
        "unique": unique_count,
        "repeated": repeated_count,
    }


def main():
    """Validate all mixed workloads against source corpora."""

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

    mixed_rows = [
        row for row in workload_rows
        if row["workload_type"] == "mixed"
    ]

    successful_rows = [
        row for row in workload_rows
        if row["workload_type"] == "successful"
    ]

    unsuccessful_rows = [
        row for row in workload_rows
        if row["workload_type"] == "unsuccessful"
    ]

    print("=" * 65)
    print("MIXED QUERY WORKLOAD INDEPENDENT VALIDATION")
    print("=" * 65)

    print("Corpus samples:", len(sample_rows))
    print("Combined manifest entries:", len(workload_rows))
    print("Successful entries:", len(successful_rows))
    print("Unsuccessful entries:", len(unsuccessful_rows))
    print("Mixed entries:", len(mixed_rows))

    if len(successful_rows) != expected_per_type:
        raise ValueError("Incorrect number of successful entries.")

    if len(unsuccessful_rows) != expected_per_type:
        raise ValueError("Incorrect number of unsuccessful entries.")

    if len(mixed_rows) != expected_per_type:
        raise ValueError("Incorrect number of mixed entries.")

    workload_ids = [
        row["workload_id"] for row in workload_rows
    ]

    if len(workload_ids) != len(set(workload_ids)):
        raise ValueError("Duplicate workload IDs found.")

    print("Workload counts and unique IDs: PASS")

    validated_workloads = 0
    total_records = 0
    total_successful = 0
    total_unsuccessful = 0

    for sample in sample_rows:

        corpus_id = sample["corpus_id"]
        sample_size = int(sample["sample_size"])

        # Load and verify the corresponding source corpus.
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
                "Source corpus has duplicate titles."
            )

        matching_mixed = [
            row for row in mixed_rows
            if row["corpus_id"] == corpus_id
            and int(row["sample_size"]) == sample_size
        ]

        if len(matching_mixed) != len(query_counts):
            raise ValueError(
                f"{corpus_id} n={sample_size}: "
                "Incorrect number of mixed workloads."
            )

        observed_counts = sorted(
            int(row["query_count"])
            for row in matching_mixed
        )

        if observed_counts != sorted(query_counts):
            raise ValueError(
                f"{corpus_id} n={sample_size}: "
                "Query counts mismatch."
            )

        for workload in matching_mixed:

            result = validate_one_mixed_workload(
                workload=workload,
                sample=sample,
                titles=titles,
                title_set=title_set,
                workload_rows=workload_rows,
                search_column=search_column,
                configured_base_seed=base_seed,
            )

            validated_workloads += 1
            total_records += result["records"]
            total_successful += result["successful"]
            total_unsuccessful += result["unsuccessful"]

            print(
                f"PASS: {workload['workload_id']} | "
                f"Records: {result['records']:,} | "
                f"True: {result['successful']:,} | "
                f"False: {result['unsuccessful']:,} | "
                f"Unique: {result['unique']:,} | "
                f"Repeated: {result['repeated']:,} | "
                f"Membership: PASS | "
                f"SHA-256: PASS | "
                f"Reproducibility: PASS"
            )

        del title_set
        del titles

    print("\n" + "=" * 65)
    print("MIXED WORKLOAD VALIDATION SUMMARY")
    print("=" * 65)

    print("Validated mixed files:", validated_workloads)
    print("Validated query records:", f"{total_records:,}")
    print("Successful queries:", f"{total_successful:,}")
    print("Unsuccessful queries:", f"{total_unsuccessful:,}")

    print("\nPositive query membership: PASS")
    print("Negative query absence: PASS")
    print("50:50 distribution: PASS")
    print("Record-count validation: PASS")
    print("SHA-256 validation: PASS")
    print("Seed validation: PASS")
    print("Reproducibility validation: PASS")
    print("Unique and repeated query counts: PASS")

    print("\nALL MIXED QUERY WORKLOADS VALIDATED")


if __name__ == "__main__":
    main()
