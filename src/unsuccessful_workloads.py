
import csv
import random
import zlib

from pathlib import Path

from src.dataset_loader import load_corpus, calculate_sha256

from src.query_workloads import (
    load_experiment_config,
    load_sample_manifest,
    update_combined_manifest
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORKLOAD_FIELDS = [
    "query_id",
    "query",
    "expected_found"
]


def derive_unsuccessful_seed(
    base_seed: int,
    corpus_id: str,
    sample_size: int,
    query_count: int
) -> int:
    """
    Derive a reproducible seed for an unsuccessful workload.
    """

    identity = (
        f"{corpus_id}|{sample_size}|"
        f"{query_count}|unsuccessful"
    )

    checksum = zlib.crc32(
        identity.encode("utf-8")
    )

    return (base_seed + checksum) % (2 ** 32)


def generate_unsuccessful_queries(
    titles: list[str],
    title_set: set[str],
    query_count: int,
    seed: int
) -> list[str]:
    """
    Generate unique unsuccessful queries.

    Method:
        1. Randomly select an actual corpus title.
        2. Append a unique synthetic suffix.
        3. Verify that the generated query is absent
           from the original corpus.
        4. Reject any collision.

    Sampling the original titles uses replacement.
    Generated unsuccessful queries themselves are unique.
    """

    if not titles:
        raise ValueError("Cannot use an empty corpus.")

    if query_count <= 0:
        raise ValueError("Query count must be positive.")

    rng = random.Random(seed)

    base_titles = rng.choices(
        titles,
        k=query_count
    )

    unsuccessful_queries = []

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

        # Explicit absence check.
        assert candidate not in title_set

        generated_set.add(candidate)

        unsuccessful_queries.append(candidate)

    if len(generated_set) != query_count:
        raise RuntimeError(
            "Generated unsuccessful queries are not unique."
        )

    return unsuccessful_queries


def write_unsuccessful_workload(
    queries: list[str],
    corpus_id: str,
    sample_size: int,
    seed: int
) -> dict:
    """
    Save one unsuccessful-query workload.

    CSV files contain the same three columns used
    by successful-query workloads.
    """

    output_directory = (
        PROJECT_ROOT
        / "data"
        / "workloads"
        / "unsuccessful"
        / f"{corpus_id}_n{sample_size}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    query_count = len(queries)

    filename = (
        f"unsuccessful_q{query_count}_seed{seed}.csv"
    )

    output_path = output_directory / filename

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(
            file,
            lineterminator="\n"
        )

        writer.writerow(WORKLOAD_FIELDS)

        for index, query in enumerate(
            queries,
            start=1
        ):

            writer.writerow([
                index,
                query,
                "false"
            ])

    return {
        "workload_file": str(
            output_path.relative_to(PROJECT_ROOT)
        ),
        "workload_sha256": calculate_sha256(output_path),
        "query_count": query_count,
        "unique_query_count": len(set(queries)),
        "repeated_query_occurrences": (
            query_count - len(set(queries))
        )
    }


def main():
    """Generate unsuccessful queries for all 16 corpora."""

    config = load_experiment_config()

    workload_config = config["query_workloads"]

    base_seed = int(workload_config["seed"])

    query_counts = [
        int(value)
        for value in workload_config["query_counts"]
    ]

    search_column = config["dataset_schema"]["search_column"]

    sample_rows = load_sample_manifest()

    generated_rows = []

    total_queries = 0

    print("=" * 65)
    print("UNSUCCESSFUL QUERY WORKLOAD GENERATION")
    print("=" * 65)

    print("Base seed:", base_seed)
    print("Search column:", search_column)
    print("Query counts:", query_counts)
    print("Corpus samples:", len(sample_rows))

    for sample in sample_rows:

        corpus_id = sample["corpus_id"]

        language = sample["language"]

        sample_size = int(sample["sample_size"])

        if sample["status"] != "PASS":
            raise RuntimeError(
                f"Sample is not validated: {sample['sample_file']}"
            )

        print("\n" + "-" * 65)
        print(f"Language: {language}")
        print(f"Corpus ID: {corpus_id}")
        print(f"Sample size: {sample_size:,}")

        # Load the actual corpus.
        titles = load_corpus(
            sample_path=PROJECT_ROOT / sample["sample_file"],
            expected_count=sample_size,
            expected_sha256=sample["sample_sha256"],
            search_column=search_column
        )

        title_set = set(titles)

        if len(title_set) != sample_size:
            raise RuntimeError(
                f"Duplicate titles found in {corpus_id}."
            )

        for query_count in query_counts:

            derived_seed = derive_unsuccessful_seed(
                base_seed,
                corpus_id,
                sample_size,
                query_count
            )

            queries = generate_unsuccessful_queries(
                titles=titles,
                title_set=title_set,
                query_count=query_count,
                seed=derived_seed
            )

            result = write_unsuccessful_workload(
                queries=queries,
                corpus_id=corpus_id,
                sample_size=sample_size,
                seed=derived_seed
            )

            # Final generation-time checks.
            if result["unique_query_count"] != query_count:
                raise RuntimeError(
                    "Unsuccessful query uniqueness failure."
                )

            if result["repeated_query_occurrences"] != 0:
                raise RuntimeError(
                    "Unexpected repeated unsuccessful queries."
                )

            workload_id = (
                f"{corpus_id}_n{sample_size}"
                f"_unsuccessful_q{query_count}"
            )

            manifest_row = {
                "workload_id": workload_id,
                "corpus_id": corpus_id,
                "language": language,
                "sample_size": sample_size,
                "source_sample_file": sample["sample_file"],
                "source_sample_sha256": sample["sample_sha256"],
                "search_column": search_column,
                "workload_type": "unsuccessful",
                "query_count": result["query_count"],
                "unique_query_count": result["unique_query_count"],
                "repeated_query_occurrences": (
                    result["repeated_query_occurrences"]
                ),
                "base_seed": base_seed,
                "derived_seed": derived_seed,
                "query_sampling_method": (
                    "random_corpus_title_plus_unique_suffix"
                ),
                "expected_found": "false",
                "workload_file": result["workload_file"],
                "workload_sha256": result["workload_sha256"],
                "status": "PASS"
            }

            generated_rows.append(manifest_row)

            total_queries += query_count

            print(
                f"PASS: {query_count:,} queries | "
                f"Unique: {result['unique_query_count']:,} | "
                f"Repeated: 0 | Absence check: PASS"
            )

        # Release corpus memory before the next sample.
        del title_set
        del titles

    # Add new entries without overwriting successful workloads.
    update_combined_manifest(generated_rows)

    print("\n" + "=" * 65)
    print("UNSUCCESSFUL QUERY GENERATION SUMMARY")
    print("=" * 65)

    print("Corpus samples:", len(sample_rows))
    print("New workload files:", len(generated_rows))
    print("Total unsuccessful queries:", f"{total_queries:,}")

    print("Combined workload manifest updated.")

    print("\nUNSUCCESSFUL QUERY WORKLOAD GENERATION COMPLETED")


if __name__ == "__main__":
    main()
