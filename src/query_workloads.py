
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

OUTPUT_MANIFEST_CSV = (
    PROJECT_ROOT / "results" / "tables"
    / "query_workload_manifest.csv"
)

OUTPUT_MANIFEST_JSON = (
    PROJECT_ROOT / "results" / "tables"
    / "query_workload_manifest.json"
)


WORKLOAD_FIELDS = [
    "workload_id",
    "corpus_id",
    "language",
    "sample_size",
    "source_sample_file",
    "source_sample_sha256",
    "search_column",
    "workload_type",
    "query_count",
    "unique_query_count",
    "repeated_query_occurrences",
    "base_seed",
    "derived_seed",
    "query_sampling_method",
    "expected_found",
    "workload_file",
    "workload_sha256",
    "status"
]


def load_experiment_config():
    """Read the experiment configuration."""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def load_sample_manifest():
    """Read the validated corpus sample manifest."""

    with SAMPLE_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def derive_workload_seed(
    base_seed: int,
    corpus_id: str,
    sample_size: int,
    query_count: int
) -> int:
    """
    Derive a deterministic seed for each workload.

    CRC32 is used instead of Python's hash(), because Python's
    hash randomization can vary between interpreter processes.
    """

    identity = (
        f"{corpus_id}|{sample_size}|{query_count}|successful"
    )

    identity_checksum = zlib.crc32(
        identity.encode("utf-8")
    )

    return (base_seed + identity_checksum) % (2 ** 32)


def generate_successful_queries(
    titles: list[str],
    query_count: int,
    seed: int
) -> list[str]:
    """
    Generate successful queries using uniform random sampling
    with replacement.

    Every generated query comes directly from the corpus.

    Repeated queries are permitted so that workloads can contain
    more queries than the number of distinct corpus titles.
    """

    if not titles:
        raise ValueError("Cannot generate queries from an empty corpus.")

    if query_count <= 0:
        raise ValueError("Query count must be greater than zero.")

    rng = random.Random(seed)

    return rng.choices(titles, k=query_count)


def write_successful_workload(
    queries: list[str],
    corpus_id: str,
    sample_size: int,
    seed: int
) -> dict:
    """Write one successful-query workload to a CSV file."""

    output_directory = (
        PROJECT_ROOT
        / "data"
        / "workloads"
        / "successful"
        / f"{corpus_id}_n{sample_size}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    query_count = len(queries)

    unique_query_count = len(set(queries))

    repeated_occurrences = query_count - unique_query_count

    workload_filename = (
        f"successful_q{query_count}_seed{seed}.csv"
    )

    workload_path = output_directory / workload_filename

    with workload_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file, lineterminator="\n")

        writer.writerow([
            "query_id",
            "query",
            "expected_found"
        ])

        for index, query in enumerate(queries, start=1):

            writer.writerow([
                index,
                query,
                "true"
            ])

    workload_sha256 = calculate_sha256(workload_path)

    return {
        "workload_file": str(
            workload_path.relative_to(PROJECT_ROOT)
        ),
        "workload_sha256": workload_sha256,
        "query_count": query_count,
        "unique_query_count": unique_query_count,
        "repeated_query_occurrences": repeated_occurrences
    }


def update_combined_manifest(new_rows: list[dict]):
    """
    Create or update the combined query workload manifest.

    Existing workload entries are retained when additional
    workload types are generated in later stages.
    """

    existing_rows = {}

    if OUTPUT_MANIFEST_CSV.exists():

        with OUTPUT_MANIFEST_CSV.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            for row in csv.DictReader(file):
                existing_rows[row["workload_id"]] = row

    for row in new_rows:
        existing_rows[row["workload_id"]] = row

    combined_rows = sorted(
        existing_rows.values(),
        key=lambda row: (
            row["corpus_id"],
            int(row["sample_size"]),
            row["workload_type"],
            int(row["query_count"])
        )
    )

    OUTPUT_MANIFEST_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_MANIFEST_CSV.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=WORKLOAD_FIELDS,
            lineterminator="\n"
        )

        writer.writeheader()
        writer.writerows(combined_rows)

    json_output = {
        "project": "Multilingual Exact-Search Benchmark",
        "description": (
            "Manifest of reproducible query workloads "
            "generated from validated corpus samples."
        ),
        "total_workload_files": len(combined_rows),
        "workloads": combined_rows
    }

    with OUTPUT_MANIFEST_JSON.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_output,
            file,
            indent=2,
            ensure_ascii=False
        )

        file.write("\n")


def main():
    """Generate successful query workloads for all corpora."""

    config = load_experiment_config()

    workload_config = config["query_workloads"]

    base_seed = int(workload_config["seed"])

    query_counts = [
        int(value)
        for value in workload_config["query_counts"]
    ]

    search_column = config["dataset_schema"]["search_column"]

    sample_rows = load_sample_manifest()

    if not sample_rows:
        raise RuntimeError("Sample manifest contains no entries.")

    generated_rows = []

    total_queries = 0

    print("=" * 65)
    print("SUCCESSFUL QUERY WORKLOAD GENERATION")
    print("=" * 65)

    print("Base random seed:", base_seed)
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

        if sample["search_column"] != search_column:
            raise RuntimeError(
                f"Search column mismatch: {sample['sample_file']}"
            )

        if int(sample["sample_unique_title_count"]) != sample_size:
            raise RuntimeError(
                f"Sample uniqueness mismatch: {sample['sample_file']}"
            )

        print("\n" + "-" * 65)
        print(f"Language: {language}")
        print(f"Corpus ID: {corpus_id}")
        print(f"Sample size: {sample_size:,}")

        # Load and verify the complete sample once.
        titles = load_corpus(
            sample_path=PROJECT_ROOT / sample["sample_file"],
            expected_count=sample_size,
            expected_sha256=sample["sample_sha256"],
            search_column=search_column
        )

        for query_count in query_counts:

            derived_seed = derive_workload_seed(
                base_seed,
                corpus_id,
                sample_size,
                query_count
            )

            queries = generate_successful_queries(
                titles=titles,
                query_count=query_count,
                seed=derived_seed
            )

            result = write_successful_workload(
                queries=queries,
                corpus_id=corpus_id,
                sample_size=sample_size,
                seed=derived_seed
            )

            workload_id = (
                f"{corpus_id}_n{sample_size}"
                f"_successful_q{query_count}"
            )

            manifest_row = {
                "workload_id": workload_id,
                "corpus_id": corpus_id,
                "language": language,
                "sample_size": sample_size,
                "source_sample_file": sample["sample_file"],
                "source_sample_sha256": sample["sample_sha256"],
                "search_column": search_column,
                "workload_type": "successful",
                "query_count": result["query_count"],
                "unique_query_count": result["unique_query_count"],
                "repeated_query_occurrences": (
                    result["repeated_query_occurrences"]
                ),
                "base_seed": base_seed,
                "derived_seed": derived_seed,
                "query_sampling_method": (
                    "uniform_random_sampling_with_replacement"
                ),
                "expected_found": "true",
                "workload_file": result["workload_file"],
                "workload_sha256": result["workload_sha256"],
                "status": "PASS"
            }

            generated_rows.append(manifest_row)

            total_queries += query_count

            print(
                f"PASS: {query_count:,} queries | "
                f"Unique: {result['unique_query_count']:,} | "
                f"Repeated occurrences: "
                f"{result['repeated_query_occurrences']:,}"
            )

        # Release the current corpus before loading the next.
        del titles

    update_combined_manifest(generated_rows)

    print("\n" + "=" * 65)
    print("SUCCESSFUL QUERY GENERATION SUMMARY")
    print("=" * 65)

    print("Corpus samples processed:", len(sample_rows))
    print("Workload files generated:", len(generated_rows))
    print("Total successful queries:", f"{total_queries:,}")
    print("Base seed:", base_seed)

    print(
        "\nCSV manifest:",
        OUTPUT_MANIFEST_CSV.relative_to(PROJECT_ROOT)
    )

    print(
        "JSON manifest:",
        OUTPUT_MANIFEST_JSON.relative_to(PROJECT_ROOT)
    )

    print("\nSUCCESSFUL QUERY WORKLOAD GENERATION COMPLETED")


if __name__ == "__main__":
    main()
