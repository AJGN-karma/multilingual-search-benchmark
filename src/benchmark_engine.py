
import argparse
import csv
import gc
import json
import platform
import random
import statistics
import sys
import time

from pathlib import Path

import psutil

from src.linear_search import linear_search
from src.binary_search import (
    build_sorted_titles,
    binary_search,
)
from src.hash_search import (
    build_hash_index,
    hash_search,
)

from src.benchmark_workloads import (
    CorpusWorkloadProvider,
    read_manifest,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.json"

SAMPLE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "corpus_sample_manifest.csv"
)

WORKLOAD_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "query_workload_manifest.csv"
)

RAW_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "benchmark_runs"
)

TABLE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "benchmark_runs"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

ALGORITHMS = [
    "Linear Search",
    "Binary Search",
    "Hash-Table Search",
]

OUTCOME_TYPES = [
    "successful",
    "unsuccessful",
    "mixed",
]

REPETITION_PATTERNS = [
    "same_query_set",
    "mostly_unique",
    "mixed",
]


# ============================================================
# CSV OUTPUT
# ============================================================

def write_csv(path: Path, rows: list[dict]) -> None:
    """Write structured benchmark results to CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            f"No result rows available for {path.name}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# QUERY EXECUTION
# ============================================================

def execute_batch(search_function, records) -> int:
    """
    Execute one complete query batch.

    Returns the number of successful matches.

    Expected-result comparisons are excluded from the timed
    loop because correctness is checked separately.
    """

    found_count = 0

    for query, _ in records:

        if search_function(query):
            found_count += 1

    return found_count


def verify_search_function(
    algorithm_name: str,
    search_function,
    records: list[tuple[str, str]],
) -> int:
    """Verify every query result before timing."""

    expected_found = 0

    for query, expected_result in records:

        expected = expected_result == "true"

        actual = search_function(query)

        if actual != expected:
            raise RuntimeError(
                f"Correctness failure in {algorithm_name}. "
                f"Query: {query!r}. "
                f"Expected: {expected}. "
                f"Actual: {actual}."
            )

        if expected:
            expected_found += 1

    return expected_found


# ============================================================
# PREPROCESSING MEASUREMENTS
# ============================================================

def measure_preprocessing(
    algorithm_name: str,
    operation_name: str,
    builder,
    corpus_size: int,
    trials: int,
    warmups: int,
) -> list[dict]:
    """
    Measure preprocessing independently from query execution.

    The constructed index is released after every measurement.
    """

    rows = []

    # Warm-up preprocessing operations.
    for _ in range(warmups):

        temporary = builder()

        del temporary

        gc.collect()

    # Measured preprocessing operations.
    for trial in range(1, trials + 1):

        gc.collect()

        start = time.perf_counter_ns()

        temporary = builder()

        elapsed_ns = time.perf_counter_ns() - start

        rows.append({
            "algorithm": algorithm_name,
            "operation": operation_name,
            "trial": trial,
            "corpus_size": corpus_size,
            "elapsed_ns": elapsed_ns,
            "elapsed_ms": elapsed_ns / 1_000_000,
            "status": "PASS",
        })

        del temporary

    return rows


# ============================================================
# QUERY BENCHMARKING
# ============================================================

def measure_query_configuration(
    provider,
    corpus_titles,
    sorted_titles,
    hash_index,
    workload_type,
    repetition_pattern,
    query_count,
    trials,
    warmups,
):
    """Benchmark all three algorithms on identical queries."""

    records, workload_metadata = provider.prepare(
        workload_type=workload_type,
        query_count=query_count,
        repetition_pattern=repetition_pattern,
    )

    search_functions = {
        "Linear Search": (
            lambda query: linear_search(
                corpus_titles,
                query,
            )
        ),

        "Binary Search": (
            lambda query: binary_search(
                sorted_titles,
                query,
            )
        ),

        "Hash-Table Search": (
            lambda query: hash_search(
                hash_index,
                query,
            )
        ),
    }

    # --------------------------------------------------------
    # Correctness verification outside the timed region.
    # --------------------------------------------------------

    expected_found = None

    for algorithm_name, search_function in search_functions.items():

        current_found = verify_search_function(
            algorithm_name,
            search_function,
            records,
        )

        if expected_found is None:
            expected_found = current_found

        elif current_found != expected_found:
            raise RuntimeError(
                "Algorithms produced inconsistent results."
            )

    if expected_found != workload_metadata["successful_count"]:
        raise RuntimeError(
            "Search result count does not match workload metadata."
        )

    # --------------------------------------------------------
    # Warm-up runs.
    # --------------------------------------------------------

    for algorithm_name in ALGORITHMS:

        search_function = search_functions[algorithm_name]

        for _ in range(warmups):

            warmup_hits = execute_batch(
                search_function,
                records,
            )

            if warmup_hits != expected_found:
                raise RuntimeError(
                    f"Warm-up correctness failure: {algorithm_name}"
                )

    # --------------------------------------------------------
    # Measured trials.
    # --------------------------------------------------------

    raw_rows = []

    trial_rng = random.Random(
        workload_metadata["derived_seed"]
    )

    for trial in range(1, trials + 1):

        algorithm_order = ALGORITHMS.copy()

        # Deterministically randomize algorithm order.
        trial_rng.shuffle(algorithm_order)

        for algorithm_name in algorithm_order:

            search_function = search_functions[algorithm_name]

            start = time.perf_counter_ns()

            found_count = execute_batch(
                search_function,
                records,
            )

            elapsed_ns = time.perf_counter_ns() - start

            if found_count != expected_found:
                raise RuntimeError(
                    f"Timed trial correctness failure: "
                    f"{algorithm_name}"
                )

            elapsed_seconds = elapsed_ns / 1_000_000_000

            raw_rows.append({
                "corpus_id": workload_metadata["corpus_id"],
                "language": workload_metadata["language"],
                "sample_size": workload_metadata["sample_size"],
                "workload_type": workload_type,
                "repetition_pattern": repetition_pattern,
                "query_count": query_count,
                "trial": trial,
                "algorithm": algorithm_name,
                "elapsed_ns": elapsed_ns,
                "elapsed_ms": elapsed_ns / 1_000_000,
                "ns_per_query": elapsed_ns / query_count,
                "queries_per_second": (
                    query_count / elapsed_seconds
                ),
                "expected_found_count": expected_found,
                "actual_found_count": found_count,
                "derived_seed": workload_metadata["derived_seed"],
                "status": "PASS",
            })

    # --------------------------------------------------------
    # Summary statistics.
    # --------------------------------------------------------

    summary_rows = []

    for algorithm_name in ALGORITHMS:

        algorithm_trials = [
            row
            for row in raw_rows
            if row["algorithm"] == algorithm_name
        ]

        elapsed_values = [
            int(row["elapsed_ns"])
            for row in algorithm_trials
        ]

        median_ns = statistics.median(elapsed_values)

        summary_rows.append({
            "corpus_id": workload_metadata["corpus_id"],
            "language": workload_metadata["language"],
            "sample_size": workload_metadata["sample_size"],
            "workload_type": workload_type,
            "repetition_pattern": repetition_pattern,
            "query_count": query_count,
            "algorithm": algorithm_name,
            "trials": trials,
            "warmups": warmups,
            "median_batch_ns": int(median_ns),
            "mean_batch_ns": round(
                statistics.mean(elapsed_values),
                2,
            ),
            "stdev_batch_ns": round(
                statistics.stdev(elapsed_values),
                2,
            ),
            "median_ns_per_query": round(
                median_ns / query_count,
                2,
            ),
            "queries_per_second": round(
                query_count / (median_ns / 1_000_000_000),
                2,
            ),
            "expected_found_count": expected_found,
            "successful_query_count": (
                workload_metadata["successful_count"]
            ),
            "unsuccessful_query_count": (
                workload_metadata["unsuccessful_count"]
            ),
            "unique_query_count": (
                workload_metadata["unique_query_count"]
            ),
            "repeated_query_occurrences": (
                workload_metadata["repeated_query_occurrences"]
            ),
            "derived_seed": workload_metadata["derived_seed"],
            "status": "PASS",
        })

    return raw_rows, summary_rows


# ============================================================
# PREPROCESSING SUMMARY
# ============================================================

def build_preprocessing_summary(
    preprocessing_rows,
    corpus_size,
    trials,
):
    """Create a separate preprocessing summary."""

    summary_rows = []

    for algorithm_name in ALGORITHMS:

        rows = [
            row
            for row in preprocessing_rows
            if row["algorithm"] == algorithm_name
            and row["status"] == "PASS"
        ]

        if not rows:

            summary_rows.append({
                "algorithm": algorithm_name,
                "operation": "none",
                "corpus_size": corpus_size,
                "trials": 0,
                "median_ns": 0,
                "mean_ns": 0,
                "stdev_ns": 0,
                "status": "NOT_REQUIRED",
            })

            continue

        values = [
            int(row["elapsed_ns"])
            for row in rows
        ]

        summary_rows.append({
            "algorithm": algorithm_name,
            "operation": rows[0]["operation"],
            "corpus_size": corpus_size,
            "trials": trials,
            "median_ns": int(
                statistics.median(values)
            ),
            "mean_ns": round(
                statistics.mean(values),
                2,
            ),
            "stdev_ns": round(
                statistics.stdev(values),
                2,
            ),
            "status": "PASS",
        })

    return summary_rows


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Controlled benchmark for exact-search algorithms."
        )
    )

    parser.add_argument(
        "--language",
        choices=["Tulu", "Kannada", "English"],
        default="Tulu",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--query-count",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--run-tag",
        default="pilot",
        help="Unique identifier for this benchmark run.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate run identifier and construct output paths.
    # --------------------------------------------------------

    run_tag = args.run_tag

    if not run_tag or not all(
        character.isalnum() or character in "_-"
        for character in run_tag
    ):
        parser.error(
            "Run tag may contain only letters, numbers, "
            "underscores and hyphens."
        )

    output_directory = RAW_OUTPUT_ROOT / run_tag

    summary_path = (
        TABLE_OUTPUT_ROOT
        / f"{run_tag}_summary.csv"
    )

    preprocessing_path = (
        TABLE_OUTPUT_ROOT
        / f"{run_tag}_preprocessing.csv"
    )

    metadata_path = (
        TABLE_OUTPUT_ROOT
        / f"{run_tag}_metadata.json"
    )

    preprocessing_trials_path = (
        output_directory
        / "preprocessing_trials.csv"
    )

    trial_path = (
        output_directory
        / "benchmark_trials.csv"
    )

    # Prevent accidental overwriting of previous experiments.
    existing_paths = [
        output_directory,
        summary_path,
        preprocessing_path,
        metadata_path,
    ]

    if any(path.exists() for path in existing_paths):
        parser.error(
            f"Run tag '{run_tag}' already exists. "
            "Choose a new run tag."
        )

    # --------------------------------------------------------
    # Load configuration.
    # --------------------------------------------------------

    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    trials = int(
        config["benchmark"]["trials"]
    )

    warmups = int(
        config["benchmark"]["warmup_runs"]
    )

    base_seed = int(
        config["query_workloads"]["seed"]
    )

    search_column = (
        config["dataset_schema"]["search_column"]
    )

    sample_rows = read_manifest(
        SAMPLE_MANIFEST_PATH
    )

    workload_rows = read_manifest(
        WORKLOAD_MANIFEST_PATH
    )

    matching_samples = [
        row
        for row in sample_rows
        if row["language"] == args.language
        and int(row["sample_size"]) == args.sample_size
    ]

    if len(matching_samples) != 1:
        parser.error(
            "Requested corpus sample was not found exactly once."
        )

    configured_query_counts = [
        int(value)
        for value in config["query_workloads"]["query_counts"]
    ]

    if args.query_count not in configured_query_counts:
        parser.error(
            "Requested query count is not configured."
        )

    if trials < 2:
        parser.error(
            "At least two measured trials are required."
        )

    sample = matching_samples[0]

    if sample["status"] != "PASS":
        parser.error(
            "Requested sample has not passed validation."
        )

    # --------------------------------------------------------
    # Experiment information.
    # --------------------------------------------------------

    print("=" * 70)
    print("EXACT-SEARCH PERFORMANCE BENCHMARK")
    print("=" * 70)

    print("Run tag:", run_tag)
    print("Language:", args.language)
    print("Corpus size:", f"{args.sample_size:,}")
    print("Query count:", f"{args.query_count:,}")
    print("Warm-up runs:", warmups)
    print("Measured trials:", trials)
    print("Timer: time.perf_counter_ns")
    print("Search column:", search_column)

    # --------------------------------------------------------
    # Load source corpus and workload provider.
    # --------------------------------------------------------

    print("\nLoading corpus and query provider...")

    provider = CorpusWorkloadProvider(
        sample_row=sample,
        workload_rows=workload_rows,
        search_column=search_column,
        base_seed=base_seed,
    )

    titles = provider.titles

    print("Corpus loaded:", len(titles))
    print("Unique corpus titles:", len(provider.title_set))

    # --------------------------------------------------------
    # Preprocessing measurements.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING MEASUREMENTS")
    print("=" * 70)

    preprocessing_rows = []

    # Linear Search does not require index construction.
    preprocessing_rows.append({
        "algorithm": "Linear Search",
        "operation": "none",
        "trial": 0,
        "corpus_size": len(titles),
        "elapsed_ns": 0,
        "elapsed_ms": 0.0,
        "status": "NOT_REQUIRED",
    })

    # Binary Search sorting.
    binary_preprocessing = measure_preprocessing(
        algorithm_name="Binary Search",
        operation_name="sort_titles",
        builder=lambda: build_sorted_titles(titles),
        corpus_size=len(titles),
        trials=trials,
        warmups=warmups,
    )

    preprocessing_rows.extend(
        binary_preprocessing
    )

    # Hash-Table Search index construction.
    hash_preprocessing = measure_preprocessing(
        algorithm_name="Hash-Table Search",
        operation_name="build_hash_index",
        builder=lambda: build_hash_index(titles),
        corpus_size=len(titles),
        trials=trials,
        warmups=warmups,
    )

    preprocessing_rows.extend(
        hash_preprocessing
    )

    # Build indexes used by the actual query measurements.
    sorted_titles = build_sorted_titles(titles)

    hash_index = build_hash_index(titles)

    if sorted_titles != sorted(titles):
        raise RuntimeError(
            "Binary Search sorting verification failed."
        )

    if len(hash_index) != len(titles):
        raise RuntimeError(
            "Hash index construction verification failed."
        )

    print(
        "Binary sorting median (ms):",
        round(
            statistics.median([
                row["elapsed_ms"]
                for row in binary_preprocessing
            ]),
            4,
        ),
    )

    print(
        "Hash construction median (ms):",
        round(
            statistics.median([
                row["elapsed_ms"]
                for row in hash_preprocessing
            ]),
            4,
        ),
    )

    print("Linear Search preprocessing: NOT REQUIRED")

    # --------------------------------------------------------
    # Query execution measurements.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUERY EXECUTION MEASUREMENTS")
    print("=" * 70)

    all_raw_rows = []
    all_summary_rows = []

    for workload_type in OUTCOME_TYPES:

        for repetition_pattern in REPETITION_PATTERNS:

            print(
                f"\nWorkload: {workload_type} | "
                f"Pattern: {repetition_pattern}"
            )

            raw_rows, summary_rows = (
                measure_query_configuration(
                    provider=provider,
                    corpus_titles=titles,
                    sorted_titles=sorted_titles,
                    hash_index=hash_index,
                    workload_type=workload_type,
                    repetition_pattern=repetition_pattern,
                    query_count=args.query_count,
                    trials=trials,
                    warmups=warmups,
                )
            )

            all_raw_rows.extend(raw_rows)
            all_summary_rows.extend(summary_rows)

            for row in summary_rows:

                print(
                    f"{row['algorithm']:20} | "
                    f"Median: "
                    f"{row['median_batch_ns'] / 1_000_000:10.4f} ms | "
                    f"Per query: "
                    f"{row['median_ns_per_query']:10.2f} ns | "
                    f"QPS: "
                    f"{row['queries_per_second']:,.2f}"
                )

    # --------------------------------------------------------
    # Save raw and summary measurements.
    # --------------------------------------------------------

    write_csv(
        trial_path,
        all_raw_rows,
    )

    write_csv(
        preprocessing_trials_path,
        preprocessing_rows,
    )

    write_csv(
        summary_path,
        all_summary_rows,
    )

    preprocessing_summary = build_preprocessing_summary(
        preprocessing_rows,
        corpus_size=len(titles),
        trials=trials,
    )

    write_csv(
        preprocessing_path,
        preprocessing_summary,
    )

    # --------------------------------------------------------
    # Save experiment metadata.
    # --------------------------------------------------------

    process = psutil.Process()

    metadata = {
        "experiment": "Exact-search performance benchmark",
        "run_tag": run_tag,
        "language": args.language,
        "corpus_size": args.sample_size,
        "query_count": args.query_count,
        "workload_types": OUTCOME_TYPES,
        "repetition_patterns": REPETITION_PATTERNS,
        "algorithms": ALGORITHMS,
        "trials": trials,
        "warmups": warmups,
        "timer": "time.perf_counter_ns",
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "current_process_rss_bytes": (
            process.memory_info().rss
        ),
        "total_trial_rows": len(all_raw_rows),
        "total_summary_rows": len(all_summary_rows),
        "correctness_verified_before_timing": True,
        "raw_output_directory": str(output_directory),
        "summary_file": str(summary_path),
        "preprocessing_summary_file": str(preprocessing_path),
        "status": "PASS",
    }

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    # --------------------------------------------------------
    # Final completion report.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BENCHMARK RUN COMPLETED")
    print("=" * 70)

    print(
        "Workload configurations:",
        len(OUTCOME_TYPES) * len(REPETITION_PATTERNS),
    )

    print("Algorithms:", len(ALGORITHMS))

    print(
        "Measured query batches:",
        len(all_raw_rows),
    )

    print(
        "Preprocessing trial records:",
        len(preprocessing_rows),
    )

    print("\nGenerated reports:")

    print("Raw trials:", trial_path)

    print(
        "Preprocessing trials:",
        preprocessing_trials_path,
    )

    print("Summary:", summary_path)

    print(
        "Preprocessing summary:",
        preprocessing_path,
    )

    print("Metadata:", metadata_path)

    print("\nALL BENCHMARK RUN CHECKS PASSED")


if __name__ == "__main__":
    main()
