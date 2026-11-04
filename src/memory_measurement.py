
import argparse
import csv
import gc
import json
import statistics
import subprocess
import sys
import tracemalloc

from pathlib import Path

import psutil

from src.dataset_loader import load_corpus
from src.binary_search import build_sorted_titles
from src.hash_search import build_hash_index


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "corpus_sample_manifest.csv"
)

RAW_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "memory_pilot"
    / "memory_trials.csv"
)

SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "memory_pilot_summary.csv"
)

METADATA_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "memory_pilot_metadata.json"
)


ALGORITHMS = [
    "Linear Search",
    "Binary Search",
    "Hash-Table Search",
]


def read_manifest():
    """Read the validated corpus sample manifest."""

    with SAMPLE_MANIFEST.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def build_index(algorithm, titles):
    """Build the selected data structure."""

    if algorithm == "Linear Search":
        return None

    if algorithm == "Binary Search":
        return build_sorted_titles(titles)

    if algorithm == "Hash-Table Search":
        return build_hash_index(titles)

    raise ValueError(
        f"Unsupported algorithm: {algorithm}"
    )


def verify_index(algorithm, index, expected_count):
    """Verify the constructed index without copying it."""

    if algorithm == "Linear Search":
        if index is not None:
            raise RuntimeError(
                "Linear Search should not construct an index."
            )

    elif algorithm == "Binary Search":

        if not isinstance(index, list):
            raise RuntimeError(
                "Binary Search did not construct a list."
            )

        if len(index) != expected_count:
            raise RuntimeError(
                "Sorted array has an incorrect record count."
            )

        if any(
            index[position - 1] > index[position]
            for position in range(1, len(index))
        ):
            raise RuntimeError(
                "Binary Search array is not sorted."
            )

    elif algorithm == "Hash-Table Search":

        if not isinstance(index, set):
            raise RuntimeError(
                "Hash Search did not construct a set."
            )

        if len(index) != expected_count:
            raise RuntimeError(
                "Hash index has an incorrect record count."
            )


def run_worker(args):
    """
    Perform one isolated memory measurement.

    The corpus is loaded before tracing starts. Therefore,
    the measurements focus on additional index allocations.
    """

    titles = load_corpus(
        sample_path=args.sample_file,
        expected_count=args.sample_size,
        expected_sha256=args.sample_sha256,
        search_column="title_nfc",
    )

    if len(titles) != args.sample_size:
        raise RuntimeError(
            "Loaded corpus size does not match the expected size."
        )

    # Stabilize the process before establishing a baseline.
    gc.collect()

    process = psutil.Process()

    # Start Python allocation tracing after corpus loading.
    tracemalloc.start()

    gc.collect()

    baseline_traced, _ = tracemalloc.get_traced_memory()

    tracemalloc.reset_peak()

    baseline_rss = process.memory_info().rss

    # Construct only the selected algorithm's index.
    index = build_index(
        args.algorithm,
        titles,
    )

    # Capture Python memory while the index is still alive.
    current_traced, peak_traced = (
        tracemalloc.get_traced_memory()
    )

    # Capture process RSS while the index is still alive.
    after_rss = process.memory_info().rss

    # Verify the index after measurements have been captured.
    verify_index(
        args.algorithm,
        index,
        args.sample_size,
    )

    retained_additional = max(
        0,
        current_traced - baseline_traced,
    )

    peak_additional = max(
        0,
        peak_traced - baseline_traced,
    )

    rss_delta = after_rss - baseline_rss

    result = {
        "algorithm": args.algorithm,
        "operation": {
            "Linear Search": "no_index_baseline",
            "Binary Search": "sorted_array_construction",
            "Hash-Table Search": "hash_index_construction",
        }[args.algorithm],
        "sample_size": args.sample_size,
        "baseline_rss_bytes": baseline_rss,
        "after_rss_bytes": after_rss,
        "rss_delta_bytes": rss_delta,
        "traced_baseline_bytes": baseline_traced,
        "traced_after_bytes": current_traced,
        "traced_retained_additional_bytes": retained_additional,
        "traced_peak_additional_bytes": peak_additional,
        "index_verified": True,
        "status": "PASS",
    }

    # Stop tracing before returning from the worker.
    tracemalloc.stop()

    # Output exactly one JSON object for the parent process.
    print(json.dumps(result))


def run_isolated_measurement(
    algorithm,
    sample,
    trial,
):
    """Execute one algorithm measurement in a fresh process."""

    command = [
        sys.executable,
        "-m",
        "src.memory_measurement",
        "--worker",
        "--algorithm",
        algorithm,
        "--sample-file",
        str(PROJECT_ROOT / sample["sample_file"]),
        "--sample-size",
        str(sample["sample_size"]),
        "--sample-sha256",
        sample["sample_sha256"],
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Memory worker failed for {algorithm}, trial {trial}.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    try:
        result = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid worker output for {algorithm}, trial {trial}:\n"
            f"{completed.stdout}"
        ) from error

    result["trial"] = trial

    return result


def to_mib(value):
    """Convert bytes to MiB."""

    return value / (1024 ** 2)


def summarize_memory(rows):
    """Summarize repeated isolated memory measurements."""

    summary_rows = []

    for algorithm in ALGORITHMS:

        selected = [
            row for row in rows
            if row["algorithm"] == algorithm
        ]

        if len(selected) < 2:
            raise RuntimeError(
                f"Insufficient memory trials for {algorithm}."
            )

        metrics = [
            "rss_delta_bytes",
            "traced_retained_additional_bytes",
            "traced_peak_additional_bytes",
        ]

        summary = {
            "algorithm": algorithm,
            "operation": selected[0]["operation"],
            "sample_size": selected[0]["sample_size"],
            "trials": len(selected),
        }

        for metric in metrics:

            values = [
                int(row[metric])
                for row in selected
            ]

            median_value = statistics.median(values)

            mean_value = statistics.mean(values)

            stdev_value = statistics.stdev(values)

            summary[f"median_{metric}"] = round(
                median_value,
                2,
            )

            summary[f"mean_{metric}"] = round(
                mean_value,
                2,
            )

            summary[f"stdev_{metric}"] = round(
                stdev_value,
                2,
            )

        summary["median_rss_delta_mib"] = round(
            to_mib(
                summary["median_rss_delta_bytes"]
            ),
            4,
        )

        summary["median_traced_retained_mib"] = round(
            to_mib(
                summary[
                    "median_traced_retained_additional_bytes"
                ]
            ),
            4,
        )

        summary["median_traced_peak_mib"] = round(
            to_mib(
                summary[
                    "median_traced_peak_additional_bytes"
                ]
            ),
            4,
        )

        summary["status"] = "PASS"

        summary_rows.append(summary)

    return summary_rows


def write_csv(path, rows):
    """Save the measurement table."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
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


def main():

    parser = argparse.ArgumentParser(
        description="Isolated memory measurement for search indexes."
    )

    parser.add_argument(
        "--worker",
        action="store_true",
    )

    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
    )

    parser.add_argument(
        "--sample-file",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
    )

    parser.add_argument(
        "--sample-sha256",
    )

    parser.add_argument(
        "--language",
        choices=["Tulu", "Kannada", "English"],
        default="Tulu",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    # Internal worker mode.
    if args.worker:

        required = [
            args.algorithm,
            args.sample_file,
            args.sample_size,
            args.sample_sha256,
        ]

        if any(value is None for value in required):
            parser.error(
                "Worker mode requires algorithm and sample arguments."
            )

        run_worker(args)
        return

    if args.trials < 2:
        parser.error(
            "At least two trials are required."
        )

    # Locate exactly the requested 1,000-title or other sample.
    samples = read_manifest()

    matching = [
        row for row in samples
        if row["language"] == args.language
        and int(row["sample_size"]) == args.sample_size
    ]

    if len(matching) != 1:
        parser.error(
            "Requested corpus sample was not found exactly once."
        )

    sample = matching[0]

    if sample["status"] != "PASS":
        parser.error(
            "Requested source sample has not passed validation."
        )

    print("=" * 70)
    print("ISOLATED INDEX MEMORY MEASUREMENT")
    print("=" * 70)

    print("Language:", args.language)
    print("Corpus size:", f"{args.sample_size:,}")
    print("Trials per algorithm:", args.trials)
    print("Memory tools: tracemalloc + psutil")
    print("Process isolation: one fresh subprocess per trial")

    all_rows = []

    for algorithm in ALGORITHMS:

        print("\n" + "-" * 65)
        print("Algorithm:", algorithm)

        for trial in range(1, args.trials + 1):

            result = run_isolated_measurement(
                algorithm=algorithm,
                sample=sample,
                trial=trial,
            )

            if result["status"] != "PASS":
                raise RuntimeError(
                    f"Memory measurement failed: {algorithm}"
                )

            if not result["index_verified"]:
                raise RuntimeError(
                    f"Index verification failed: {algorithm}"
                )

            all_rows.append(result)

            print(
                f"Trial {trial:02d}/{args.trials}: "
                f"RSS delta = "
                f"{to_mib(result['rss_delta_bytes']):.4f} MiB | "
                f"Retained Python memory = "
                f"{to_mib(result['traced_retained_additional_bytes']):.4f} MiB | "
                f"Peak additional Python memory = "
                f"{to_mib(result['traced_peak_additional_bytes']):.4f} MiB"
            )

    summary_rows = summarize_memory(all_rows)

    write_csv(
        RAW_OUTPUT,
        all_rows,
    )

    write_csv(
        SUMMARY_OUTPUT,
        summary_rows,
    )

    metadata = {
        "experiment": "Isolated search-index memory pilot",
        "language": args.language,
        "sample_size": args.sample_size,
        "trials_per_algorithm": args.trials,
        "algorithms": ALGORITHMS,
        "memory_tools": [
            "tracemalloc",
            "psutil",
        ],
        "process_isolation": (
            "Fresh subprocess for each algorithm trial"
        ),
        "corpus_loaded_before_memory_baseline": True,
        "measures": [
            "baseline_process_rss",
            "retained_additional_python_memory",
            "peak_additional_python_memory",
            "process_rss_delta",
        ],
        "status": "PASS",
    }

    METADATA_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METADATA_OUTPUT.open(
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

    print("\n" + "=" * 70)
    print("MEMORY MEASUREMENT SUMMARY")
    print("=" * 70)

    for row in summary_rows:

        print("\n", row["algorithm"])

        print(
            "Median RSS delta:",
            row["median_rss_delta_mib"],
            "MiB",
        )

        print(
            "Median retained Python memory:",
            row["median_traced_retained_mib"],
            "MiB",
        )

        print(
            "Median peak additional Python memory:",
            row["median_traced_peak_mib"],
            "MiB",
        )

    print("\nRaw report:", RAW_OUTPUT)
    print("Summary:", SUMMARY_OUTPUT)
    print("Metadata:", METADATA_OUTPUT)

    print("\nALL MEMORY MEASUREMENT CHECKS PASSED")


if __name__ == "__main__":
    main()
