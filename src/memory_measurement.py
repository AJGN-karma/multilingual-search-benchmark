
import argparse
import csv
import gc
import json
import re
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

RAW_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "memory_runs"
)

TABLE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "memory_runs"
)

ALGORITHMS = [
    "Linear Search",
    "Binary Search",
    "Hash-Table Search",
]


def read_manifest():
    """Read the validated corpus sample manifest."""

    if not SAMPLE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Sample manifest not found: {SAMPLE_MANIFEST}"
        )

    with SAMPLE_MANIFEST.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def validate_run_tag(run_tag):
    """Reject unsafe run tags and path traversal."""

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}",
        run_tag,
    ):
        raise ValueError(
            "Run tag must start with a letter or number and "
            "contain only letters, numbers, underscores, "
            "and hyphens. Maximum length: 80 characters."
        )


def get_output_paths(run_tag):
    """Build unique output paths for one memory run."""

    raw_directory = RAW_OUTPUT_ROOT / run_tag

    raw_output = (
        raw_directory
        / "memory_trials.csv"
    )

    summary_output = (
        TABLE_OUTPUT_ROOT
        / f"{run_tag}_summary.csv"
    )

    metadata_output = (
        TABLE_OUTPUT_ROOT
        / f"{run_tag}_metadata.json"
    )

    return {
        "raw": raw_output,
        "summary": summary_output,
        "metadata": metadata_output,
    }


def check_output_paths(output_paths):
    """Prevent accidental overwriting of an earlier run."""

    existing = [
        str(path)
        for path in output_paths.values()
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "One or more output files already exist. "
            "Choose a new --run-tag.\n"
            + "\n".join(existing)
        )


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
    measurements focus on additional index allocations.
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

    tracemalloc.stop()

    # Output exactly one JSON object for the parent process.
    print(json.dumps(result))


def run_isolated_measurement(
    algorithm,
    sample,
    trial,
):
    """Execute one algorithm measurement in a fresh process."""

    sample_path = PROJECT_ROOT / sample["sample_file"]

    command = [
        sys.executable,
        "-m",
        "src.memory_measurement",
        "--worker",
        "--algorithm",
        algorithm,
        "--sample-file",
        str(sample_path),
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

    if not rows:
        raise ValueError(
            f"Cannot write an empty CSV report: {path}"
        )

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
        description=(
            "Isolated memory measurement for exact-search indexes."
        )
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

    parser.add_argument(
        "--run-tag",
        default="memory_run",
        help="Unique identifier for this memory measurement run.",
    )

    args = parser.parse_args()

    # Internal worker mode.
    # The worker does not write reports, so it needs no run tag.
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

        if args.sample_size < 1:
            parser.error(
                "Worker sample size must be positive."
            )

        run_worker(args)
        return

    if args.trials < 2:
        parser.error(
            "At least two trials are required."
        )

    if args.sample_size is None or args.sample_size < 1:
        parser.error(
            "A positive --sample-size is required."
        )

    # Validate the run tag before building output paths.
    try:
        validate_run_tag(args.run_tag)

    except ValueError as error:
        parser.error(str(error))

    output_paths = get_output_paths(args.run_tag)

    try:
        check_output_paths(output_paths)

    except FileExistsError as error:
        parser.error(str(error))

    # Locate exactly the requested corpus sample.
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
            "Requested corpus sample has not passed validation."
        )

    print("=" * 70)
    print("ISOLATED INDEX MEMORY MEASUREMENT")
    print("=" * 70)

    print("Run tag:", args.run_tag)
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

    # Verify the expected number of trial records.
    expected_records = len(ALGORITHMS) * args.trials

    if len(all_rows) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} measurements, "
            f"but recorded {len(all_rows)}."
        )

    summary_rows = summarize_memory(all_rows)

    if len(summary_rows) != len(ALGORITHMS):
        raise RuntimeError(
            "Memory summary does not contain all algorithms."
        )

    # Write reports only after all measurements have succeeded.
    write_csv(
        output_paths["raw"],
        all_rows,
    )

    write_csv(
        output_paths["summary"],
        summary_rows,
    )

    metadata = {
        "experiment": "Isolated search-index memory measurement",
        "run_tag": args.run_tag,
        "language": args.language,
        "sample_size": args.sample_size,
        "sample_file": sample["sample_file"],
        "sample_sha256": sample["sample_sha256"],
        "trials_per_algorithm": args.trials,
        "total_measurements": len(all_rows),
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
        "interpretation_notes": [
            "RSS delta measures process-level memory change.",
            "tracemalloc measures traced Python memory allocations.",
            "RSS and tracemalloc measure different aspects of memory.",
            "A zero RSS delta does not imply zero additional memory use."
        ],
        "output_files": {
            "raw_trials": str(
                output_paths["raw"].relative_to(PROJECT_ROOT)
            ),
            "summary": str(
                output_paths["summary"].relative_to(PROJECT_ROOT)
            ),
            "metadata": str(
                output_paths["metadata"].relative_to(PROJECT_ROOT)
            ),
        },
        "status": "PASS",
    }

    output_paths["metadata"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_paths["metadata"].open(
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

    print("\nRaw report:", output_paths["raw"])
    print("Summary:", output_paths["summary"])
    print("Metadata:", output_paths["metadata"])

    print("\nALL MEMORY MEASUREMENT CHECKS PASSED")


if __name__ == "__main__":
    main()
