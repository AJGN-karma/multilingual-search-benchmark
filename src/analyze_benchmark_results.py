from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

BENCHMARK_DIR = TABLES / "benchmark_runs"
MEMORY_SUMMARY_DIR = TABLES / "memory_runs"

RAW_BENCHMARK_DIR = ROOT / "results" / "raw" / "benchmark_runs"
RAW_MEMORY_DIR = ROOT / "results" / "raw" / "memory_runs"

CAMPAIGN_PREFIX = "main_campaign_20261104_"
MEMORY_PREFIX = "main_memory_20261104_"

EXPECTED_BENCHMARK_CONFIGURATIONS = 29
EXPECTED_MEMORY_CONFIGURATIONS = 16
EXPECTED_BENCHMARK_SUMMARY_ROWS = 783
EXPECTED_PREPROCESSING_SUMMARY_ROWS = 87
EXPECTED_MEMORY_SUMMARY_ROWS = 48
EXPECTED_BENCHMARK_TRIAL_ROWS = 7830
EXPECTED_PREPROCESSING_TRIAL_ROWS = 609
EXPECTED_MEMORY_TRIAL_ROWS = 480

BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 2026

ALGORITHMS = [
    "Linear Search",
    "Binary Search",
    "Hash-Table Search",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> pd.DataFrame:
    require(path.exists(), f"Required file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def require_status(
    frame: pd.DataFrame,
    accepted: set[str],
    label: str,
) -> None:
    actual = set(frame["status"].dropna().astype(str).unique())
    unexpected = actual - accepted
    require(
        not unexpected,
        f"{label}: unexpected statuses: {sorted(unexpected)}",
    )


def parse_memory_tag(filename: str) -> tuple[str, int]:
    match = re.fullmatch(
        r"main_memory_20261104_(.+)_n(\d+)_summary\.csv",
        filename,
    )
    require(match is not None, f"Cannot parse memory filename: {filename}")
    return match.group(1), int(match.group(2))


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    destination = TABLES / filename
    frame.to_csv(destination, index=False)
    print(f"Saved: {destination.relative_to(ROOT)} ({len(frame)} rows)")


def load_main_benchmark_data():
    summary_files = sorted(
        BENCHMARK_DIR.glob(f"{CAMPAIGN_PREFIX}*_summary.csv")
    )
    preprocessing_files = sorted(
        BENCHMARK_DIR.glob(f"{CAMPAIGN_PREFIX}*_preprocessing.csv")
    )

    require(
        len(summary_files) == EXPECTED_BENCHMARK_CONFIGURATIONS,
        f"Expected 29 benchmark summaries, found {len(summary_files)}",
    )
    require(
        len(preprocessing_files) == EXPECTED_BENCHMARK_CONFIGURATIONS,
        f"Expected 29 preprocessing summaries, found {len(preprocessing_files)}",
    )

    summaries = []
    preprocessing_summaries = []
    benchmark_trials = []
    raw_preprocessing_trials = []

    run_context = {}

    for path in summary_files:
        run_tag = path.name.removesuffix("_summary.csv")
        frame = read_csv(path)
        frame["run_tag"] = run_tag
        frame["source_report"] = path.name

        require(len(frame) == 27, f"Unexpected rows in {path.name}")
        require_status(frame, {"PASS"}, path.name)

        summaries.append(frame)

        first = frame.iloc[0]
        run_context[run_tag] = {
            "corpus_id": first["corpus_id"],
            "language": first["language"],
            "sample_size": int(first["sample_size"]),
            "query_count": int(first["query_count"]),
        }

        raw_directory = RAW_BENCHMARK_DIR / run_tag
        raw_path = raw_directory / "benchmark_trials.csv"
        raw_frame = read_csv(raw_path)

        raw_frame["run_tag"] = run_tag
        raw_frame["source_file"] = str(raw_path.relative_to(ROOT))

        require(
            len(raw_frame) == 270,
            f"Expected 270 benchmark trials in {run_tag}",
        )
        require_status(raw_frame, {"PASS"}, run_tag)
        benchmark_trials.append(raw_frame)

        raw_preprocessing_path = raw_directory / "preprocessing_trials.csv"
        raw_preprocessing_frame = read_csv(raw_preprocessing_path)
        raw_preprocessing_frame["run_tag"] = run_tag
        raw_preprocessing_frame["language"] = first["language"]
        raw_preprocessing_frame["corpus_id"] = first["corpus_id"]
        raw_preprocessing_frame["query_count"] = int(first["query_count"])
        raw_preprocessing_trials.append(raw_preprocessing_frame)

    for path in preprocessing_files:
        run_tag = path.name.removesuffix("_preprocessing.csv")
        require(run_tag in run_context, f"No benchmark context for {run_tag}")

        frame = read_csv(path)
        context = run_context[run_tag]

        frame["run_tag"] = run_tag
        frame["language"] = context["language"]
        frame["corpus_id"] = context["corpus_id"]
        frame["sample_size"] = context["sample_size"]
        frame["query_count"] = context["query_count"]
        frame["source_report"] = path.name

        require(len(frame) == 3, f"Unexpected preprocessing rows: {path.name}")
        require_status(frame, {"PASS", "NOT_REQUIRED"}, path.name)
        preprocessing_summaries.append(frame)

    benchmark_summary = pd.concat(summaries, ignore_index=True)
    preprocessing_summary = pd.concat(
        preprocessing_summaries,
        ignore_index=True,
    )
    benchmark_raw = pd.concat(benchmark_trials, ignore_index=True)
    preprocessing_raw = pd.concat(
        raw_preprocessing_trials,
        ignore_index=True,
    )

    return (
        benchmark_summary,
        preprocessing_summary,
        benchmark_raw,
        preprocessing_raw,
        run_context,
    )


def load_main_memory_data():
    summary_files = sorted(
        MEMORY_SUMMARY_DIR.glob(f"{MEMORY_PREFIX}*_summary.csv")
    )

    require(
        len(summary_files) == EXPECTED_MEMORY_CONFIGURATIONS,
        f"Expected 16 memory summaries, found {len(summary_files)}",
    )

    summaries = []
    raw_trials = []

    for path in summary_files:
        corpus_id, sample_size = parse_memory_tag(path.name)

        frame = read_csv(path)
        frame["corpus_id"] = corpus_id
        frame["language"] = corpus_id.capitalize()
        frame["source_report"] = path.name

        require(len(frame) == 3, f"Unexpected memory rows: {path.name}")
        require_status(frame, {"PASS"}, path.name)
        summaries.append(frame)

        run_tag = path.name.removesuffix("_summary.csv")
        raw_path = RAW_MEMORY_DIR / run_tag / "memory_trials.csv"
        raw_frame = read_csv(raw_path)

        raw_frame["corpus_id"] = corpus_id
        raw_frame["language"] = corpus_id.capitalize()
        raw_frame["source_file"] = str(raw_path.relative_to(ROOT))

        require(
            len(raw_frame) == 30,
            f"Expected 30 memory records in {run_tag}",
        )
        require_status(raw_frame, {"PASS"}, run_tag)

        require(
            raw_frame["index_verified"].astype(str).str.lower().eq("true").all(),
            f"Index verification failed in {run_tag}",
        )
        raw_trials.append(raw_frame)

    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(raw_trials, ignore_index=True),
    )


def convert_numeric_columns(frame: pd.DataFrame, columns: list[str]):
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")


def calculate_trial_statistics(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "corpus_id",
        "language",
        "sample_size",
        "workload_type",
        "repetition_pattern",
        "query_count",
        "algorithm",
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for group_key, group in raw.groupby(keys, sort=True):
        values = group["ns_per_query"].to_numpy(dtype=float)

        require(len(values) == 10, f"Expected 10 trials for {group_key}")
        require(np.isfinite(values).all(), f"Non-finite latency in {group_key}")

        bootstrap = rng.choice(
            values,
            size=(BOOTSTRAP_REPLICATES, len(values)),
            replace=True,
        )
        bootstrap_medians = np.median(bootstrap, axis=1)
        ci_low, ci_high = np.quantile(
            bootstrap_medians,
            [0.025, 0.975],
        )

        record = dict(zip(keys, group_key))
        record.update({
            "trials": len(values),
            "mean_ns_per_query": float(np.mean(values)),
            "median_ns_per_query": float(np.median(values)),
            "stdev_ns_per_query": float(np.std(values, ddof=1)),
            "minimum_ns_per_query": float(np.min(values)),
            "maximum_ns_per_query": float(np.max(values)),
            "p95_ns_per_query": float(np.quantile(values, 0.95)),
            "bootstrap_95ci_median_lower_ns": float(ci_low),
            "bootstrap_95ci_median_upper_ns": float(ci_high),
        })
        records.append(record)

    return pd.DataFrame(records)


def calculate_speedups(summary: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "corpus_id",
        "language",
        "sample_size",
        "workload_type",
        "repetition_pattern",
        "query_count",
    ]

    wide = summary.pivot(
        index=keys,
        columns="algorithm",
        values="median_ns_per_query",
    ).reset_index()

    wide.columns.name = None

    for algorithm in ALGORITHMS:
        require(algorithm in wide.columns, f"Missing algorithm: {algorithm}")

    wide["linear_to_binary_speedup"] = (
        wide["Linear Search"] / wide["Binary Search"]
    )
    wide["linear_to_hash_speedup"] = (
        wide["Linear Search"] / wide["Hash-Table Search"]
    )
    wide["binary_to_hash_speedup"] = (
        wide["Binary Search"] / wide["Hash-Table Search"]
    )

    wide["fastest_algorithm"] = wide[
        ALGORITHMS
    ].idxmin(axis=1)

    return wide


def calculate_preprocessing_costs(
    preprocessing: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        preprocessing
        .groupby(
            ["corpus_id", "language", "sample_size", "algorithm", "operation"],
            as_index=False,
        )
        .agg(
            report_instances=("median_ns", "count"),
            median_preprocessing_ns=("median_ns", "median"),
            mean_preprocessing_ns=("mean_ns", "mean"),
        )
    )
    return result


def calculate_break_even(
    benchmark_summary: pd.DataFrame,
    preprocessing_costs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "corpus_id",
        "language",
        "sample_size",
        "workload_type",
        "repetition_pattern",
        "query_count",
    ]

    wide = benchmark_summary.pivot(
        index=keys,
        columns="algorithm",
        values="median_ns_per_query",
    ).reset_index()
    wide.columns.name = None

    prep_lookup = {
        (row.corpus_id, int(row.sample_size), row.algorithm):
            float(row.median_preprocessing_ns)
        for row in preprocessing_costs.itertuples(index=False)
    }

    records = []

    for row in wide.to_dict("records"):
        linear_latency = float(row["Linear Search"])

        for algorithm in ["Binary Search", "Hash-Table Search"]:
            indexed_latency = float(row[algorithm])
            prep_ns = prep_lookup.get(
                (row["corpus_id"], int(row["sample_size"]), algorithm)
            )

            saving_per_query = linear_latency - indexed_latency

            if prep_ns is None:
                break_even = np.nan
            elif saving_per_query <= 0:
                break_even = np.nan
            else:
                break_even = prep_ns / saving_per_query

            records.append({
                **{key: row[key] for key in keys},
                "indexed_algorithm": algorithm,
                "linear_ns_per_query": linear_latency,
                "indexed_ns_per_query": indexed_latency,
                "preprocessing_ns": prep_ns,
                "saving_per_query_ns": saving_per_query,
                "estimated_break_even_queries": break_even,
                "break_even_status": (
                    "FINITE_ESTIMATE"
                    if np.isfinite(break_even)
                    else "NO_POSITIVE_BREAK_EVEN"
                ),
            })

    return pd.DataFrame(records)


def make_latency_scaling_figure(summary: pd.DataFrame):
    data = summary[
        (summary["query_count"] == 100)
        & (summary["workload_type"] == "mixed")
        & (summary["repetition_pattern"] == "mixed")
    ].copy()

    require(not data.empty, "No common 100-query mixed workload data")

    languages = ["Tulu", "Kannada", "English"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, language in zip(axes, languages):
        subset = data[data["language"] == language]

        for algorithm in ALGORITHMS:
            points = subset[subset["algorithm"] == algorithm].sort_values(
                "sample_size"
            )
            if points.empty:
                continue

            ax.plot(
                points["sample_size"],
                points["median_ns_per_query"],
                marker="o",
                label=algorithm,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(language)
        ax.set_xlabel("Corpus size (titles)")
        ax.set_ylabel("Median query latency (ns/query)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Exact-search latency scaling: common 100-query mixed workload"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "latency_scaling_100_query_mixed.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_workload_figure(summary: pd.DataFrame):
    data = summary[
        (summary["corpus_id"] == "english")
        & (summary["sample_size"] == 100000)
        & (summary["query_count"] == 100)
    ].copy()

    require(not data.empty, "English 100,000-title workload data not found")

    pivot = data.pivot(
        index=["workload_type", "repetition_pattern"],
        columns="algorithm",
        values="median_ns_per_query",
    )

    order = [
        (workload, repetition)
        for workload in ["successful", "unsuccessful", "mixed"]
        for repetition in ["same_query_set", "mostly_unique", "mixed"]
    ]
    pivot = pivot.reindex(order)

    x = np.arange(len(pivot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for index, algorithm in enumerate(ALGORITHMS):
        ax.bar(
            x + (index - 1) * width,
            pivot[algorithm].to_numpy() / 1000,
            width,
            label=algorithm,
        )

    labels = [
        f"{workload}\n{repetition}"
        for workload, repetition in pivot.index
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Median query latency (μs/query)")
    ax.set_title(
        "Algorithm latency by workload and repetition pattern\n"
        "English corpus: 100,000 titles; 100 queries"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES / "workload_and_repetition_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_preprocessing_figure(costs: pd.DataFrame):
    data = costs[
        costs["algorithm"].isin(["Binary Search", "Hash-Table Search"])
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, language in zip(axes, ["Tulu", "Kannada", "English"]):
        subset = data[data["language"] == language]

        for algorithm in ["Binary Search", "Hash-Table Search"]:
            points = subset[subset["algorithm"] == algorithm].sort_values(
                "sample_size"
            )
            if points.empty:
                continue

            ax.plot(
                points["sample_size"],
                points["median_preprocessing_ns"] / 1e6,
                marker="o",
                label=algorithm,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(language)
        ax.set_xlabel("Corpus size (titles)")
        ax.set_ylabel("Median preprocessing time (ms)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Index construction cost across corpus sizes")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "preprocessing_cost_scaling.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_memory_figure(memory: pd.DataFrame, metric: str, filename: str, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, language in zip(axes, ["Tulu", "Kannada", "English"]):
        subset = memory[memory["language"] == language]

        for algorithm in ALGORITHMS:
            points = subset[subset["algorithm"] == algorithm].sort_values(
                "sample_size"
            )
            if points.empty:
                continue

            ax.plot(
                points["sample_size"],
                points[metric],
                marker="o",
                label=algorithm,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(language)
        ax.set_xlabel("Corpus size (titles)")
        ax.set_ylabel("Additional memory (MiB)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(
        FIGURES / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Loading main benchmark and preprocessing reports...")
    (
        benchmark_summary,
        preprocessing_summary,
        benchmark_raw,
        preprocessing_raw,
        run_context,
    ) = load_main_benchmark_data()

    print("Loading main memory reports...")
    memory_summary, memory_raw = load_main_memory_data()

    convert_numeric_columns(
        benchmark_summary,
        [
            "sample_size", "query_count", "trials", "warmups",
            "median_batch_ns", "mean_batch_ns", "stdev_batch_ns",
            "median_ns_per_query", "queries_per_second",
        ],
    )
    convert_numeric_columns(
        benchmark_raw,
        [
            "sample_size", "query_count", "trial", "elapsed_ns",
            "elapsed_ms", "ns_per_query", "queries_per_second",
        ],
    )
    convert_numeric_columns(
        preprocessing_summary,
        [
            "sample_size", "query_count", "corpus_size", "trials",
            "median_ns", "mean_ns", "stdev_ns",
        ],
    )
    convert_numeric_columns(
        preprocessing_raw,
        [
            "corpus_size", "query_count", "trial",
            "elapsed_ns", "elapsed_ms",
        ],
    )
    convert_numeric_columns(
        memory_summary,
        [
            "sample_size", "trials",
            "median_rss_delta_bytes", "median_rss_delta_mib",
            "median_traced_retained_mib", "median_traced_peak_mib",
        ],
    )
    convert_numeric_columns(
        memory_raw,
        [
            "sample_size", "trial", "rss_delta_bytes",
            "traced_retained_additional_bytes",
            "traced_peak_additional_bytes",
        ],
    )

    print("\n===== VALIDATING MAIN CAMPAIGNS =====")

    observed_counts = {
        "benchmark summaries": len(benchmark_summary),
        "preprocessing summaries": len(preprocessing_summary),
        "memory summaries": len(memory_summary),
        "benchmark raw trials": len(benchmark_raw),
        "preprocessing raw trials": len(preprocessing_raw),
        "memory raw trials": len(memory_raw),
    }

    expected_counts = {
        "benchmark summaries": EXPECTED_BENCHMARK_SUMMARY_ROWS,
        "preprocessing summaries": EXPECTED_PREPROCESSING_SUMMARY_ROWS,
        "memory summaries": EXPECTED_MEMORY_SUMMARY_ROWS,
        "benchmark raw trials": EXPECTED_BENCHMARK_TRIAL_ROWS,
        "preprocessing raw trials": EXPECTED_PREPROCESSING_TRIAL_ROWS,
        "memory raw trials": EXPECTED_MEMORY_TRIAL_ROWS,
    }

    for label, observed in observed_counts.items():
        expected = expected_counts[label]
        print(f"{label}: {observed} (expected {expected})")
        require(observed == expected, f"Count mismatch for {label}")

    require_status(benchmark_summary, {"PASS"}, "benchmark summaries")
    require_status(benchmark_raw, {"PASS"}, "benchmark trials")
    require_status(
        preprocessing_summary,
        {"PASS", "NOT_REQUIRED"},
        "preprocessing summaries",
    )
    require_status(
        preprocessing_raw,
        {"PASS", "NOT_REQUIRED"},
        "preprocessing trials",
    )
    require_status(memory_summary, {"PASS"}, "memory summaries")
    require_status(memory_raw, {"PASS"}, "memory trials")

    require(
        benchmark_summary["algorithm"].nunique() == 3,
        "Expected three benchmark algorithms",
    )
    require(
        memory_summary["algorithm"].nunique() == 3,
        "Expected three memory algorithms",
    )

    print("\nAll main campaign source counts and statuses verified.")

    print("\n===== CALCULATING STATISTICS =====")
    trial_statistics = calculate_trial_statistics(benchmark_raw)
    require(
        len(trial_statistics) == EXPECTED_BENCHMARK_SUMMARY_ROWS,
        "Unexpected benchmark trial-statistics row count",
    )

    speedups = calculate_speedups(benchmark_summary)
    preprocessing_costs = calculate_preprocessing_costs(
        preprocessing_summary
    )
    break_even = calculate_break_even(
        benchmark_summary,
        preprocessing_costs,
    )

    performance_summary = (
        benchmark_summary
        .groupby(
            ["language", "algorithm", "workload_type", "repetition_pattern"],
            as_index=False,
        )
        .agg(
            configurations=("median_ns_per_query", "count"),
            median_latency_ns_per_query=("median_ns_per_query", "median"),
            mean_latency_ns_per_query=("median_ns_per_query", "mean"),
        )
    )

    memory_growth = memory_summary[
        [
            "corpus_id",
            "language",
            "sample_size",
            "algorithm",
            "operation",
            "median_rss_delta_mib",
            "median_traced_retained_mib",
            "median_traced_peak_mib",
            "status",
        ]
    ].copy()

    outputs = [
        (benchmark_summary, "consolidated_benchmark_summary.csv"),
        (preprocessing_summary, "consolidated_preprocessing_summary.csv"),
        (memory_summary, "consolidated_memory_summary.csv"),
        (benchmark_raw, "consolidated_benchmark_trials.csv"),
        (preprocessing_raw, "consolidated_preprocessing_trials.csv"),
        (memory_raw, "consolidated_memory_trials.csv"),
        (trial_statistics, "benchmark_trial_statistics.csv"),
        (speedups, "algorithm_speedup_comparison.csv"),
        (preprocessing_costs, "preprocessing_cost_by_corpus.csv"),
        (break_even, "preprocessing_break_even_estimates.csv"),
        (performance_summary, "performance_by_language_workload.csv"),
        (memory_growth, "memory_growth_summary.csv"),
    ]

    print("\n===== SAVING CONSOLIDATED TABLES =====")
    for frame, filename in outputs:
        save_csv(frame, filename)

    print("\n===== GENERATING FIGURES =====")
    make_latency_scaling_figure(benchmark_summary)
    print("Created latency_scaling_100_query_mixed.png")

    make_workload_figure(benchmark_summary)
    print("Created workload_and_repetition_comparison.png")

    make_preprocessing_figure(preprocessing_costs)
    print("Created preprocessing_cost_scaling.png")

    make_memory_figure(
        memory_summary,
        "median_traced_retained_mib",
        "memory_retained_scaling.png",
        "Additional retained memory by index structure",
    )
    print("Created memory_retained_scaling.png")

    make_memory_figure(
        memory_summary,
        "median_traced_peak_mib",
        "memory_peak_scaling.png",
        "Peak additional memory by index structure",
    )
    print("Created memory_peak_scaling.png")

    validation = {
        "analysis_status": "PASS",
        "campaign_prefix": CAMPAIGN_PREFIX,
        "memory_prefix": MEMORY_PREFIX,
        "benchmark_configurations": len(run_context),
        "memory_configurations": EXPECTED_MEMORY_CONFIGURATIONS,
        "source_row_counts": observed_counts,
        "consolidated_table_rows": {
            filename: len(frame) for frame, filename in outputs
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "figures": [
            "latency_scaling_100_query_mixed.png",
            "workload_and_repetition_comparison.png",
            "preprocessing_cost_scaling.png",
            "memory_retained_scaling.png",
            "memory_peak_scaling.png",
        ],
        "interpretation_notes": [
            "Bootstrap intervals describe uncertainty in the observed trial median; they are not, by themselves, proof of statistical significance.",
            "Break-even query counts are estimates based on measured median preprocessing and per-query latency.",
            "Memory measurements describe additional index construction allocation after the corpus has been loaded, not total process memory.",
            "Linear Search is the no-index baseline.",
            "Corpus scaling graph uses the common 100-query mixed workload and mixed repetition pattern.",
        ],
    }

    validation_path = TABLES / "analysis_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {validation_path.relative_to(ROOT)}")
    print("\n===== ANALYSIS COMPLETE =====")
    print("Source campaigns: all verified")
    print("Consolidated CSV tables:", len(outputs))
    print("Research figures: 5")
    print("Validation status: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nANALYSIS FAILED: {exc}", file=sys.stderr)
        raise
