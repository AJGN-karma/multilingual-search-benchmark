
import argparse
import csv
import json
import re
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

from src.benchmark_workloads import read_manifest


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

CAMPAIGN_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "benchmark_campaigns"
)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

ALGORITHMS = [
    "Linear Search",
    "Binary Search",
    "Hash-Table Search",
]

CSV_FIELDS = [
    "sequence",
    "run_tag",
    "language",
    "sample_size",
    "query_count",
    "status",
    "return_code",
    "started_at_utc",
    "completed_at_utc",
    "elapsed_seconds",
    "summary_path",
    "raw_trials_path",
    "metadata_path",
    "error",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now():
    """Return the current UTC time in ISO format."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def load_json(path):
    """Read a JSON configuration file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_campaign_tag(tag):
    """Validate the campaign identifier."""

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}",
        tag,
    ):
        raise ValueError(
            "Campaign tag must start with a letter or number "
            "and contain only letters, numbers, underscores, "
            "or hyphens. Maximum length is 40 characters."
        )


def build_run_tag(campaign_tag, language, sample_size, query_count):
    """Generate a unique benchmark run identifier."""

    tag = (
        f"{campaign_tag}_{language.lower()}"
        f"_n{sample_size}_q{query_count}"
    )

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}",
        tag,
    ):
        raise ValueError(
            f"Generated benchmark tag is invalid: {tag}"
        )

    return tag


# ============================================================
# SCHEDULE GENERATION
# ============================================================

def select_query_counts(sample_size, policies):
    """Select query counts using the configured size policy."""

    matches = [
        policy
        for policy in policies
        if (
            int(policy["minimum_sample_size"])
            <= sample_size
            <= int(policy["maximum_sample_size"])
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one query policy for "
            f"sample size {sample_size}; found {len(matches)}."
        )

    return [
        int(value)
        for value in matches[0]["query_counts"]
    ]


def find_exactly_one(rows, description):
    """Require exactly one matching manifest record."""

    if len(rows) != 1:
        raise RuntimeError(
            f"{description}: expected exactly one record, "
            f"found {len(rows)}."
        )

    return rows[0]


def build_experiment_plan(config, sample_rows, workload_rows):
    """
    Construct and validate all campaign configurations.

    Checks sample manifests and the workload files required
    by the benchmark engine before running any experiments.
    """

    schedule = config["main_benchmark_schedule"]

    policies = schedule["sample_size_query_policy"]

    workload_types = schedule["workload_types"]

    repetition_patterns = schedule["repetition_patterns"]

    allowed_query_counts = {
        int(value)
        for value in config["query_workloads"]["query_counts"]
    }

    plan = []

    for dataset in config["datasets"].values():

        language = dataset["language"]

        for sample_size in dataset["candidate_corpus_sizes"]:

            sample_size = int(sample_size)

            sample_matches = [
                row for row in sample_rows
                if row["language"] == language
                and int(row["sample_size"]) == sample_size
            ]

            sample = find_exactly_one(
                sample_matches,
                f"{language} sample n={sample_size}",
            )

            if sample["status"] != "PASS":
                raise RuntimeError(
                    f"Sample has not passed validation: "
                    f"{language}, n={sample_size}"
                )

            sample_file = PROJECT_ROOT / sample["sample_file"]

            if not sample_file.is_file():
                raise FileNotFoundError(
                    f"Sample file is missing: {sample_file}"
                )

            query_counts = select_query_counts(
                sample_size,
                policies,
            )

            for query_count in query_counts:

                if query_count not in allowed_query_counts:
                    raise RuntimeError(
                        f"Query count {query_count} is not "
                        "present in query_workloads.query_counts."
                    )

                required_workloads = set(workload_types)

                # The provider always loads this negative
                # candidate pool during initialization.
                required_workloads.add("unsuccessful")

                for workload_type in required_workloads:

                    source_matches = [
                        row for row in workload_rows
                        if row["corpus_id"] == sample["corpus_id"]
                        and int(row["sample_size"]) == sample_size
                        and int(row["query_count"]) == (
                            100000
                            if workload_type == "unsuccessful"
                            and query_count != 100000
                            and workload_type not in workload_types
                            else query_count
                        )
                        and row["workload_type"] == workload_type
                        and row["status"] == "PASS"
                    ]

                    # The provider requires a 100,000-query
                    # negative pool for every corpus.
                    if workload_type == "unsuccessful":
                        negative_pool_matches = [
                            row for row in workload_rows
                            if row["corpus_id"] == sample["corpus_id"]
                            and int(row["sample_size"]) == sample_size
                            and int(row["query_count"]) == 100000
                            and row["workload_type"] == "unsuccessful"
                            and row["status"] == "PASS"
                        ]

                        find_exactly_one(
                            negative_pool_matches,
                            (
                                f"{language} n={sample_size} "
                                "100,000-query negative pool"
                            ),
                        )

                    find_exactly_one(
                        source_matches,
                        (
                            f"{language} n={sample_size} "
                            f"q={query_count} {workload_type} workload"
                        ),
                    )

                    source_row = source_matches[0]

                    workload_file = (
                        PROJECT_ROOT / source_row["workload_file"]
                    )

                    if not workload_file.is_file():
                        raise FileNotFoundError(
                            f"Workload file is missing: {workload_file}"
                        )

                run_tag_placeholder = (
                    f"{language.lower()}_n{sample_size}_q{query_count}"
                )

                plan.append({
                    "language": language,
                    "sample_size": sample_size,
                    "query_count": query_count,
                    "corpus_id": sample["corpus_id"],
                })

    expected_combinations = int(
        schedule["total_corpus_query_combinations"]
    )

    if len(plan) != expected_combinations:
        raise RuntimeError(
            f"Schedule mismatch: configuration declares "
            f"{expected_combinations} combinations, "
            f"but manifests and policies produce {len(plan)}."
        )

    trials = int(schedule["trials"])
    warmups = int(schedule["warmup_runs"])

    expected_measured = (
        len(plan)
        * len(workload_types)
        * len(repetition_patterns)
        * len(ALGORITHMS)
        * trials
    )

    expected_warmups = (
        len(plan)
        * len(workload_types)
        * len(repetition_patterns)
        * len(ALGORITHMS)
        * warmups
    )

    if expected_measured != int(
        schedule["total_measured_query_batches"]
    ):
        raise RuntimeError(
            "Declared measured batch total does not match "
            "the experiment schedule."
        )

    if expected_warmups != int(
        schedule["total_warmup_query_batches"]
    ):
        raise RuntimeError(
            "Declared warm-up batch total does not match "
            "the experiment schedule."
        )

    if trials != int(config["benchmark"]["trials"]):
        raise RuntimeError(
            "Schedule trial count differs from benchmark configuration."
        )

    if warmups != int(config["benchmark"]["warmup_runs"]):
        raise RuntimeError(
            "Schedule warm-up count differs from benchmark configuration."
        )

    return plan


# ============================================================
# OUTPUT COLLISION PROTECTION
# ============================================================

def check_run_output_collisions(plan, campaign_tag):
    """Prevent overwriting previous benchmark or campaign outputs."""

    campaign_directory = (
        CAMPAIGN_OUTPUT_ROOT / campaign_tag
    )

    if campaign_directory.exists():
        raise FileExistsError(
            f"Campaign directory already exists: "
            f"{campaign_directory}"
        )

    for item in plan:

        run_tag = build_run_tag(
            campaign_tag,
            item["language"],
            item["sample_size"],
            item["query_count"],
        )

        paths = [
            RAW_OUTPUT_ROOT / run_tag,
            TABLE_OUTPUT_ROOT / f"{run_tag}_summary.csv",
            TABLE_OUTPUT_ROOT / f"{run_tag}_preprocessing.csv",
            TABLE_OUTPUT_ROOT / f"{run_tag}_metadata.json",
        ]

        for path in paths:

            if path.exists():
                raise FileExistsError(
                    f"Benchmark output already exists: {path}. "
                    "Use a different campaign tag."
                )


# ============================================================
# CAMPAIGN MANIFEST
# ============================================================

def campaign_counts(rows):
    """Count campaign records by status."""

    return {
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": sum(row["status"] == "FAIL" for row in rows),
        "running": sum(row["status"] == "RUNNING" for row in rows),
        "interrupted": sum(
            row["status"] == "INTERRUPTED"
            for row in rows
        ),
        "pending": sum(row["status"] == "PENDING" for row in rows),
    }


def write_campaign_state(
    campaign_directory,
    rows,
    metadata,
    campaign_status,
):
    """Write campaign progress atomically."""

    manifest_path = (
        campaign_directory / "campaign_manifest.csv"
    )

    temporary_csv = (
        campaign_directory / "campaign_manifest.csv.tmp"
    )

    with temporary_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary_csv.replace(manifest_path)

    metadata["status"] = campaign_status
    metadata["updated_at_utc"] = utc_now()
    metadata["counts"] = campaign_counts(rows)

    metadata_path = (
        campaign_directory / "campaign_metadata.json"
    )

    temporary_json = (
        campaign_directory / "campaign_metadata.json.tmp"
    )

    with temporary_json.open(
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

    temporary_json.replace(metadata_path)


# ============================================================
# BENCHMARK SUBPROCESS
# ============================================================

def run_one_configuration(item, run_tag):
    """Launch one existing benchmark-engine configuration."""

    command = [
        sys.executable,
        "-m",
        "src.benchmark_engine",
        "--language",
        item["language"],
        "--sample-size",
        str(item["sample_size"]),
        "--query-count",
        str(item["query_count"]),
        "--run-tag",
        run_tag,
    ]

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )


# ============================================================
# MAIN CAMPAIGN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Execute the configured multilingual exact-search "
            "benchmark campaign."
        )
    )

    parser.add_argument(
        "--campaign-tag",
        required=True,
        help="Unique campaign identifier.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N configurations (for smoke testing).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the schedule without running benchmarks.",
    )

    args = parser.parse_args()

    try:
        validate_campaign_tag(args.campaign_tag)

        if args.limit is not None and args.limit < 1:
            raise ValueError(
                "--limit must be a positive integer."
            )

        config = load_json(CONFIG_PATH)

        sample_rows = read_manifest(
            SAMPLE_MANIFEST_PATH
        )

        workload_rows = read_manifest(
            WORKLOAD_MANIFEST_PATH
        )

        complete_plan = build_experiment_plan(
            config,
            sample_rows,
            workload_rows,
        )

        selected_plan = (
            complete_plan[:args.limit]
            if args.limit is not None
            else complete_plan
        )

        check_run_output_collisions(
            selected_plan,
            args.campaign_tag,
        )

    except Exception as error:
        parser.error(str(error))

    # --------------------------------------------------------
    # Display validated campaign plan.
    # --------------------------------------------------------

    print("=" * 75)
    print("MULTILINGUAL EXACT-SEARCH BENCHMARK CAMPAIGN")
    print("=" * 75)

    print("Campaign tag:", args.campaign_tag)
    print("Total configured combinations:", len(complete_plan))
    print("Selected combinations:", len(selected_plan))

    print(
        "Workload configurations per combination:",
        len(config["main_benchmark_schedule"]["workload_types"])
        * len(config["main_benchmark_schedule"]["repetition_patterns"]),
    )

    print("Algorithms:", len(ALGORITHMS))
    print("Trials:", config["main_benchmark_schedule"]["trials"])
    print("Warm-ups:", config["main_benchmark_schedule"]["warmup_runs"])

    print("\nValidated execution schedule:")

    for number, item in enumerate(
        selected_plan,
        start=1,
    ):

        print(
            f"{number:02d}. "
            f"{item['language']:8} | "
            f"Corpus: {item['sample_size']:>8,} | "
            f"Queries: {item['query_count']:>6,}"
        )

    if args.dry_run:

        print("\nDRY RUN PASSED.")
        print("No benchmark process was started.")
        print("No campaign output files were created.")

        return 0

    # --------------------------------------------------------
    # Create campaign files.
    # --------------------------------------------------------

    campaign_directory = (
        CAMPAIGN_OUTPUT_ROOT / args.campaign_tag
    )

    campaign_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    rows = []

    for sequence, item in enumerate(
        selected_plan,
        start=1,
    ):

        run_tag = build_run_tag(
            args.campaign_tag,
            item["language"],
            item["sample_size"],
            item["query_count"],
        )

        rows.append({
            "sequence": sequence,
            "run_tag": run_tag,
            "language": item["language"],
            "sample_size": item["sample_size"],
            "query_count": item["query_count"],
            "status": "PENDING",
            "return_code": "",
            "started_at_utc": "",
            "completed_at_utc": "",
            "elapsed_seconds": "",
            "summary_path": (
                f"results/tables/benchmark_runs/"
                f"{run_tag}_summary.csv"
            ),
            "raw_trials_path": (
                f"results/raw/benchmark_runs/"
                f"{run_tag}/benchmark_trials.csv"
            ),
            "metadata_path": (
                f"results/tables/benchmark_runs/"
                f"{run_tag}_metadata.json"
            ),
            "error": "",
        })

    metadata = {
        "experiment": (
            "Performance Evaluation of Exact-Search "
            "Data Structures for Multilingual Title Retrieval"
        ),
        "campaign_tag": args.campaign_tag,
        "created_at_utc": utc_now(),
        "planned_configurations": len(complete_plan),
        "selected_configurations": len(selected_plan),
        "workload_types": (
            config["main_benchmark_schedule"]["workload_types"]
        ),
        "repetition_patterns": (
            config["main_benchmark_schedule"]["repetition_patterns"]
        ),
        "algorithms": ALGORITHMS,
        "trials": int(
            config["main_benchmark_schedule"]["trials"]
        ),
        "warmup_runs": int(
            config["main_benchmark_schedule"]["warmup_runs"]
        ),
        "schedule": selected_plan,
    }

    write_campaign_state(
        campaign_directory,
        rows,
        metadata,
        "RUNNING",
    )

    # --------------------------------------------------------
    # Execute each benchmark configuration.
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("CAMPAIGN EXECUTION STARTED")
    print("=" * 75)

    campaign_status = "RUNNING"

    for index, row in enumerate(rows):

        item = selected_plan[index]

        print("\n")
        print("#" * 75)
        print(
            f"CAMPAIGN RUN {index + 1}/{len(rows)}: "
            f"{item['language']} n={item['sample_size']} "
            f"q={item['query_count']}"
        )
        print("#" * 75)

        row["status"] = "RUNNING"
        row["started_at_utc"] = utc_now()

        write_campaign_state(
            campaign_directory,
            rows,
            metadata,
            "RUNNING",
        )

        start_time = time.perf_counter()

        try:
            completed = run_one_configuration(
                item,
                row["run_tag"],
            )

            row["return_code"] = completed.returncode

            row["elapsed_seconds"] = round(
                time.perf_counter() - start_time,
                3,
            )

            row["completed_at_utc"] = utc_now()

            if completed.returncode == 0:

                row["status"] = "PASS"

                print(
                    f"\nCampaign configuration "
                    f"{index + 1} completed successfully."
                )

            else:

                row["status"] = "FAIL"

                row["error"] = (
                    f"Benchmark engine returned exit code "
                    f"{completed.returncode}."
                )

                campaign_status = "FAILED"

        except KeyboardInterrupt:

            row["status"] = "INTERRUPTED"

            row["completed_at_utc"] = utc_now()

            row["elapsed_seconds"] = round(
                time.perf_counter() - start_time,
                3,
            )

            row["error"] = "Campaign interrupted by user."

            campaign_status = "INTERRUPTED"

        except OSError as error:

            row["status"] = "FAIL"

            row["completed_at_utc"] = utc_now()

            row["elapsed_seconds"] = round(
                time.perf_counter() - start_time,
                3,
            )

            row["error"] = str(error)

            campaign_status = "FAILED"

        if campaign_status != "RUNNING":

            write_campaign_state(
                campaign_directory,
                rows,
                metadata,
                campaign_status,
            )

            print(
                "\nCampaign stopped. "
                "Remaining configurations were not executed."
            )

            break

        write_campaign_state(
            campaign_directory,
            rows,
            metadata,
            "RUNNING",
        )

    # --------------------------------------------------------
    # Final campaign status.
    # --------------------------------------------------------

    if campaign_status == "RUNNING":

        if len(selected_plan) == len(complete_plan):
            campaign_status = "COMPLETED"
        else:
            campaign_status = "PARTIAL"

    write_campaign_state(
        campaign_directory,
        rows,
        metadata,
        campaign_status,
    )

    counts = campaign_counts(rows)

    print("\n" + "=" * 75)
    print("CAMPAIGN EXECUTION SUMMARY")
    print("=" * 75)

    print("Campaign status:", campaign_status)
    print("Successful runs:", counts["passed"])
    print("Failed runs:", counts["failed"])
    print("Pending runs:", counts["pending"])

    print(
        "Campaign manifest:",
        campaign_directory / "campaign_manifest.csv",
    )

    print(
        "Campaign metadata:",
        campaign_directory / "campaign_metadata.json",
    )

    if campaign_status == "FAILED":
        print("\nCAMPAIGN FAILED. Inspect the failed run before retrying.")
        return 1

    if campaign_status == "INTERRUPTED":
        print("\nCAMPAIGN INTERRUPTED.")
        return 130

    if campaign_status == "COMPLETED":

        print("\nALL 29 BENCHMARK CONFIGURATIONS COMPLETED.")

    else:

        print(
            "\nLIMITED CAMPAIGN COMPLETED. "
            "This was a partial execution."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
