
import argparse
import csv
import json
import re
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path

from src.benchmark_workloads import read_manifest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.json"

SAMPLE_MANIFEST_PATH = (
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

CAMPAIGN_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "memory_campaigns"
)

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
    "status",
    "return_code",
    "started_at_utc",
    "completed_at_utc",
    "elapsed_seconds",
    "raw_trials_path",
    "summary_path",
    "metadata_path",
    "error",
]


def utc_now():
    """Return the current UTC time."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def load_json(path):
    """Read a JSON file."""

    with path.open(encoding="utf-8") as file:
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
            "or hyphens. Maximum length: 40 characters."
        )


def build_run_tag(campaign_tag, language, sample_size):
    """Create a unique memory run identifier."""

    tag = (
        f"{campaign_tag}_{language.lower()}"
        f"_n{sample_size}"
    )

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}",
        tag,
    ):
        raise ValueError(
            f"Invalid generated memory run tag: {tag}"
        )

    return tag


def build_plan(config, sample_rows):
    """Build and validate all 16 memory configurations."""

    plan = []

    for dataset in config["datasets"].values():

        language = dataset["language"]

        for sample_size in dataset["candidate_corpus_sizes"]:

            sample_size = int(sample_size)

            matches = [
                row for row in sample_rows
                if row["language"] == language
                and int(row["sample_size"]) == sample_size
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one validated sample for "
                    f"{language}, n={sample_size}; "
                    f"found {len(matches)}."
                )

            sample = matches[0]

            if sample["status"] != "PASS":
                raise RuntimeError(
                    f"Sample validation failed: "
                    f"{language}, n={sample_size}"
                )

            sample_file = PROJECT_ROOT / sample["sample_file"]

            if not sample_file.is_file():
                raise FileNotFoundError(
                    f"Sample file missing: {sample_file}"
                )

            plan.append({
                "language": language,
                "sample_size": sample_size,
                "sample_file": sample["sample_file"],
                "sample_sha256": sample["sample_sha256"],
                "corpus_id": sample["corpus_id"],
            })

    if len(plan) != 16:
        raise RuntimeError(
            f"Expected 16 corpus samples, found {len(plan)}."
        )

    return plan


def get_output_paths(run_tag):
    """Return the output files for a memory run."""

    return {
        "raw": (
            RAW_OUTPUT_ROOT
            / run_tag
            / "memory_trials.csv"
        ),
        "summary": (
            TABLE_OUTPUT_ROOT
            / f"{run_tag}_summary.csv"
        ),
        "metadata": (
            TABLE_OUTPUT_ROOT
            / f"{run_tag}_metadata.json"
        ),
    }


def check_output_collisions(plan, campaign_tag):
    """Reject existing campaign and memory output paths."""

    campaign_directory = (
        CAMPAIGN_OUTPUT_ROOT / campaign_tag
    )

    if campaign_directory.exists():
        raise FileExistsError(
            f"Memory campaign already exists: "
            f"{campaign_directory}"
        )

    for item in plan:

        run_tag = build_run_tag(
            campaign_tag,
            item["language"],
            item["sample_size"],
        )

        for path in get_output_paths(run_tag).values():

            if path.exists():
                raise FileExistsError(
                    f"Memory output already exists: {path}. "
                    "Use another campaign tag."
                )


def write_campaign_state(
    campaign_directory,
    rows,
    metadata,
    status,
):
    """Save campaign progress safely."""

    manifest_path = (
        campaign_directory / "memory_campaign_manifest.csv"
    )

    temporary_manifest = (
        campaign_directory / "memory_campaign_manifest.csv.tmp"
    )

    with temporary_manifest.open(
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

    temporary_manifest.replace(manifest_path)

    metadata["status"] = status
    metadata["updated_at_utc"] = utc_now()

    metadata["counts"] = {
        "passed": sum(r["status"] == "PASS" for r in rows),
        "failed": sum(r["status"] == "FAIL" for r in rows),
        "interrupted": sum(
            r["status"] == "INTERRUPTED"
            for r in rows
        ),
        "pending": sum(r["status"] == "PENDING" for r in rows),
    }

    metadata_path = (
        campaign_directory / "memory_campaign_metadata.json"
    )

    temporary_metadata = (
        campaign_directory / "memory_campaign_metadata.json.tmp"
    )

    with temporary_metadata.open(
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

    temporary_metadata.replace(metadata_path)


def run_memory_configuration(item, run_tag, trials):
    """Execute one sample's complete memory experiment."""

    command = [
        sys.executable,
        "-m",
        "src.memory_measurement",
        "--language",
        item["language"],
        "--sample-size",
        str(item["sample_size"]),
        "--trials",
        str(trials),
        "--run-tag",
        run_tag,
    ]

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )


def verify_memory_output(run_tag, expected_trials):
    """Independently verify the three output reports."""

    paths = get_output_paths(run_tag)

    for path in paths.values():

        if not path.is_file():
            raise RuntimeError(
                f"Expected memory report is missing: {path}"
            )

    with paths["raw"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        raw_rows = list(csv.DictReader(file))

    with paths["summary"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        summary_rows = list(csv.DictReader(file))

    metadata = load_json(paths["metadata"])

    expected_count = len(ALGORITHMS) * expected_trials

    if len(raw_rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} memory records, "
            f"found {len(raw_rows)}."
        )

    if len(summary_rows) != len(ALGORITHMS):
        raise RuntimeError(
            "Memory summary must contain exactly three algorithms."
        )

    if not all(row["status"] == "PASS" for row in raw_rows):
        raise RuntimeError(
            "At least one raw memory measurement did not pass."
        )

    if not all(
        row["index_verified"] == "True"
        for row in raw_rows
    ):
        raise RuntimeError(
            "At least one constructed index failed verification."
        )

    if not all(row["status"] == "PASS" for row in summary_rows):
        raise RuntimeError(
            "At least one memory summary failed."
        )

    if metadata["status"] != "PASS":
        raise RuntimeError(
            "Memory metadata does not report PASS."
        )

    if metadata["total_measurements"] != expected_count:
        raise RuntimeError(
            "Memory metadata measurement count is incorrect."
        )

    counts = {
        row["algorithm"]: sum(
            r["algorithm"] == row["algorithm"]
            for r in raw_rows
        )
        for row in summary_rows
    }

    if any(value != expected_trials for value in counts.values()):
        raise RuntimeError(
            "Memory trial count differs across algorithms."
        )

    return len(raw_rows), len(summary_rows)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run isolated memory measurements across "
            "all validated multilingual corpus samples."
        )
    )

    parser.add_argument(
        "--campaign-tag",
        required=True,
        help="Unique memory campaign identifier.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N samples for testing.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the plan without executing measurements.",
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

        plan = build_plan(
            config,
            sample_rows,
        )

        trials = int(
            config["benchmark"]["trials"]
        )

        if trials < 2:
            raise RuntimeError(
                "At least two memory trials are required."
            )

        selected_plan = (
            plan[:args.limit]
            if args.limit is not None
            else plan
        )

        check_output_collisions(
            selected_plan,
            args.campaign_tag,
        )

    except Exception as error:
        parser.error(str(error))

    total_measurements = (
        len(selected_plan)
        * len(ALGORITHMS)
        * trials
    )

    print("=" * 75)
    print("MULTILINGUAL SEARCH MEMORY CAMPAIGN")
    print("=" * 75)

    print("Campaign tag:", args.campaign_tag)
    print("Total validated samples:", len(plan))
    print("Selected samples:", len(selected_plan))
    print("Algorithms:", len(ALGORITHMS))
    print("Trials per algorithm:", trials)
    print("Expected measurements:", total_measurements)

    print("\nValidated execution schedule:")

    for index, item in enumerate(selected_plan, start=1):

        print(
            f"{index:02d}. "
            f"{item['language']:8} | "
            f"Corpus size: {item['sample_size']:>9,}"
        )

    if args.dry_run:

        print("\nMEMORY CAMPAIGN DRY RUN PASSED.")
        print("No memory subprocesses were started.")
        print("No campaign output files were created.")

        return 0

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
        )

        paths = get_output_paths(run_tag)

        rows.append({
            "sequence": sequence,
            "run_tag": run_tag,
            "language": item["language"],
            "sample_size": item["sample_size"],
            "status": "PENDING",
            "return_code": "",
            "started_at_utc": "",
            "completed_at_utc": "",
            "elapsed_seconds": "",
            "raw_trials_path": str(
                paths["raw"].relative_to(PROJECT_ROOT)
            ),
            "summary_path": str(
                paths["summary"].relative_to(PROJECT_ROOT)
            ),
            "metadata_path": str(
                paths["metadata"].relative_to(PROJECT_ROOT)
            ),
            "error": "",
        })

    metadata = {
        "experiment": (
            "Comparative Memory Evaluation of Exact-Search "
            "Data Structures for Multilingual Title Retrieval"
        ),
        "campaign_tag": args.campaign_tag,
        "created_at_utc": utc_now(),
        "planned_samples": len(plan),
        "selected_samples": len(selected_plan),
        "algorithms": ALGORITHMS,
        "trials_per_algorithm": trials,
        "expected_measurements": total_measurements,
        "measurement_tools": [
            "tracemalloc",
            "psutil",
        ],
        "process_isolation": (
            "Fresh subprocess for every algorithm trial"
        ),
        "schedule": selected_plan,
    }

    write_campaign_state(
        campaign_directory,
        rows,
        metadata,
        "RUNNING",
    )

    campaign_status = "RUNNING"

    print("\n" + "=" * 75)
    print("MEMORY CAMPAIGN EXECUTION STARTED")
    print("=" * 75)

    for index, row in enumerate(rows):

        item = selected_plan[index]

        print("\n" + "#" * 75)
        print(
            f"MEMORY RUN {index + 1}/{len(rows)}: "
            f"{item['language']} n={item['sample_size']}"
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

        import time
        start_time = time.perf_counter()

        try:

            completed = run_memory_configuration(
                item,
                row["run_tag"],
                trials,
            )

            row["return_code"] = completed.returncode

            row["elapsed_seconds"] = round(
                time.perf_counter() - start_time,
                3,
            )

            row["completed_at_utc"] = utc_now()

            if completed.returncode != 0:
                raise RuntimeError(
                    f"Memory measurement exited with "
                    f"code {completed.returncode}."
                )

            verify_memory_output(
                row["run_tag"],
                trials,
            )

            row["status"] = "PASS"

            print(
                f"\nMemory configuration {index + 1} "
                "completed and verified."
            )

        except KeyboardInterrupt:

            row["status"] = "INTERRUPTED"
            row["completed_at_utc"] = utc_now()
            row["error"] = "Campaign interrupted by user."
            campaign_status = "INTERRUPTED"

        except Exception as error:

            row["status"] = "FAIL"
            row["completed_at_utc"] = utc_now()
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
                "Remaining samples were not executed."
            )

            break

        write_campaign_state(
            campaign_directory,
            rows,
            metadata,
            "RUNNING",
        )

    if campaign_status == "RUNNING":

        campaign_status = (
            "COMPLETED"
            if len(selected_plan) == len(plan)
            else "PARTIAL"
        )

    write_campaign_state(
        campaign_directory,
        rows,
        metadata,
        campaign_status,
    )

    passed = sum(
        row["status"] == "PASS"
        for row in rows
    )

    failed = sum(
        row["status"] == "FAIL"
        for row in rows
    )

    pending = sum(
        row["status"] == "PENDING"
        for row in rows
    )

    print("\n" + "=" * 75)
    print("MEMORY CAMPAIGN SUMMARY")
    print("=" * 75)

    print("Campaign status:", campaign_status)
    print("Successful samples:", passed)
    print("Failed samples:", failed)
    print("Pending samples:", pending)

    print(
        "Campaign manifest:",
        campaign_directory / "memory_campaign_manifest.csv",
    )

    print(
        "Campaign metadata:",
        campaign_directory / "memory_campaign_metadata.json",
    )

    if campaign_status == "COMPLETED":
        print("\nALL MEMORY CONFIGURATIONS COMPLETED.")
        return 0

    if campaign_status == "INTERRUPTED":
        return 130

    if campaign_status == "FAILED":
        return 1

    print("\nLIMITED MEMORY CAMPAIGN COMPLETED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
