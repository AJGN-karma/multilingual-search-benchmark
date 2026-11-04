
import csv
import json
from collections import Counter
from pathlib import Path

from src.dataset_loader import calculate_sha256


ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT / "config" / "experiment.json"

SAMPLE_MANIFEST = (
    ROOT / "results" / "tables" / "corpus_sample_manifest.csv"
)

WORKLOAD_MANIFEST = (
    ROOT / "results" / "tables" / "query_workload_manifest.csv"
)

OUTPUT_CSV = (
    ROOT / "results" / "tables" / "final_workload_audit.csv"
)

OUTPUT_JSON = (
    ROOT / "results" / "tables" / "final_workload_audit.json"
)

EXPECTED_COLUMNS = [
    "query_id",
    "query",
    "expected_found",
]

WORKLOAD_TYPES = [
    "successful",
    "unsuccessful",
    "mixed",
]


def read_csv(path):
    """Read CSV manifest records."""

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def validate_workload_file(row):
    """Verify one workload file's structure, count and checksum."""

    workload_id = row["workload_id"]
    query_count = int(row["query_count"])
    workload_type = row["workload_type"]

    relative_path = Path(row["workload_file"])

    if relative_path.is_absolute():
        raise ValueError(
            f"{workload_id}: Absolute workload path is not allowed."
        )

    path = (ROOT / relative_path).resolve()

    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError(
            f"{workload_id}: Workload path escapes project directory."
        )

    if not path.is_file():
        raise ValueError(
            f"{workload_id}: Workload file does not exist."
        )

    actual_hash = calculate_sha256(path)

    if actual_hash != row["workload_sha256"]:
        raise ValueError(
            f"{workload_id}: SHA-256 mismatch."
        )

    result_counts = Counter()
    records_checked = 0

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"{workload_id}: Incorrect CSV columns."
            )

        for expected_id, record in enumerate(reader, start=1):

            records_checked += 1

            if int(record["query_id"]) != expected_id:
                raise ValueError(
                    f"{workload_id}: Invalid query ID."
                )

            if not record["query"]:
                raise ValueError(
                    f"{workload_id}: Empty query found."
                )

            result = record["expected_found"]

            if result not in ("true", "false"):
                raise ValueError(
                    f"{workload_id}: Invalid expected result."
                )

            result_counts[result] += 1

    if records_checked != query_count:
        raise ValueError(
            f"{workload_id}: Expected {query_count} records, "
            f"found {records_checked}."
        )

    if workload_type == "successful":
        if result_counts["true"] != query_count:
            raise ValueError(
                f"{workload_id}: Successful workload label mismatch."
            )

    elif workload_type == "unsuccessful":
        if result_counts["false"] != query_count:
            raise ValueError(
                f"{workload_id}: Unsuccessful workload label mismatch."
            )

    elif workload_type == "mixed":
        if query_count % 2 != 0:
            raise ValueError(
                f"{workload_id}: Mixed query count must be even."
            )

        if result_counts["true"] != query_count // 2:
            raise ValueError(
                f"{workload_id}: Incorrect positive-query count."
            )

        if result_counts["false"] != query_count // 2:
            raise ValueError(
                f"{workload_id}: Incorrect negative-query count."
            )

    unique_count = int(row["unique_query_count"])
    repeated_count = int(row["repeated_query_occurrences"])

    if unique_count < 1 or unique_count > query_count:
        raise ValueError(
            f"{workload_id}: Invalid unique-query count."
        )

    if query_count - unique_count != repeated_count:
        raise ValueError(
            f"{workload_id}: Unique/repeated count mismatch."
        )

    if workload_type == "unsuccessful":
        if unique_count != query_count or repeated_count != 0:
            raise ValueError(
                f"{workload_id}: Negative queries must be unique."
            )

    return {
        "workload_id": workload_id,
        "corpus_id": row["corpus_id"],
        "language": row["language"],
        "sample_size": int(row["sample_size"]),
        "workload_type": workload_type,
        "query_count": query_count,
        "true_count": result_counts["true"],
        "false_count": result_counts["false"],
        "unique_query_count": unique_count,
        "repeated_query_occurrences": repeated_count,
        "workload_file": row["workload_file"],
        "file_bytes": path.stat().st_size,
        "workload_sha256": actual_hash,
        "status": "PASS",
    }


def main():
    """Perform the consolidated audit of all query workloads."""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    query_counts = [
        int(value)
        for value in config["query_workloads"]["query_counts"]
    ]

    samples = read_csv(SAMPLE_MANIFEST)
    workloads = read_csv(WORKLOAD_MANIFEST)

    print("=" * 65)
    print("FINAL QUERY WORKLOAD AUDIT")
    print("=" * 65)

    print("Corpus samples:", len(samples))
    print("Workload entries:", len(workloads))
    print("Configured query counts:", query_counts)

    # Verify unique workload identifiers.
    workload_ids = [
        row["workload_id"] for row in workloads
    ]

    if len(workload_ids) != len(set(workload_ids)):
        raise ValueError("Duplicate workload IDs detected.")

    # Build the expected sample/type/count combinations.
    expected_combinations = set()

    for sample in samples:
        for workload_type in WORKLOAD_TYPES:
            for query_count in query_counts:
                expected_combinations.add((
                    sample["corpus_id"],
                    int(sample["sample_size"]),
                    workload_type,
                    query_count,
                ))

    actual_combinations = set()

    sample_lookup = {
        (
            row["corpus_id"],
            int(row["sample_size"])
        ): row
        for row in samples
    }

    for row in workloads:

        combination = (
            row["corpus_id"],
            int(row["sample_size"]),
            row["workload_type"],
            int(row["query_count"]),
        )

        if combination in actual_combinations:
            raise ValueError(
                f"Duplicate workload combination: {combination}"
            )

        actual_combinations.add(combination)

        sample_key = (
            row["corpus_id"],
            int(row["sample_size"])
        )

        if sample_key not in sample_lookup:
            raise ValueError(
                f"Unknown source corpus: {sample_key}"
            )

        source_sample = sample_lookup[sample_key]

        if row["status"] != "PASS":
            raise ValueError(
                f"Workload has failed status: {row['workload_id']}"
            )

        if row["source_sample_file"] != source_sample["sample_file"]:
            raise ValueError(
                f"Source sample filename mismatch: {row['workload_id']}"
            )

        if row["source_sample_sha256"] != source_sample["sample_sha256"]:
            raise ValueError(
                f"Source sample hash mismatch: {row['workload_id']}"
            )

    if actual_combinations != expected_combinations:
        missing = expected_combinations - actual_combinations
        unexpected = actual_combinations - expected_combinations

        raise ValueError(
            f"Workload combination mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    print("Workload combination coverage: PASS")
    print("Unique workload IDs: PASS")
    print("Source sample references: PASS")

    # Audit every individual workload file.
    audit_rows = []

    totals_by_type = Counter()
    files_by_type = Counter()
    records_by_type = Counter()
    total_bytes = 0

    print("\nAuditing individual workload files...")

    for row in workloads:

        result = validate_workload_file(row)

        audit_rows.append(result)

        workload_type = result["workload_type"]

        files_by_type[workload_type] += 1
        records_by_type[workload_type] += result["query_count"]

        total_bytes += result["file_bytes"]

        totals_by_type[workload_type] += (
            result["true_count"] + result["false_count"]
        )

    # Verify expected totals.
    expected_per_type = len(samples) * sum(query_counts)

    for workload_type in WORKLOAD_TYPES:

        if files_by_type[workload_type] != len(samples) * len(query_counts):
            raise ValueError(
                f"Incorrect file count for {workload_type}."
            )

        if records_by_type[workload_type] != expected_per_type:
            raise ValueError(
                f"Incorrect record count for {workload_type}."
            )

        print(
            f"{workload_type.upper()}: "
            f"{files_by_type[workload_type]} files | "
            f"{records_by_type[workload_type]:,} records | PASS"
        )

    expected_total_records = expected_per_type * len(WORKLOAD_TYPES)

    actual_total_records = sum(records_by_type.values())

    if actual_total_records != expected_total_records:
        raise ValueError("Total query record count mismatch.")

    # Save detailed audit report.
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(audit_rows[0].keys()),
            lineterminator="\n"
        )

        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "audit_name": "Final Query Workload Audit",
        "corpus_sample_count": len(samples),
        "query_counts": query_counts,
        "workload_file_count": len(workloads),
        "workloads_by_type": dict(files_by_type),
        "records_by_type": dict(records_by_type),
        "total_query_records": actual_total_records,
        "total_workload_file_bytes": total_bytes,
        "coverage_validation": "PASS",
        "file_existence_validation": "PASS",
        "sha256_validation": "PASS",
        "csv_structure_validation": "PASS",
        "record_count_validation": "PASS",
        "result_distribution_validation": "PASS",
        "unique_repeated_count_validation": "PASS",
        "overall_status": "PASS",
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

        file.write("\n")

    print("\n" + "=" * 65)
    print("FINAL AUDIT SUMMARY")
    print("=" * 65)

    print("Total workload files:", len(workloads))
    print("Total query records:", f"{actual_total_records:,}")

    print(
        "Total workload file size:",
        f"{total_bytes / (1024 * 1024):.2f} MiB"
    )

    print("Combination coverage: PASS")
    print("File existence: PASS")
    print("SHA-256 checksums: PASS")
    print("CSV structure: PASS")
    print("Record counts: PASS")
    print("Expected-result distributions: PASS")
    print("Unique/repeated metadata: PASS")

    print(
        "\nCSV audit report:",
        OUTPUT_CSV.relative_to(ROOT)
    )

    print(
        "JSON audit report:",
        OUTPUT_JSON.relative_to(ROOT)
    )

    print("\nFINAL QUERY WORKLOAD AUDIT: PASS")


if __name__ == "__main__":
    main()
