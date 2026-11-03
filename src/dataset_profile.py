
"""
Dataset profiling for the Wikipedia multilingual
exact-search benchmarking project.

Reads the CSV files one record at a time.

Produces:
    results/tables/corpus_profile.csv
    results/tables/title_length_distribution.csv
    results/tables/unicode_characteristics.csv

No corpus-wide Python list is created.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import unicodedata

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PROFILE_FIELDS = [
    "corpus_id",
    "language",
    "source",
    "source_snapshot",
    "record_count",
    "mean_source_codepoint_length",
    "median_source_codepoint_length",
    "population_std_source_codepoint_length",
    "minimum_source_codepoint_length",
    "maximum_source_codepoint_length",
    "mean_utf8_bytes",
    "minimum_utf8_bytes",
    "maximum_utf8_bytes",
    "ascii_only_title_count",
    "non_ascii_title_count",
    "titles_with_underscores",
    "underscore_occurrences",
    "distinct_unicode_codepoints",
    "length_reference_column",
    "search_column",
    "normalization_policy"
]


def median_from_histogram(
    histogram: Counter,
    total: int
) -> float:
    """Calculate an exact median from a frequency histogram."""

    if total == 0:
        raise ValueError("Cannot calculate median of empty corpus.")

    positions = (
        (total + 1) // 2,
        (total + 2) // 2
    )

    cumulative = 0
    values = []

    for value in sorted(histogram):
        cumulative += histogram[value]

        for position in positions:
            if cumulative >= position and len(values) < len(set(positions)):
                values.append(value)

    if len(positions) == 1:
        return float(values[0])

    if positions[0] == positions[1]:
        return float(values[0])

    return (values[0] + values[1]) / 2.0


def write_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict]
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def profile_dataset(
    dataset_key: str,
    dataset: dict,
    required_columns: list[str],
    source_type: str,
    source_snapshot: str
):

    input_path = ROOT / dataset["relative_path"]

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    record_count = 0

    # Title length statistics
    length_histogram = Counter()
    length_mean = 0.0
    length_m2 = 0.0

    min_length = None
    max_length = None

    # UTF-8 statistics
    utf8_total = 0
    min_utf8 = None
    max_utf8 = None

    # Title-level characteristics
    ascii_only_count = 0
    non_ascii_count = 0
    titles_with_underscores = 0
    underscore_occurrences = 0

    # Unique code points, not character frequencies
    distinct_codepoints = set()

    started = time.monotonic()

    print(f"\nProfiling {dataset['language']}...")
    print(f"File: {input_path}")

    with input_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != required_columns:
            raise ValueError(
                f"Unexpected CSV schema for {dataset_key}."
            )

        for row in reader:

            if None in row:
                raise ValueError(
                    f"Unexpected extra CSV fields near line "
                    f"{reader.line_num}."
                )

            title_source = row["title_source"]
            title_nfc = row["title_nfc"]

            length = int(row["codepoint_length"])
            utf8_bytes = int(row["utf8_bytes"])

            record_count += 1

            # Exact frequency distribution of source lengths
            length_histogram[length] += 1

            # Welford's online mean and variance
            delta = length - length_mean

            length_mean += delta / record_count

            length_m2 += delta * (
                length - length_mean
            )

            if min_length is None or length < min_length:
                min_length = length

            if max_length is None or length > max_length:
                max_length = length

            # UTF-8 byte measurements
            utf8_total += utf8_bytes

            if min_utf8 is None or utf8_bytes < min_utf8:
                min_utf8 = utf8_bytes

            if max_utf8 is None or utf8_bytes > max_utf8:
                max_utf8 = utf8_bytes

            # ASCII and non-ASCII title counts
            if title_nfc.isascii():
                ascii_only_count += 1
            else:
                non_ascii_count += 1

            # Underscore characteristics
            underscore_count = title_source.count("_")

            underscore_occurrences += underscore_count

            if underscore_count > 0:
                titles_with_underscores += 1

            # Collect distinct Unicode characters.
            # This set is bounded by the number of distinct
            # Unicode code points, not by the number of titles.
            distinct_codepoints.update(title_nfc)

            if record_count % 1_000_000 == 0:
                elapsed = time.monotonic() - started

                print(
                    f"Processed {record_count:,} records "
                    f"in {elapsed:.1f} seconds"
                )

    if record_count == 0:
        raise ValueError(
            f"No data records found in {dataset_key}."
        )

    median_length = median_from_histogram(
        length_histogram,
        record_count
    )

    population_std = math.sqrt(
        length_m2 / record_count
    )

    mean_utf8 = utf8_total / record_count

    # Record source-level summary.
    profile = {
        "corpus_id": dataset_key,
        "language": dataset["language"],
        "source": source_type,
        "source_snapshot": source_snapshot,
        "record_count": record_count,
        "mean_source_codepoint_length": round(length_mean, 4),
        "median_source_codepoint_length": median_length,
        "population_std_source_codepoint_length": round(
            population_std, 4
        ),
        "minimum_source_codepoint_length": min_length,
        "maximum_source_codepoint_length": max_length,
        "mean_utf8_bytes": round(mean_utf8, 4),
        "minimum_utf8_bytes": min_utf8,
        "maximum_utf8_bytes": max_utf8,
        "ascii_only_title_count": ascii_only_count,
        "non_ascii_title_count": non_ascii_count,
        "titles_with_underscores": titles_with_underscores,
        "underscore_occurrences": underscore_occurrences,
        "distinct_unicode_codepoints": len(distinct_codepoints),
        "length_reference_column": "codepoint_length",
        "search_column": "title_nfc",
        "normalization_policy": "Unicode NFC"
    }

    # Title length distribution
    length_rows = []

    for length, count in sorted(length_histogram.items()):

        length_rows.append({
            "corpus_id": dataset_key,
            "language": dataset["language"],
            "source_codepoint_length": length,
            "title_count": count,
            "percentage": round(
                count * 100 / record_count,
                6
            )
        })

    # Unicode characteristics.
    # Category and name-prefix counts below describe
    # distinct code points, NOT character occurrences.
    category_counts = Counter()
    name_prefix_counts = Counter()

    for character in distinct_codepoints:

        category = unicodedata.category(character)
        category_counts[category] += 1

        unicode_name = unicodedata.name(
            character,
            "UNNAMED"
        )

        prefix = unicode_name.split()[0]
        name_prefix_counts[prefix] += 1

    unicode_rows = []

    def add_unicode_metric(
        feature: str,
        value: int | str,
        unit: str,
        note: str = ""
    ):

        unicode_rows.append({
            "corpus_id": dataset_key,
            "language": dataset["language"],
            "feature": feature,
            "value": value,
            "unit": unit,
            "note": note
        })

    add_unicode_metric(
        "distinct_unicode_codepoints",
        len(distinct_codepoints),
        "distinct code points"
    )

    add_unicode_metric(
        "ascii_only_title_count",
        ascii_only_count,
        "titles"
    )

    add_unicode_metric(
        "non_ascii_title_count",
        non_ascii_count,
        "titles"
    )

    add_unicode_metric(
        "titles_with_underscores",
        titles_with_underscores,
        "titles"
    )

    add_unicode_metric(
        "underscore_occurrences",
        underscore_occurrences,
        "characters"
    )

    for category, count in sorted(category_counts.items()):

        add_unicode_metric(
            f"unique_codepoints_category_{category}",
            count,
            "distinct code points",
            "Unicode general category"
        )

    for prefix, count in name_prefix_counts.most_common(20):

        add_unicode_metric(
            f"unique_codepoints_name_prefix_{prefix}",
            count,
            "distinct code points",
            "Prefix of Unicode character name; not a language classifier"
        )

    print(
        f"Completed {dataset['language']}: "
        f"{record_count:,} records"
    )

    print(
        f"Mean title length: {profile['mean_source_codepoint_length']}"
    )

    print(
        f"Median title length: {profile['median_source_codepoint_length']}"
    )

    print(
        f"Distinct Unicode code points: "
        f"{profile['distinct_unicode_codepoints']:,}"
    )

    return profile, length_rows, unicode_rows


def main():

    parser = argparse.ArgumentParser(
        description="Profile multilingual Wikipedia title datasets."
    )

    parser.add_argument(
        "--config",
        default="config/experiment.json"
    )

    parser.add_argument(
        "--dataset",
        choices=["tulu", "kannada", "english"],
        default=None
    )

    args = parser.parse_args()

    config_path = ROOT / args.config

    with config_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    required_columns = config["dataset_schema"][
        "required_columns"
    ]

    with (ROOT / config["paths"]["manifest"]).open(
        "r",
        encoding="utf-8"
    ) as file:
        manifest = json.load(file)

    profile_rows = []
    distribution_rows = []
    unicode_rows = []

    datasets = config["datasets"]

    if args.dataset:
        datasets = {
            args.dataset: datasets[args.dataset]
        }

    for key, dataset in datasets.items():

        profile, lengths, unicode_data = profile_dataset(
            key,
            dataset,
            required_columns,
            manifest.get("source_type", "Unknown"),
            manifest.get("source_snapshot", "Unknown")
        )

        profile_rows.append(profile)
        distribution_rows.extend(lengths)
        unicode_rows.extend(unicode_data)

    output_directory = ROOT / "results/tables"

    write_csv(
        output_directory / "corpus_profile.csv",
        PROFILE_FIELDS,
        profile_rows
    )

    write_csv(
        output_directory / "title_length_distribution.csv",
        [
            "corpus_id",
            "language",
            "source_codepoint_length",
            "title_count",
            "percentage"
        ],
        distribution_rows
    )

    write_csv(
        output_directory / "unicode_characteristics.csv",
        [
            "corpus_id",
            "language",
            "feature",
            "value",
            "unit",
            "note"
        ],
        unicode_rows
    )

    print("\n" + "=" * 60)
    print("DATASET PROFILING COMPLETED")
    print("=" * 60)

    print("Reports:")
    print(output_directory / "corpus_profile.csv")
    print(output_directory / "title_length_distribution.csv")
    print(output_directory / "unicode_characteristics.csv")


if __name__ == "__main__":
    main()
