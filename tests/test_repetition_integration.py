
import csv
import json
import random
import unittest
from pathlib import Path
from collections import Counter

from src.dataset_loader import load_corpus
from src.repetition_patterns import (
    generate_repetition_pattern,
    derive_repetition_seed,
)


ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT / "config" / "experiment.json"

SAMPLE_MANIFEST = (
    ROOT / "results" / "tables" / "corpus_sample_manifest.csv"
)

WORKLOAD_MANIFEST = (
    ROOT / "results" / "tables" / "query_workload_manifest.csv"
)


def read_manifest(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def load_workload_records(row):
    """Load the saved query records from a workload file."""

    path = ROOT / row["workload_file"]

    records = []

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for record in reader:
            records.append((
                record["query"],
                record["expected_found"]
            ))

    return records


class TestRealDatasetRepetitionIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        with CONFIG_PATH.open(encoding="utf-8") as file:
            cls.config = json.load(file)

        cls.search_column = (
            cls.config["dataset_schema"]["search_column"]
        )

        cls.base_seed = int(
            cls.config["query_workloads"]["seed"]
        )

        cls.samples = read_manifest(SAMPLE_MANIFEST)
        cls.workloads = read_manifest(WORKLOAD_MANIFEST)

        cls.test_data = {}

        # Select the smallest 1,000-title sample for each language.
        for language in ["Tulu", "Kannada", "English"]:

            matching_samples = [
                row for row in cls.samples
                if row["language"] == language
                and int(row["sample_size"]) == 1000
            ]

            if len(matching_samples) != 1:
                raise RuntimeError(
                    f"Unable to locate the 1,000-title "
                    f"{language} sample."
                )

            sample = matching_samples[0]

            titles = load_corpus(
                sample_path=ROOT / sample["sample_file"],
                expected_count=1000,
                expected_sha256=sample["sample_sha256"],
                search_column=cls.search_column
            )

            workload_records = {}

            # Use the 100-query workloads for the smoke test.
            for workload_type in [
                "successful",
                "unsuccessful",
                "mixed"
            ]:

                matches = [
                    row for row in cls.workloads
                    if row["corpus_id"] == sample["corpus_id"]
                    and int(row["sample_size"]) == 1000
                    and int(row["query_count"]) == 100
                    and row["workload_type"] == workload_type
                ]

                if len(matches) != 1:
                    raise RuntimeError(
                        f"Missing {workload_type} workload "
                        f"for {language}."
                    )

                workload_records[workload_type] = (
                    load_workload_records(matches[0])
                )

            cls.test_data[language] = {
                "sample": sample,
                "titles": titles,
                "title_set": set(titles),
                "records": workload_records,
                "negative_pool": [
                    query
                    for query, result
                    in workload_records["unsuccessful"]
                    if result == "false"
                ]
            }

    def test_all_patterns_preserve_membership_and_counts(self):

        patterns = [
            "same_query_set",
            "mostly_unique",
            "mixed"
        ]

        for language, data in self.test_data.items():

            titles = data["titles"]
            title_set = data["title_set"]
            negative_pool = data["negative_pool"]

            for workload_type, records in data["records"].items():

                original_counts = Counter(
                    result for _, result in records
                )

                sample = data["sample"]

                for pattern in patterns:

                    seed = derive_repetition_seed(
                        self.base_seed,
                        sample["corpus_id"],
                        1000,
                        workload_type,
                        len(records),
                        pattern
                    )

                    with self.subTest(
                        language=language,
                        workload_type=workload_type,
                        pattern=pattern
                    ):

                        generated = generate_repetition_pattern(
                            records=records,
                            pattern=pattern,
                            seed=seed,
                            positive_candidates=titles,
                            negative_candidates=negative_pool
                        )

                        self.assertEqual(
                            len(generated),
                            len(records)
                        )

                        generated_counts = Counter(
                            result for _, result in generated
                        )

                        self.assertEqual(
                            generated_counts,
                            original_counts
                        )

                        # Every successful query must exist.
                        for query, result in generated:

                            if result == "true":
                                self.assertIn(query, title_set)

                            # Every unsuccessful query must be absent.
                            else:
                                self.assertNotIn(query, title_set)

    def test_same_query_set_preserves_exact_sequence(self):

        for language, data in self.test_data.items():

            for workload_type, records in data["records"].items():

                generated = generate_repetition_pattern(
                    records=records,
                    pattern="same_query_set",
                    seed=2026,
                    positive_candidates=data["titles"],
                    negative_candidates=data["negative_pool"]
                )

                with self.subTest(
                    language=language,
                    workload_type=workload_type
                ):
                    self.assertEqual(generated, records)

    def test_all_patterns_are_reproducible(self):

        patterns = [
            "same_query_set",
            "mostly_unique",
            "mixed"
        ]

        data = self.test_data["English"]

        for pattern in patterns:

            seed = derive_repetition_seed(
                self.base_seed,
                "english",
                1000,
                "mixed",
                100,
                pattern
            )

            first = generate_repetition_pattern(
                records=data["records"]["mixed"],
                pattern=pattern,
                seed=seed,
                positive_candidates=data["titles"],
                negative_candidates=data["negative_pool"]
            )

            second = generate_repetition_pattern(
                records=data["records"]["mixed"],
                pattern=pattern,
                seed=seed,
                positive_candidates=data["titles"],
                negative_candidates=data["negative_pool"]
            )

            with self.subTest(pattern=pattern):
                self.assertEqual(first, second)

    def test_original_records_are_not_modified(self):

        for language, data in self.test_data.items():

            for workload_type, records in data["records"].items():

                original = records.copy()

                generate_repetition_pattern(
                    records=records,
                    pattern="mixed",
                    seed=2026,
                    positive_candidates=data["titles"],
                    negative_candidates=data["negative_pool"]
                )

                with self.subTest(
                    language=language,
                    workload_type=workload_type
                ):
                    self.assertEqual(records, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
