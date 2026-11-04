
import unittest
from collections import Counter

from src.repetition_patterns import (
    derive_repetition_seed,
    generate_repetition_pattern,
)


class TestRepetitionPatterns(unittest.TestCase):

    def setUp(self):

        self.records = [
            ("Title_A", "true"),
            ("Title_B", "true"),
            ("Title_C", "true"),
            ("Missing_A", "false"),
            ("Missing_B", "false"),
            ("Missing_C", "false"),
        ]

        self.positive_pool = [
            f"Title_{index}"
            for index in range(100)
        ]

        self.negative_pool = [
            f"Missing_{index}"
            for index in range(100)
        ]

    def test_same_query_set_preserves_exact_order(self):

        result = generate_repetition_pattern(
            self.records,
            "same_query_set",
            seed=2026,
        )

        self.assertEqual(result, self.records)
        self.assertIsNot(result, self.records)

    def test_mostly_unique_avoids_repetition_when_possible(self):

        result = generate_repetition_pattern(
            self.records,
            "mostly_unique",
            seed=2026,
            positive_candidates=self.positive_pool,
            negative_candidates=self.negative_pool,
        )

        queries = [query for query, _ in result]

        self.assertEqual(len(result), len(self.records))
        self.assertEqual(len(set(queries)), len(queries))

    def test_mostly_unique_repeats_only_after_pool_exhaustion(self):

        positive_pool = ["A", "B"]
        negative_pool = ["X"]

        records = [
            ("A", "true"),
            ("B", "true"),
            ("X", "false"),
            ("X", "false"),
            ("X", "false"),
        ]

        result = generate_repetition_pattern(
            records,
            "mostly_unique",
            seed=2026,
            positive_candidates=positive_pool,
            negative_candidates=negative_pool,
        )

        true_queries = [
            query for query, label in result if label == "true"
        ]

        false_queries = [
            query for query, label in result if label == "false"
        ]

        self.assertEqual(set(true_queries), {"A", "B"})
        self.assertEqual(set(false_queries), {"X"})
        self.assertEqual(len(result), 5)

    def test_mixed_pattern_preserves_outcome_counts(self):

        result = generate_repetition_pattern(
            self.records,
            "mixed",
            seed=2026,
            positive_candidates=self.positive_pool,
            negative_candidates=self.negative_pool,
        )

        original_counts = Counter(
            label for _, label in self.records
        )

        result_counts = Counter(
            label for _, label in result
        )

        self.assertEqual(result_counts, original_counts)
        self.assertEqual(len(result), len(self.records))

    def test_mixed_pattern_is_reproducible(self):

        first = generate_repetition_pattern(
            self.records,
            "mixed",
            seed=12345,
            positive_candidates=self.positive_pool,
            negative_candidates=self.negative_pool,
        )

        second = generate_repetition_pattern(
            self.records,
            "mixed",
            seed=12345,
            positive_candidates=self.positive_pool,
            negative_candidates=self.negative_pool,
        )

        self.assertEqual(first, second)

    def test_mixed_pattern_creates_repeated_hot_queries(self):

        positive_pool = [
            f"P{index}"
            for index in range(10)
        ]

        negative_pool = [
            f"N{index}"
            for index in range(10)
        ]

        records = (
            [(f"P{index}", "true") for index in range(20)]
            + [(f"N{index}", "false") for index in range(20)]
        )

        result = generate_repetition_pattern(
            records,
            "mixed",
            seed=2026,
            positive_candidates=positive_pool,
            negative_candidates=negative_pool,
        )

        queries = [query for query, _ in result]

        self.assertLess(len(set(queries)), len(queries))

        self.assertEqual(len(result), 40)

    def test_repetition_seed_is_deterministic(self):

        first = derive_repetition_seed(
            2026, "english", 1000, "mixed", 10000,
            "mostly_unique"
        )

        second = derive_repetition_seed(
            2026, "english", 1000, "mixed", 10000,
            "mostly_unique"
        )

        self.assertEqual(first, second)

    def test_input_records_are_not_modified(self):

        original = self.records.copy()

        generate_repetition_pattern(
            self.records,
            "mostly_unique",
            seed=2026,
            positive_candidates=self.positive_pool,
            negative_candidates=self.negative_pool,
        )

        self.assertEqual(self.records, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
