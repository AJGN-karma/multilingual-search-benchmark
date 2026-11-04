
import unittest

from src.linear_search import linear_search
from src.binary_search import build_sorted_titles, binary_search
from src.hash_search import build_hash_index, hash_search


class TestSearchAlgorithms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Prepare one common dataset and the required indexes.
        All algorithms use exactly the same source titles.
        """

        cls.titles = [
            "Artificial Intelligence",
            "Data Science",
            "Machine Learning",
            "Neural Network",
            "Python Programming",
            "ಕನ್ನಡ",
            "Café",
            "New_York",
            "Space Title",
            "A-B"
        ]

        cls.sorted_titles = build_sorted_titles(cls.titles)
        cls.hash_index = build_hash_index(cls.titles)

        cls.algorithms = [
            (
                "Linear Search",
                lambda query: linear_search(cls.titles, query)
            ),
            (
                "Binary Search",
                lambda query: binary_search(
                    cls.sorted_titles, query
                )
            ),
            (
                "Hash-Table Search",
                lambda query: hash_search(cls.hash_index, query)
            )
        ]

    def test_all_algorithms_find_existing_titles(self):
        """All algorithms must find every existing title."""

        for title in self.titles:
            for name, search_function in self.algorithms:
                with self.subTest(algorithm=name, query=title):
                    self.assertTrue(search_function(title))

    def test_all_algorithms_reject_missing_titles(self):
        """All algorithms must reject titles that do not exist."""

        missing_queries = [
            "Deep Learning",
            "machine learning",
            "Space  Title",
            "New York",
            "A B",
            "Cafe"
        ]

        for query in missing_queries:
            for name, search_function in self.algorithms:
                with self.subTest(algorithm=name, query=query):
                    self.assertFalse(search_function(query))

    def test_empty_corpus(self):
        """All algorithms must handle an empty corpus."""

        query = "Artificial Intelligence"

        self.assertFalse(linear_search([], query))
        self.assertFalse(binary_search([], query))
        self.assertFalse(hash_search(set(), query))

    def test_single_element_corpus(self):
        """All algorithms must handle a one-title corpus."""

        titles = ["Machine Learning"]

        sorted_titles = build_sorted_titles(titles)
        hash_index = build_hash_index(titles)

        self.assertTrue(linear_search(titles, "Machine Learning"))
        self.assertTrue(
            binary_search(sorted_titles, "Machine Learning")
        )
        self.assertTrue(
            hash_search(hash_index, "Machine Learning")
        )

        self.assertFalse(linear_search(titles, "Deep Learning"))
        self.assertFalse(
            binary_search(sorted_titles, "Deep Learning")
        )
        self.assertFalse(
            hash_search(hash_index, "Deep Learning")
        )

    def test_preprocessing_does_not_modify_original_data(self):
        """Sorting and indexing must preserve the source sequence."""

        original = self.titles.copy()

        build_sorted_titles(self.titles)
        build_hash_index(self.titles)

        self.assertEqual(self.titles, original)

    def test_hash_index_contains_all_unique_titles(self):
        """Hash index must contain every unique source title."""

        self.assertEqual(len(self.hash_index), len(self.titles))

        for title in self.titles:
            self.assertIn(title, self.hash_index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
