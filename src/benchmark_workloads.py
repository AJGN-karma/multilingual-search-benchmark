
import csv
from collections import Counter
from pathlib import Path

from src.dataset_loader import load_corpus, calculate_sha256
from src.repetition_patterns import (
    generate_repetition_pattern,
    derive_repetition_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_COLUMNS = [
    "query_id",
    "query",
    "expected_found",
]


class BenchmarkWorkloadError(RuntimeError):
    """Raised when benchmark workload preparation fails."""


class CorpusWorkloadProvider:
    """
    Prepare reproducible query sequences for one corpus sample.

    A provider loads one validated corpus and its unique negative
    query pool, then prepares the requested query workload and
    repetition pattern without repeatedly loading the dataset.
    """

    def __init__(
        self,
        sample_row: dict,
        workload_rows: list[dict],
        search_column: str = "title_nfc",
        base_seed: int = 2026,
    ):

        self.sample = sample_row
        self.workload_rows = workload_rows
        self.search_column = search_column
        self.base_seed = base_seed

        self.corpus_id = sample_row["corpus_id"]
        self.language = sample_row["language"]
        self.sample_size = int(sample_row["sample_size"])

        if sample_row["status"] != "PASS":
            raise BenchmarkWorkloadError(
                "Source sample has not passed validation."
            )

        # Load and verify the source corpus once.
        self.titles = load_corpus(
            sample_path=PROJECT_ROOT / sample_row["sample_file"],
            expected_count=self.sample_size,
            expected_sha256=sample_row["sample_sha256"],
            search_column=search_column,
        )

        self.title_set = set(self.titles)

        if len(self.title_set) != self.sample_size:
            raise BenchmarkWorkloadError(
                "Source corpus contains duplicate titles."
            )

        # Load one reusable pool of 100,000 unique negative queries.
        negative_row = self._find_workload(
            workload_type="unsuccessful",
            query_count=100000,
        )

        negative_records = self._read_workload(
            negative_row
        )

        self.negative_candidates = [
            query
            for query, expected in negative_records
            if expected == "false"
        ]

        if len(self.negative_candidates) != 100000:
            raise BenchmarkWorkloadError(
                "Negative candidate pool must contain "
                "100,000 queries."
            )

        if len(set(self.negative_candidates)) != 100000:
            raise BenchmarkWorkloadError(
                "Negative candidate pool contains duplicates."
            )

        if any(
            query in self.title_set
            for query in self.negative_candidates
        ):
            raise BenchmarkWorkloadError(
                "Negative candidate pool contains a corpus title."
            )

    def _find_workload(
        self,
        workload_type: str,
        query_count: int,
    ) -> dict:
        """Find exactly one matching validated workload."""

        matches = [
            row for row in self.workload_rows
            if row["corpus_id"] == self.corpus_id
            and int(row["sample_size"]) == self.sample_size
            and int(row["query_count"]) == query_count
            and row["workload_type"] == workload_type
            and row["status"] == "PASS"
        ]

        if len(matches) != 1:
            raise BenchmarkWorkloadError(
                f"Expected one {workload_type} workload "
                f"for {self.corpus_id}, n={self.sample_size}, "
                f"q={query_count}; found {len(matches)}."
            )

        return matches[0]

    def _read_workload(
        self,
        workload_row: dict,
    ) -> list[tuple[str, str]]:
        """Read and verify a workload file."""

        path = PROJECT_ROOT / workload_row["workload_file"]

        if not path.is_file():
            raise BenchmarkWorkloadError(
                f"Workload file missing: {path}"
            )

        actual_hash = calculate_sha256(path)

        if actual_hash != workload_row["workload_sha256"]:
            raise BenchmarkWorkloadError(
                f"Workload checksum mismatch: {path.name}"
            )

        expected_count = int(workload_row["query_count"])

        records = []
        result_counts = Counter()

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames != EXPECTED_COLUMNS:
                raise BenchmarkWorkloadError(
                    f"Incorrect workload CSV structure: {path.name}"
                )

            for expected_id, row in enumerate(
                reader,
                start=1,
            ):

                if int(row["query_id"]) != expected_id:
                    raise BenchmarkWorkloadError(
                        f"Invalid query ID: {path.name}"
                    )

                query = row["query"]
                result = row["expected_found"]

                if not query:
                    raise BenchmarkWorkloadError(
                        f"Empty query: {path.name}"
                    )

                if result not in ("true", "false"):
                    raise BenchmarkWorkloadError(
                        f"Invalid expected result: {path.name}"
                    )

                # Independently verify query membership.
                exists = query in self.title_set

                if result == "true" and not exists:
                    raise BenchmarkWorkloadError(
                        f"Positive query is absent: {query!r}"
                    )

                if result == "false" and exists:
                    raise BenchmarkWorkloadError(
                        f"Negative query exists in corpus: {query!r}"
                    )

                records.append((query, result))
                result_counts[result] += 1

        if len(records) != expected_count:
            raise BenchmarkWorkloadError(
                f"Query count mismatch: {path.name}"
            )

        workload_type = workload_row["workload_type"]

        if workload_type == "successful":
            if result_counts["true"] != expected_count:
                raise BenchmarkWorkloadError(
                    "Successful workload contains incorrect labels."
                )

        elif workload_type == "unsuccessful":
            if result_counts["false"] != expected_count:
                raise BenchmarkWorkloadError(
                    "Unsuccessful workload contains incorrect labels."
                )

        elif workload_type == "mixed":
            if (
                result_counts["true"] != expected_count // 2
                or result_counts["false"] != expected_count // 2
            ):
                raise BenchmarkWorkloadError(
                    "Mixed workload does not have a 50:50 ratio."
                )

        return records

    def prepare(
        self,
        workload_type: str,
        query_count: int,
        repetition_pattern: str,
    ) -> tuple[list[tuple[str, str]], dict]:
        """
        Prepare a query sequence for one benchmark experiment.

        Returns:
            A list of (query, expected_result) records,
            plus reproducibility and workload metadata.
        """

        if workload_type not in (
            "successful",
            "unsuccessful",
            "mixed",
        ):
            raise ValueError(
                f"Unsupported workload type: {workload_type}"
            )

        if repetition_pattern not in (
            "same_query_set",
            "mostly_unique",
            "mixed",
        ):
            raise ValueError(
                f"Unsupported repetition pattern: "
                f"{repetition_pattern}"
            )

        source_row = self._find_workload(
            workload_type,
            query_count,
        )

        source_records = self._read_workload(source_row)

        seed = derive_repetition_seed(
            base_seed=self.base_seed,
            corpus_id=self.corpus_id,
            sample_size=self.sample_size,
            workload_type=workload_type,
            query_count=query_count,
            pattern=repetition_pattern,
        )

        prepared_records = generate_repetition_pattern(
            records=source_records,
            pattern=repetition_pattern,
            seed=seed,
            positive_candidates=self.titles,
            negative_candidates=self.negative_candidates,
        )

        # Final validation before returning the workload.
        source_counts = Counter(
            result for _, result in source_records
        )

        prepared_counts = Counter(
            result for _, result in prepared_records
        )

        if len(prepared_records) != query_count:
            raise BenchmarkWorkloadError(
                "Prepared workload has incorrect query count."
            )

        if source_counts != prepared_counts:
            raise BenchmarkWorkloadError(
                "Prepared workload changed the result distribution."
            )

        for query, expected in prepared_records:

            exists = query in self.title_set

            if expected == "true" and not exists:
                raise BenchmarkWorkloadError(
                    f"Prepared positive query is absent: {query!r}"
                )

            if expected == "false" and exists:
                raise BenchmarkWorkloadError(
                    f"Prepared negative query exists: {query!r}"
                )

        unique_count = len({
            query for query, _ in prepared_records
        })

        metadata = {
            "corpus_id": self.corpus_id,
            "language": self.language,
            "sample_size": self.sample_size,
            "workload_type": workload_type,
            "query_count": query_count,
            "repetition_pattern": repetition_pattern,
            "base_seed": self.base_seed,
            "derived_seed": seed,
            "successful_count": prepared_counts["true"],
            "unsuccessful_count": prepared_counts["false"],
            "unique_query_count": unique_count,
            "repeated_query_occurrences": (
                query_count - unique_count
            ),
            "source_workload_file": source_row["workload_file"],
            "status": "PASS",
        }

        return prepared_records, metadata


def read_manifest(path: Path) -> list[dict]:
    """Load a CSV manifest."""

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))
