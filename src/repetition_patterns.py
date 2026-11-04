
import math
import random
import zlib

from collections import Counter
from collections.abc import Sequence


VALID_PATTERNS = {
    "same_query_set",
    "mostly_unique",
    "mixed",
}

VALID_RESULTS = {"true", "false"}

QueryRecord = tuple[str, str]


def derive_repetition_seed(
    base_seed: int,
    corpus_id: str,
    sample_size: int,
    workload_type: str,
    query_count: int,
    pattern: str,
) -> int:
    """Derive a stable seed for each repetition experiment."""

    if pattern not in VALID_PATTERNS:
        raise ValueError(f"Unknown repetition pattern: {pattern}")

    identity = (
        f"{corpus_id}|{sample_size}|"
        f"{workload_type}|{query_count}|{pattern}"
    )

    checksum = zlib.crc32(identity.encode("utf-8"))

    return (base_seed + checksum) % (2 ** 32)


def validate_query_records(
    records: Sequence[QueryRecord],
) -> None:
    """Validate query records before applying a pattern."""

    if not records:
        raise ValueError("Query record sequence cannot be empty.")

    for index, record in enumerate(records):

        if not isinstance(record, tuple) or len(record) != 2:
            raise ValueError(
                f"Invalid query record at position {index}."
            )

        query, expected_result = record

        if not isinstance(query, str) or not query:
            raise ValueError(
                f"Invalid query at position {index}."
            )

        if expected_result not in VALID_RESULTS:
            raise ValueError(
                f"Invalid expected result at position {index}."
            )


def sample_unique_then_repeat(
    candidates: Sequence[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    """
    Select distinct candidates before introducing repetitions.

    If count is greater than the candidate pool size, every
    candidate is selected once before additional selections
    are made with replacement.
    """

    if count < 0:
        raise ValueError("Count cannot be negative.")

    if count == 0:
        return []

    if not candidates:
        raise ValueError("Candidate pool cannot be empty.")

    first_count = min(count, len(candidates))

    selected = rng.sample(
        candidates,
        k=first_count,
    )

    remaining = count - first_count

    if remaining:
        selected.extend(
            rng.choices(
                candidates,
                k=remaining,
            )
        )

    return selected


def sample_hot_cold(
    candidates: Sequence[str],
    count: int,
    rng: random.Random,
    hot_fraction: float = 0.10,
    hot_access_fraction: float = 0.50,
) -> list[str]:
    """
    Generate a hot/cold query distribution.

    Hot queries:
        Drawn with replacement from a small hot subset.

    Cold queries:
        Drawn from the remaining candidates without replacement
        wherever possible, with replacement only after exhaustion.
    """

    if count < 0:
        raise ValueError("Count cannot be negative.")

    if count == 0:
        return []

    if not candidates:
        raise ValueError("Candidate pool cannot be empty.")

    if not 0 < hot_fraction < 1:
        raise ValueError(
            "Hot fraction must be between zero and one."
        )

    if not 0 <= hot_access_fraction <= 1:
        raise ValueError(
            "Hot access fraction must be between zero and one."
        )

    # Handle a one-element candidate pool.
    if len(candidates) == 1:
        return rng.choices(candidates, k=count)

    hot_size = max(
        1,
        math.ceil(len(candidates) * hot_fraction),
    )

    hot_size = min(hot_size, len(candidates) - 1)

    hot_pool = rng.sample(
        candidates,
        k=hot_size,
    )

    hot_set = set(hot_pool)

    cold_pool = [
        candidate
        for candidate in candidates
        if candidate not in hot_set
    ]

    hot_count = round(count * hot_access_fraction)
    cold_count = count - hot_count

    hot_queries = (
        rng.choices(hot_pool, k=hot_count)
        if hot_count
        else []
    )

    cold_queries = sample_unique_then_repeat(
        cold_pool,
        cold_count,
        rng,
    )

    result = hot_queries + cold_queries

    rng.shuffle(result)

    return result


def generate_repetition_pattern(
    records: Sequence[QueryRecord],
    pattern: str,
    seed: int,
    positive_candidates: Sequence[str] | None = None,
    negative_candidates: Sequence[str] | None = None,
    hot_fraction: float = 0.10,
    hot_access_fraction: float = 0.50,
) -> list[QueryRecord]:
    """
    Generate a reproducible query sequence for a repetition pattern.

    Parameters:
        records:
            Original query records as (query, expected_result).

        pattern:
            same_query_set, mostly_unique, or mixed.

        seed:
            Deterministic random seed.

        positive_candidates:
            Available successful queries, normally the full corpus.

        negative_candidates:
            Available unsuccessful queries from the validated
            unsuccessful-query pool.

        hot_fraction:
            Fraction of candidates forming the hot subset.

        hot_access_fraction:
            Fraction of accesses directed to the hot subset.

    Returns:
        A new sequence of query records.

    Important:
        - The number of records is preserved.
        - The number of successful and unsuccessful queries
          is preserved separately.
        - Input records are never modified.
        - Query order is reproducible for a given seed.
    """

    validate_query_records(records)

    if pattern not in VALID_PATTERNS:
        raise ValueError(
            f"Unsupported repetition pattern: {pattern}"
        )

    original_records = list(records)

    # Baseline: use the exact original query sequence.
    if pattern == "same_query_set":
        return original_records.copy()

    # Count how many queries belong to each outcome class.
    outcome_counts = Counter(
        expected_result
        for _, expected_result in original_records
    )

    rng = random.Random(seed)

    output_records = []

    for expected_result in ("true", "false"):

        required_count = outcome_counts[expected_result]

        if required_count == 0:
            continue

        if expected_result == "true":
            candidates = positive_candidates
        else:
            candidates = negative_candidates

        # Fallback for small isolated unit tests.
        if candidates is None:
            candidates = list(dict.fromkeys(
                query
                for query, result in original_records
                if result == expected_result
            ))

        if not candidates:
            raise ValueError(
                f"No candidate pool available for "
                f"expected result {expected_result}."
            )

        if pattern == "mostly_unique":

            selected_queries = sample_unique_then_repeat(
                candidates,
                required_count,
                rng,
            )

        elif pattern == "mixed":

            selected_queries = sample_hot_cold(
                candidates,
                required_count,
                rng,
                hot_fraction=hot_fraction,
                hot_access_fraction=hot_access_fraction,
            )

        else:
            raise ValueError(
                f"Unsupported repetition pattern: {pattern}"
            )

        output_records.extend(
            (query, expected_result)
            for query in selected_queries
        )

    # Randomize positive and negative query positions together.
    rng.shuffle(output_records)

    if len(output_records) != len(original_records):
        raise RuntimeError(
            "Repetition pattern changed the query count."
        )

    new_outcome_counts = Counter(
        expected_result
        for _, expected_result in output_records
    )

    if new_outcome_counts != outcome_counts:
        raise RuntimeError(
            "Repetition pattern changed the result distribution."
        )

    return output_records
