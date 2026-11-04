
from collections.abc import Sequence, Set


def build_hash_index(titles: Sequence[str]) -> set[str]:
    """
    Construct a hash-table index using Python's built-in set.

    Parameters:
        titles: Sequence of NFC-normalized title strings.

    Returns:
        A set containing the title strings.
    """

    return set(titles)


def hash_search(title_index: Set[str], query: str) -> bool:
    """
    Search for an exact title using hash-table lookup.

    Parameters:
        title_index: Previously constructed hash-table index.
        query: NFC-normalized title to search for.

    Returns:
        True if the exact query exists, otherwise False.

    Notes:
        - Uses Python's set membership operation.
        - Matching is exact and case-sensitive.
        - Does not build the index or normalize the query.
    """

    return query in title_index
