from collections.abc import Sequence


def linear_search(titles: Sequence[str], query: str) -> bool:
    """
    Search for an exact title using Linear Search.

    Parameters:
        titles: Sequence of NFC-normalized title strings.
        query: NFC-normalized title to search for.

    Returns:
        True if the exact query exists, otherwise False.

    Notes:
        - Searches titles sequentially from beginning to end.
        - Uses exact, case-sensitive string comparison.
        - Does not modify or normalize the input data.
        - Does not require sorting or index construction.
    """

    for title in titles:
        if title == query:
            return True

    return False