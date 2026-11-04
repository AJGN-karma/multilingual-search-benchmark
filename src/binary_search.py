
from collections.abc import Sequence


def build_sorted_titles(titles: Sequence[str]) -> list[str]:
    """
    Construct a sorted copy of the title collection.

    Sorting is performed separately so that its preprocessing
    time can be measured independently of search time.

    The original collection is not modified.
    """

    return sorted(titles)


def binary_search(sorted_titles: Sequence[str], query: str) -> bool:
    """
    Search for an exact title using iterative Binary Search.

    Parameters:
        sorted_titles: Titles sorted in ascending Python string order.
        query: NFC-normalized title to search for.

    Returns:
        True if the exact query exists, otherwise False.

    Important:
        The input collection must already be sorted.
        This function does not sort or normalize the data.
    """

    low = 0
    high = len(sorted_titles) - 1

    while low <= high:

        mid = (low + high) // 2

        if sorted_titles[mid] == query:
            return True

        elif sorted_titles[mid] < query:
            low = mid + 1

        else:
            high = mid - 1

    return False
