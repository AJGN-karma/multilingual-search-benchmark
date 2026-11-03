"""
Unicode normalization utilities for the multilingual
Wikipedia title retrieval benchmark.

Policy:
    - Unicode NFC normalization
    - Case-sensitive matching
    - Preserve whitespace
    - Preserve punctuation
    - Preserve underscores
    - Do not remove accents or diacritics
    - Do not transliterate text
"""

import unicodedata


NORMALIZATION_FORM = "NFC"


def normalize_title(title: str) -> str:
    """
    Normalize a Wikipedia title using Unicode NFC.

    Parameters
    ----------
    title : str
        Original title string.

    Returns
    -------
    str
        NFC-normalized title.

    Raises
    ------
    TypeError
        If the input is not a string.
    """

    if not isinstance(title, str):
        raise TypeError(
            "Title must be a string. "
            f"Received: {type(title).__name__}"
        )

    return unicodedata.normalize(NORMALIZATION_FORM, title)


def normalize_query(query: str) -> str:
    """
    Apply the same normalization policy to a search query.

    This ensures that corpus titles and user queries
    use identical comparison semantics.
    """

    return normalize_title(query)


def is_nfc_normalized(text: str) -> bool:
    """
    Check whether a string is already in NFC form.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    return unicodedata.is_normalized(NORMALIZATION_FORM, text)


def run_self_tests() -> None:
    """
    Run small internal tests for Unicode normalization.
    """

    # Canonically equivalent Unicode representations
    decomposed = "e\u0301"
    composed = "é"

    assert normalize_title(decomposed) == composed

    # Underscores must be preserved
    assert normalize_title("Main_Page") == "Main_Page"

    # Whitespace must be preserved
    assert normalize_title(" Book Title ") == " Book Title "

    # Case must be preserved
    assert normalize_title("Book") != normalize_title("book")

    # Punctuation must be preserved
    assert normalize_title("Book-Title") != normalize_title("Book Title")

    # Kannada text must remain valid
    kannada = "ಕನ್ನಡ"
    assert normalize_title(kannada) == kannada

    # Query normalization must follow the same rule
    assert normalize_query(decomposed) == composed

    # Empty strings remain empty
    assert normalize_title("") == ""

    # Invalid input must raise TypeError
    try:
        normalize_title(123)
    except TypeError:
        pass
    else:
        raise AssertionError("Invalid input was not rejected.")

    print("======================================")
    print("UNICODE NORMALIZATION SELF-TEST")
    print("======================================")
    print("NFC normalization: PASS")
    print("Underscore preservation: PASS")
    print("Whitespace preservation: PASS")
    print("Case-sensitive policy: PASS")
    print("Punctuation preservation: PASS")
    print("Kannada text handling: PASS")
    print("Query normalization: PASS")
    print("Empty string handling: PASS")
    print("Invalid input handling: PASS")
    print("======================================")
    print("ALL NORMALIZATION TESTS PASSED")
    print("======================================")


if __name__ == "__main__":
    run_self_tests()