
import csv
import hashlib
import unicodedata

from pathlib import Path


class DatasetLoadError(ValueError):
    """Raised when a sample corpus fails loading or validation."""


def calculate_sha256(
    file_path: str | Path,
    chunk_size: int = 1024 * 1024
) -> str:
    """
    Calculate the SHA-256 checksum of a file.

    Reads the file in chunks to avoid loading it entirely into memory.
    """

    path = Path(file_path)

    if not path.is_file():
        raise DatasetLoadError(f"Dataset file not found: {path}")

    checksum = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            checksum.update(chunk)

    return checksum.hexdigest()


def load_corpus(
    sample_path: str | Path,
    expected_count: int | None = None,
    expected_sha256: str | None = None,
    search_column: str = "title_nfc"
) -> list[str]:
    """
    Load the search-title column from a validated sample CSV.

    Parameters:
        sample_path:
            Path to the processed sample CSV.

        expected_count:
            Expected number of title records.

        expected_sha256:
            Expected SHA-256 checksum. If supplied, the file
            checksum is verified before loading.

        search_column:
            CSV column used for exact title retrieval.

    Returns:
        A list containing the NFC-normalized title strings.

    Important:
        - Does not modify source files.
        - Does not normalize titles again.
        - Preserves case, whitespace, punctuation and underscores.
        - Checks that every title is already NFC-normalized.
    """

    path = Path(sample_path)

    if not path.is_file():
        raise DatasetLoadError(
            f"Sample corpus does not exist: {path}"
        )

    # Verify the file checksum when an expected hash is supplied.
    if expected_sha256 is not None:
        actual_sha256 = calculate_sha256(path)

        if actual_sha256.lower() != expected_sha256.lower():
            raise DatasetLoadError(
                f"SHA-256 mismatch for {path.name}. "
                f"Expected: {expected_sha256}, "
                f"Actual: {actual_sha256}"
            )

    titles = []

    try:
        with path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.reader(file, strict=True)

            header = next(reader, None)

            if header is None:
                raise DatasetLoadError(
                    f"CSV file is empty: {path}"
                )

            if len(header) != len(set(header)):
                raise DatasetLoadError(
                    f"Duplicate column names found in {path.name}"
                )

            if search_column not in header:
                raise DatasetLoadError(
                    f"Required search column '{search_column}' "
                    f"not found in {path.name}"
                )

            title_index = header.index(search_column)

            for record_number, row in enumerate(reader, start=2):

                if len(row) != len(header):
                    raise DatasetLoadError(
                        f"Invalid CSV record at row {record_number} "
                        f"in {path.name}"
                    )

                title = row[title_index]

                if title == "":
                    raise DatasetLoadError(
                        f"Empty title at row {record_number} "
                        f"in {path.name}"
                    )

                if not unicodedata.is_normalized("NFC", title):
                    raise DatasetLoadError(
                        f"Non-NFC title at row {record_number} "
                        f"in {path.name}"
                    )

                titles.append(title)

    except UnicodeDecodeError as error:
        raise DatasetLoadError(
            f"Invalid UTF-8 encoding in {path.name}"
        ) from error

    except csv.Error as error:
        raise DatasetLoadError(
            f"Invalid CSV structure in {path.name}"
        ) from error

    if expected_count is not None:
        if len(titles) != expected_count:
            raise DatasetLoadError(
                f"Record count mismatch for {path.name}. "
                f"Expected: {expected_count}, "
                f"Actual: {len(titles)}"
            )

    return titles
