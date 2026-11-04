
"""
Validation of generated nested corpus samples.

Checks:
    - Required files exist
    - Correct CSV schema
    - Exact record counts
    - No empty titles
    - NFC consistency
    - No duplicate values within each sample
    - Smaller samples are prefixes of larger samples
"""

import csv
import unicodedata

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_titles(path):

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != ["title_nfc"]:
            raise AssertionError(
                f"Wrong CSV schema: {path}"
            )

        for row in reader:

            title = row["title_nfc"]

            if not title:
                raise AssertionError(
                    f"Empty title in {path}"
                )

            if not unicodedata.is_normalized("NFC", title):
                raise AssertionError(
                    f"Non-NFC title in {path}"
                )

            yield title


def validate_dataset(
    language,
    sizes,
    seed=2026
):

    directory = ROOT / "data/processed" / language

    previous_path = None

    for size in sizes:

        filename = (
            f"{language}_n{size}_seed{seed}.csv"
        )

        current_path = directory / filename

        if not current_path.exists():
            raise AssertionError(
                f"Missing file: {current_path}"
            )

        # Independent duplicate check for this sample.
        seen = set()

        count = 0

        for title in read_titles(current_path):

            if title in seen:
                raise AssertionError(
                    f"Duplicate in sample: {title!r}"
                )

            seen.add(title)
            count += 1

        if count != size:
            raise AssertionError(
                f"{filename}: expected {size}, found {count}"
            )

        print(
            f"{filename}: {count:,} records — PASS"
        )

        # Compare prefixes to establish nested sampling.
        if previous_path is not None:

            previous = read_titles(previous_path)
            current = read_titles(current_path)

            for index, previous_title in enumerate(previous):

                try:
                    current_title = next(current)
                except StopIteration:
                    raise AssertionError(
                        "Larger sample ended unexpectedly."
                    )

                if previous_title != current_title:
                    raise AssertionError(
                        f"Nested sample mismatch at row "
                        f"{index + 1}"
                    )

            print(
                f"Nested relation verified: "
                f"{previous_path.name} -> {filename}"
            )

        previous_path = current_path

    print(f"\nALL {language.upper()} SAMPLE TESTS PASSED")


if __name__ == "__main__":

    validate_dataset(
        "tulu",
        [1000, 2500, 4000, 4536]
    )
