from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_FIELDS = ["English", "中文", "Deutsch"]


def validate_vocab(path: str = "vocab.tsv") -> None:
    vocab_path = Path(path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Not found: {vocab_path}")

    with vocab_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Missing header row")

        missing = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        bad_rows: list[int] = []
        seen: set[str] = set()
        duplicates: list[str] = []

        for line_no, row in enumerate(reader, start=2):
            english = row.get("English", "").strip().lower()
            chinese = row.get("中文", "").strip()
            german = row.get("Deutsch", "").strip()

            if not english or not chinese or not german:
                bad_rows.append(line_no)

            if english in seen:
                duplicates.append(english)
            seen.add(english)

    if bad_rows:
        raise ValueError(f"Rows with empty required fields: {bad_rows}")
    if duplicates:
        raise ValueError(f"Duplicate English entries: {duplicates}")

    print(f"OK: {vocab_path} is valid.")


if __name__ == "__main__":
    validate_vocab()
