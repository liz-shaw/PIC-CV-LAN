from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


DEFAULT_FIELDNAMES = ["English", "中文", "Deutsch", "Count", "Confidence", "Source"]


def ensure_output_dir(output_dir: str | Path = "outputs") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_rows_to_tsv(
    rows: Iterable[dict[str, object]],
    output_path: str | Path = "outputs/latest_image_vocab.tsv",
    fieldnames: list[str] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    selected_fieldnames = fieldnames or DEFAULT_FIELDNAMES

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in selected_fieldnames})

    return path
