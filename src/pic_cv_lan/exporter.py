from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def ensure_output_dir(output_dir: str | Path = "outputs") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_rows_to_tsv(
    rows: Iterable[dict[str, object]],
    output_path: str | Path = "outputs/latest_image_vocab.tsv",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["English", "中文", "Deutsch", "Confidence", "Source"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return path
