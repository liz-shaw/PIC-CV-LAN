from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VocabEntry:
    english: str
    chinese: str
    german: str
    note: str = ""


class VocabStore:
    """Load and query local English/Chinese/German vocabulary."""

    def __init__(self, vocab_path: str | Path = "vocab.tsv") -> None:
        self.vocab_path = Path(vocab_path)
        self.entries: dict[str, VocabEntry] = {}
        self.load()

    @staticmethod
    def normalize(label: str) -> str:
        return label.strip().lower().replace("_", " ")

    def load(self) -> None:
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        with self.vocab_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            required = {"English", "中文", "Deutsch"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError("vocab.tsv must contain: English, 中文, Deutsch")

            for row in reader:
                english = self.normalize(row["English"])
                if not english:
                    continue
                self.entries[english] = VocabEntry(
                    english=english,
                    chinese=row.get("中文", "").strip() or "待补充",
                    german=row.get("Deutsch", "").strip() or "待补充",
                    note=row.get("Note", "").strip(),
                )

    def lookup(self, english_label: str) -> VocabEntry | None:
        return self.entries.get(self.normalize(english_label))


def append_unknown_word(
    english: str,
    unknown_path: str | Path = "unknown_words.tsv",
    source: str = "image-detect",
) -> None:
    """Append unknown word once to unknown_words.tsv."""

    path = Path(unknown_path)
    normalized = VocabStore.normalize(english)

    existing: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                value = row.get("English", "")
                if value:
                    existing.add(VocabStore.normalize(value))

    if normalized in existing:
        return

    need_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        fieldnames = ["English", "中文", "Deutsch", "Status", "Source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if need_header:
            writer.writeheader()
        writer.writerow(
            {
                "English": normalized,
                "中文": "待补充",
                "Deutsch": "待补充",
                "Status": "未处理",
                "Source": source,
            }
        )
