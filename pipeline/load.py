"""Load the metadata CSV and the parsed documents."""

import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class Doc:
    id: str
    genre: str
    language: str
    source_file: str | None  # e.g. "data/original/D00a47577ef.pdf"; None when listed but missing
    ext: str | None  # "pdf" or "html"
    raw: str | None  # parsed markdown, None when missing


def load_docs(data_dir: Path = DATA) -> list[Doc]:
    """Every row of documentation.csv, with the parsed text when the file exists."""
    if not (data_dir / "original").is_dir() or not (data_dir / "documentation.csv").is_file():
        raise FileNotFoundError("Challenge dataset missing. See README.md, or run python -m pipeline.demo for the redistributable demo.")
    originals = {p.stem: p for p in (data_dir / "original").iterdir() if p.is_file()}
    docs = []
    with open(data_dir / "documentation.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = data_dir / "parsed" / f"{row['id']}.md"
            original = originals.get(row["id"])
            has_file = parsed.exists() and original is not None
            docs.append(Doc(
                id=row["id"],
                genre=row["genre"],
                language=row["language"],
                source_file=str(original.relative_to(data_dir.parent)) if has_file else None,
                ext=original.suffix.lstrip(".") if has_file else None,
                raw=parsed.read_text(encoding="utf-8") if has_file else None,
            ))
    return docs
