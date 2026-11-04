from pathlib import Path

import pytest

from pipeline.clean import clean_text
from pipeline.hygiene import find_removed

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "data" / "parsed"
ORIGINAL = ROOT / "data" / "original"

BODY = " ".join(f"word{i} appears in sentence number {i} of the report." for i in range(200))


def test_identical_group_of_three_or_more_is_navigation():
    texts = {f"N{i}": "Latest News\n\nMenu item one\n\nMenu item two" for i in range(4)}
    texts["A"] = BODY
    assert find_removed(texts) == {f"N{i}": "navigation page" for i in range(4)}


def test_exact_pair_keeps_one_deterministically():
    texts = {"P07": BODY, "D9": BODY, "X": "something else entirely " * 20}
    assert find_removed(texts) == {"P07": "duplicate of D9"}


def test_exact_match_ignores_markup_and_case():
    texts = {"A": BODY + " [emergency department]", "B": BODY.upper() + " \\[emergency department\\]"}
    assert list(find_removed(texts).values()) == ["duplicate of A"]


def test_near_duplicate_keeps_longer():
    longer = BODY + " An extra closing sentence that only the longer edition has."
    texts = {"A": BODY, "B": longer}
    assert find_removed(texts) == {"A": "duplicate of B"}


def test_partial_overlap_is_not_duplicate():
    half = " ".join(f"other{i} text in a different report {i}." for i in range(200))
    texts = {"A": BODY, "B": BODY[: len(BODY) // 2] + " " + half}
    assert find_removed(texts) == {}


def test_empty_texts_do_not_crash():
    assert find_removed({"A": "", "B": BODY}) == {}


@pytest.mark.skipif(not PARSED.exists(), reason="corpus not available")
def test_corpus_removed_list():
    ext = {p.stem: p.suffix[1:] for p in ORIGINAL.iterdir()}
    texts = {i: clean_text(PARSED.joinpath(f"{i}.md").read_text(encoding="utf-8"), e) for i, e in ext.items()}
    removed = find_removed(texts)
    nav = sorted(i for i, r in removed.items() if r == "navigation page")
    dups = {i: r for i, r in removed.items() if r != "navigation page"}
    assert len(nav) == 18
    assert dups == {
        # Exact duplicates (P07/Dd4ee6d16b7 differ only by a markdown escape).
        "P07": "duplicate of Dd4ee6d16b7",
        "P16": "duplicate of D5da0aed2fd",
        "P42": "duplicate of Da68d33c442",
        "P44": "duplicate of Dbd6e2ef149",
        # Another edition of the Rwanda genocide case study (Jaccard 0.94).
        "Da384a9ee90": "duplicate of Da68d33c442",
    }
    assert len(removed) == 23
