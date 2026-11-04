import csv
import time
from pathlib import Path

import pytest

from pipeline.clean import clean_text, extract_title, html_page_title

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "data" / "parsed"
ORIGINAL = ROOT / "data" / "original"


def paras(text: str) -> list[str]:
    return text.split("\n\n")


# --- characters --------------------------------------------------------------


def test_ligatures_and_split_ligatures():
    raw = "Preserving Eﬀ ective TB Treatment.\nThe ﬁ ndings were ﬂ agged by oﬃcials.\n"
    out = clean_text(raw, "pdf")
    assert "Effective" in out
    assert "findings" in out and "flagged" in out and "officials" in out
    assert "ﬁ" not in out and "ﬀ" not in out


def test_soft_hyphen_line_break_is_joined():
    out = clean_text("clinical and anthropo­\nmetric data were collected from all children.", "pdf")
    assert "anthropometric data" in out


def test_control_chars_removed():
    out = clean_text("Adults on HAART 24 \x01 2 months were included.", "pdf")
    assert "\x01" not in out
    assert "24" in out and "2 months" in out


# --- PDF reflow ---------------------------------------------------------------


def test_pdf_lines_merge_into_paragraphs():
    raw = (
        "The demonstration offices of Cedar and\n"
        "Pine, which between them store half\n"
        "of the training notebooks,\n"
        "are the main sites of the fictional\n"
        "exercise.\n"
        "Second-line shelf labels are printed\n"
        "in bright colours by a group of\n"
        "volunteers in the training room.\n"
    )
    out = clean_text(raw, "pdf")
    assert "\n" not in out.replace("\n\n", "")
    ps = paras(out)
    assert ps[0].startswith("The demonstration offices of Cedar and Pine")
    assert ps[0].endswith("sites of the fictional exercise.")
    assert ps[1].startswith("Second-line shelf labels")


def test_hyphenation_rejoined_but_real_hyphens_kept():
    raw = (
        "these inappropriate pre-\n"
        "scriptions would be dispensed with no checks and the\n"
        "share of multidrug-resistant cases is high, and most\n"
        "of the multidrug-\n"
        "resistant cases were given second-line drugs; a minimum of three new second-\n"
        "line drugs is needed for a long duration in total.\n"
    )
    out = clean_text(raw, "pdf")
    assert "prescriptions" in out
    assert "multidrug-resistant cases were" in out
    assert "new second-line drugs" in out


def test_capitalised_continuation_keeps_hyphen():
    out = clean_text("patients with MDR-\nTB and XDR were enrolled in the study cohort.", "pdf")
    assert "MDR-TB" in out


def test_running_headers_and_page_numbers_removed():
    topics = ["cholera", "measles", "malaria", "Ebola", "malnutrition"]
    pages = []
    for k, topic in enumerate(topics, start=1):
        pages.append(
            f"www.thelancet.com Vol 381 February 23, 2013\n"
            f"Paragraph {k} is about {topic} in the field and runs long\n"
            f"enough to look like body text in a column on {topic}.\n"
            f"{600 + k}\n"
        )
    out = clean_text("\n\n\n".join(pages), "pdf")
    # Header kept once (it carries the date), not on every page.
    assert out.count("www.thelancet.com") == 1
    for k in range(1, 6):
        assert f"Paragraph {k} is about" in out
        assert f"\n\n{600 + k}" not in out and not out.endswith(str(600 + k))


def test_table_numbers_at_page_edges_are_kept():
    # Numbers at page edges that do not form a page sequence are data.
    pages = ["Total cases\n4817", "Deaths recorded in the period\n92", "Children under five\n1350"]
    out = clean_text("\n\n\n".join(pages), "pdf")
    for n in ("4817", "92", "1350"):
        assert n in out


def test_paragraph_continues_across_page_break():
    raw = "The outbreak spread quickly through the\ncamp and the\n\n\nsurrounding villages within two weeks."
    out = clean_text(raw, "pdf")
    assert "the camp and the surrounding villages within two weeks." in out


# --- HTML --------------------------------------------------------------------


GUIDELINE_PAGE = """Skip to main content
A fictional navigation banner for this parser test.
Google Play
or the
App Store
.
×
MSF.org
MSF Staff
English
العربية
Español
Français
.هذه الصفحة غير متاحة باللغة العربية
Esta página no está disponible en este idioma.
Cette page n’est pas disponible dans cette langue.
This page is not available in this language.
Clinical guidelines - Diagnosis and treatment manual
Select language:
English
(Arabic) العربية
Español (Spanish)
Français (French)
Enter
Permalink
For curative programmes in hospitals and dispensaries
"""


def test_guidelines_chrome_removed_and_title_found():
    out = clean_text(GUIDELINE_PAGE, "html")
    assert paras(out) == [
        "Clinical guidelines - Diagnosis and treatment manual",
        "For curative programmes in hospitals and dispensaries",
    ]
    assert extract_title(GUIDELINE_PAGE, "html", out) == "Clinical guidelines - Diagnosis and treatment manual"


def test_html_markdown_stripped_and_lists_kept():
    raw = "## Background\n\nThe **outbreak** began in [Goma](https://x.org/a).\n\n- First point\n- Second point\n"
    out = clean_text(raw, "html")
    assert paras(out) == ["Background", "The outbreak began in Goma.", "• First point", "• Second point"]
    # "Background" is a generic label, so the title falls through to the text.
    assert extract_title(raw, "html", out) == "The outbreak began in Goma."


def test_html_does_not_touch_real_content_lines():
    raw = "Share the burden: MSF teams share supplies with the ministry.\n\nPermalink is not a word here.\n"
    out = clean_text(raw, "html")
    assert "Share the burden" in out and "Permalink is not a word here." in out


@pytest.mark.parametrize(
    "text",
    [
        "يحتوي هذا الدفتر التجريبي على أسماء الملفات وتاريخ الجلسة وملاحظات الفريق بعد انتهاء التدريب.",
        "Les équipes de MSF ont pris en charge des enfants souffrant de malnutrition sévère à Maradi.",
        "Los equipos de MSF atendieron a personas desplazadas por la violencia en Sudán.",
    ],
)
def test_non_english_text_survives(text):
    assert clean_text(text + "\n", "html") == text
    assert clean_text(text + "\n", "pdf") == text


# --- titles ------------------------------------------------------------------


def test_title_from_markdown_heading():
    raw = "# Madagascar: cholera response\n\nMSF teams are responding."
    assert extract_title(raw, "html", clean_text(raw, "html")) == "Madagascar: cholera response"


def test_title_from_pdf_lines_skips_journal_header():
    raw = (
        "www.thelancet.com Vol 381 February 23, 2013\n625\n"
        "Decentralisation of multidrug-resistant\ntuberculosis care and management\n"
        "John Smith1, Jane Doe2\n"
    )
    title = extract_title(raw, "pdf", clean_text(raw, "pdf"))
    assert title == "Decentralisation of multidrug-resistant tuberculosis care and management"


def test_title_never_empty():
    assert extract_title("", "html", "") == "Untitled document"
    long = "word " * 200
    t = extract_title(long, "html", clean_text(long, "html"))
    assert t and len(t) <= 201


# --- corpus ------------------------------------------------------------------


def _corpus():
    ext = {p.stem: p.suffix[1:] for p in ORIGINAL.iterdir()}
    return {i: (PARSED.joinpath(f"{i}.md").read_text(encoding="utf-8"), e) for i, e in ext.items()}


@pytest.mark.skipif(not PARSED.exists(), reason="corpus not available")
def test_corpus_cleaning():
    corpus = _corpus()
    lang = {r["id"]: r["language"] for r in csv.DictReader(open(ROOT / "data" / "documentation.csv"))}
    t0 = time.time()
    cleaned = {i: clean_text(raw, ext) for i, (raw, ext) in corpus.items()}
    assert time.time() - t0 < 30
    for i, text in cleaned.items():
        assert text.strip(), i
        assert "\n\n\n" not in text
        assert all("\n" not in p for p in text.split("\n\n")), i
        assert "ﬁ" not in text and "ﬀ" not in text
        title = extract_title(corpus[i][0], corpus[i][1], text)
        assert title.strip(), i
    assert "Effective TB Treatment" in cleaned["D00a47577ef"]
    # Arabic documents still read as Arabic.
    for i, text in cleaned.items():
        if lang.get(i) == "ar":
            arabic = sum("؀" <= ch <= "ۿ" for ch in text)
            assert arabic > 0.5 * sum(ch.isalpha() for ch in text), i


def test_html_page_title_drops_site_and_adds_guide():
    page = ("<html><head><title>Annexes | Guides médicaux MSF</title></head><body>"
            "<span>Sobrescribir enlaces de ayuda a la navegación</span><a>Accueil</a>"
            "<a>Prise en charge d’une ÉPIDÉMIE DE CHOLÉRA</a><span>Annexes</span></body></html>")
    assert html_page_title(page) == "Annexes (Prise en charge d’une ÉPIDÉMIE DE CHOLÉRA)"
    assert html_page_title("<title>Les traumatismes en Haïti | Journal Article / Research | MSF Science Portal</title>") \
        == "Les traumatismes en Haïti"
    assert html_page_title("<title>غزة | هجوم على سيارات المنظمة | أطباء بلا حدود</title>") == "غزة: هجوم على سيارات المنظمة"
    assert html_page_title("<p>no title</p>") == ""


def test_html_title_tag_wins_and_country_pages_keep_heading():
    raw = "## Responding to armed conflict in the northeast\n\nMSF teams work in Borno."
    assert extract_title(raw, "html", clean_text(raw, "html"), "<title>Nigeria | MSF</title>") \
        == "Nigeria: Responding to armed conflict in the northeast"
    raw = "## Current situation\n\nThe outbreak continues."
    assert extract_title(raw, "html", clean_text(raw, "html"), "<title>Democratic Republic of Congo: Ebola update June 2018 | MSF</title>") \
        == "Democratic Republic of Congo: Ebola update June 2018"
