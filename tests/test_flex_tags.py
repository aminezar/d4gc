import json

import pytest

from pipeline.flex_tags import UNSURE, load_tags, normalize, verify

RAW = (
    "The late delivery of notebooks to Cedar and \n"
    "Pine is a minor limitation of the Preserving \n"
    "Eﬀ ective File Transfer study. The ﬁ ndings of \n"
    "the SAMPLE reviewers ﬁnd useful and original examples. Multidrug-\n"
    "resistant tuberculosis is now an epidemic. Treat-\nment was “free” – mostly.\n"
    "## Outcomes\n\n**Cholera** cases rose sharply in the camp after the floods."
)


def test_exact_quote():
    assert verify("is a minor limitation of the Preserving", RAW)


def test_case_and_whitespace_insensitive():
    assert verify("the LATE delivery of   notebooks to cedar and pine", RAW)


def test_ligatures_nfkc():
    assert verify("the SAMPLE reviewers find useful and original", RAW)


def test_ligature_with_stray_space():
    # PDF text "Eﬀ ective" / "ﬁ ndings" -> model writes the fixed word
    assert verify("Preserving Effective File Transfer study. The findings of", RAW)


def test_hyphenated_line_break_both_forms():
    assert verify("Multidrug-resistant tuberculosis is now an epidemic", RAW)
    assert verify("Treatment was \"free\" - mostly", RAW)
    assert verify("Treatment was “free” – mostly", RAW)


def test_quotes_and_dashes_normalised():
    assert normalize("“free” – ‘x’") == normalize('"free" - \'x\'')


def test_works_on_cleaned_text():
    cleaned = "Cholera cases rose sharply in the camp after the floods."
    quote = "**Cholera** cases rose sharply in the camp"
    assert verify(quote, cleaned)
    assert verify("Cholera cases rose sharply in the camp", RAW)


def test_rejects_paraphrase_and_absent():
    assert not verify("Cholera cases increased sharply in the camp after the floods", RAW)
    assert not verify("Ebola outbreak was declared in the province", RAW)


def test_rejects_empty_and_trivial():
    assert not verify("", RAW)
    assert not verify("the", RAW)
    assert not verify("quote", "")


def test_arabic_and_french():
    ar = "يكتب الفريق أسماء الملفات التجريبية في دفتر التدريب بعد نهاية الجلسة"
    assert verify("أسماء الملفات التجريبية في دفتر التدريب بعد نهاية الجلسة", ar)
    fr = "Depuis 2011, le conflit entre groupes d’opposition armés"
    assert verify("le conflit entre groupes d'opposition armés", fr)


@pytest.fixture
def tag_file(tmp_path):
    data = {
        "meta": {},
        "tags": {
            "A": [
                {"tag": "cholera", "evidence": "Cholera cases rose sharply in the camp"},
                {"tag": "ebola", "evidence": "Ebola outbreak was declared in the province"},
                {"tag": "earthquake", "evidence": "after the floods"},
                {"tag": "tuberculosis", "evidence": ""},
                {"tag": "Cholera", "evidence": "Cholera cases rose sharply in the camp"},
            ],
            "B": [],
            "C": [{"tag": "hiv", "evidence": "not used"}],
        },
        "failures": {"D": "timeout"},
    }
    p = tmp_path / "flex_tags.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_tags(tag_file):
    out = load_tags({"A": RAW, "B": "some text", "D": "x"}, tag_file)
    assert set(out) == {"A", "B", "D"}  # C not requested
    assert out["A"] == [
        ("cholera", "Cholera cases rose sharply in the camp"),
        (UNSURE, "Ebola outbreak was declared in the province"),
        (UNSURE, "after the floods"),  # tag outside vocabulary
        (UNSURE, "tuberculosis"),  # no evidence -> tag name
    ]  # duplicate (case-insensitive tag) dropped
    assert out["B"] == []
    assert out["D"] == []  # failed doc: no tags


def test_load_tags_flat_format(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"A": [{"tag": "cholera", "evidence": "Cholera cases rose sharply in the camp"}]}))
    assert load_tags({"A": RAW}, p) == {"A": [("cholera", "Cholera cases rose sharply in the camp")]}
