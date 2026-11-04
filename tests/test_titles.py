import json

from pipeline import titles


def test_load_titles_keeps_only_verbatim_titles(tmp_path):
    path = tmp_path / "titles.json"
    path.write_text(json.dumps({"titles": {
        "a": "Decentralisation of multidrug-resistant tuberculosis care",
        "b": "A title the model made up entirely",
        "c": None,
    }}))
    texts = {
        "a": "Lancet 2013\nDecentralisation of multidrug-\nresistant tuberculosis care\nJ Smith",
        "b": "Some other document text about cholera.",
        "c": "No title here.",
    }
    assert titles.load_titles(texts, path) == {"a": "Decentralisation of multidrug-resistant tuberculosis care"}


def test_parse_accepts_fenced_json_and_null():
    assert titles._parse('```json\n{"title": "Ebola  in\\nGuinea"}\n```') == "Ebola in Guinea"
    assert titles._parse('{"title": null}') is None
