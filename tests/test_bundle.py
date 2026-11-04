import sqlite3

import numpy as np
import pytest

from pipeline.bundle import (BundleDoc, bundle_candidates, export_subset, join_countries, split_countries,
                             write_bundle)
from pipeline.embed import DIMS


def vecs(n, seed):
    v = np.random.default_rng(seed).normal(size=(n, DIMS)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def docs():
    return [
        BundleDoc("A", "Cholera in Kinshasa", "Project Update", "fr", 2019, ["DR Congo"], "data/original/A.html",
                  ["Le choléra à Kinshasa.", "Traitement par SRO."], vecs(2, 1), [("cholera", "Le choléra")]),
        BundleDoc("B", "Malaria chapter", "Clinical Guideline Chapter", "en", 2018, [], "data/original/B.pdf",
                  ["Artesunate for severe malaria."], vecs(1, 2)),
        BundleDoc("C", "Caracas update", "Press Release", "es", 2020, ["Venezuela"], "data/original/C.html",
                  ["Medicamentos en Caracas."], vecs(1, 3)),
        BundleDoc("D", None, "Press Release", "en", None, [], None),  # listed but missing
    ]


def test_write_and_export(tmp_path):
    full = tmp_path / "full.sqlite"
    write_bundle(full, "test", docs(), {"X": "navigation page"})
    db = sqlite3.connect(full)
    meta = dict(db.execute("SELECT key, value FROM meta"))
    assert meta["dims"] == str(DIMS) and meta["doc_count"] == "3" and meta["chunk_count"] == "4"
    assert {"T_vec", "T_kw", "model", "valid_until", "created"} <= meta.keys()
    assert db.execute("SELECT countries FROM docs WHERE id='A'").fetchone()[0] == "DR Congo"
    assert db.execute("SELECT countries FROM docs WHERE id='B'").fetchone()[0] == ""
    blob = db.execute("SELECT vector FROM chunks WHERE doc_id='B'").fetchone()[0]
    assert np.isclose(np.linalg.norm(np.frombuffer(blob, np.float32)), 1.0)
    hits = db.execute("SELECT c.doc_id FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid "
                      "WHERE chunks_fts MATCH 'text:cholera'").fetchall()
    assert hits == [("A",)]  # diacritics removed: "choléra" matches "cholera"
    assert db.execute("SELECT * FROM removed").fetchall() == [("X", "navigation page")]

    cand = bundle_candidates(full, "DR Congo")
    assert [d["id"] for d in cand["country"]] == ["A"] and [d["id"] for d in cand["global"]] == ["B"]
    assert split_countries(join_countries(["DR Congo", "Uganda"])) == ["DR Congo", "Uganda"]

    info = export_subset(full, "DR Congo", tmp_path / "cod.sqlite")
    assert info["doc_count"] == 2 and info["chunk_count"] == 3
    sub = sqlite3.connect(tmp_path / "cod.sqlite")
    assert dict(sub.execute("SELECT key, value FROM meta"))["country"] == "DR Congo"
    assert sub.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'artesunate'").fetchone()[0] == 1
    assert sub.execute("SELECT tag FROM tags").fetchall() == [("cholera",)]
    sub_vec = sub.execute("SELECT vector FROM chunks WHERE doc_id='B'").fetchone()[0]
    assert sub_vec == blob

    only_a = export_subset(full, "DR Congo", tmp_path / "a.sqlite", doc_ids=["A"])
    assert only_a["doc_count"] == 1


def test_export_preserves_expiry_and_rejects_source_overwrite(tmp_path):
    full = tmp_path / "full.sqlite"
    write_bundle(full, "test", docs(), {}, {"valid_until": "2000-01-01"})
    before = full.read_bytes()
    with pytest.raises(ValueError, match="destination"):
        export_subset(full, "DR Congo", full)
    assert full.read_bytes() == before
    out = tmp_path / "country.sqlite"
    export_subset(full, "DR Congo", out)
    with sqlite3.connect(out) as db:
        assert dict(db.execute("SELECT key,value FROM meta"))["valid_until"] == "2000-01-01"
