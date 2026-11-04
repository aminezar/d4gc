"""End-to-end API checks using only original, redistributable sample documents."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import server
from pipeline.demo import build_demo
from pipeline.bundle import export_subset


@pytest.fixture
def client(tmp_path, monkeypatch):
    full = tmp_path / "demo.sqlite"
    build_demo(full)
    server.load(full, None)
    monkeypatch.setattr(server, "GAPS", tmp_path / "gaps.jsonl")
    with TestClient(server.app) as c:
        yield c
    server.state.clear()


def test_pages_catalog_and_reading(client):
    assert client.get("/").url.path == "/brussels"
    for route in ("/brussels", "/field", "/static/app.css", "/static/world.json"):
        assert client.get(route).status_code == 200
    catalog = client.get("/api/catalog").json()
    assert catalog["total"] == 8
    assert client.get("/api/catalog?country=DR%20Congo").json()["total"] == 2
    assert client.get("/api/catalog?language=ar").json()["total"] == 1
    assert client.get("/api/quality").json()["without_file"] == 1
    doc = client.get("/api/doc/demo-global-arabic").json()
    assert doc["language"] == "ar" and "التدريب" in doc["text"]
    assert client.get("/api/doc/missing").status_code == 404
    assert client.get("/api/doc/demo-global-backup/file").status_code == 404


def test_search_refusal_validation_and_gap_log(client):
    result = client.post("/api/search", json={"query": "water containers label"}).json()
    assert not result["refused"]
    assert result["results"][0]["doc_id"] == "demo-cod-water"
    for query in ("quantum spacecraft", "the and for"):
        assert client.post("/api/search", json={"query": query}).json()["refused"]
    assert client.get("/api/gaps").json()["total"] == 2
    assert client.post("/api/search", json={"query": " "}).status_code == 400
    assert client.post("/api/search", json={"query": "x" * 2001}).status_code == 422
    server.state["no_gap_log"] = True
    client.post("/api/search", json={"query": "private question"})
    assert client.get("/api/gaps").json() == {"gaps": [], "total": 0}
    assert len(server.GAPS.read_text().splitlines()) == 2


def test_download_is_valid_sqlite_and_does_not_overwrite_source(client, tmp_path):
    source = server.state["path"].read_bytes()
    r = client.post("/api/bundle/DR%20Congo/build", json={"doc_ids": ["demo-cod-water", "demo-global-backup"]})
    assert r.status_code == 200
    assert r.content.startswith(b"SQLite format 3\0")
    assert "dr-congo.sqlite" in r.headers["content-disposition"]
    assert r.headers["x-document-count"] == "2"
    out = tmp_path / "download.sqlite"
    out.write_bytes(r.content)
    with sqlite3.connect(out) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert {r[0] for r in db.execute("SELECT id FROM docs")} == {"demo-cod-water", "demo-global-backup"}
        assert dict(db.execute("SELECT key,value FROM meta"))["demo"] == "true"
    assert server.state["path"].read_bytes() == source
    assert client.post("/api/bundle/DR%20Congo/build", json={"doc_ids": ["demo-haiti-storage"]}).status_code == 400
    assert client.post("/api/bundle/DR%20Congo/build", json={"doc_ids": []}).status_code == 422


def test_concurrent_exports_remain_independent(client, tmp_path):
    selections = [["demo-cod-water"], ["demo-global-backup"]] * 3
    def download(ids):
        return client.post("/api/bundle/DR%20Congo/build", json={"doc_ids": ids})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, selections))
    for i, (r, ids) in enumerate(zip(results, selections)):
        assert r.status_code == 200
        path = tmp_path / f"export-{i}.sqlite"
        path.write_bytes(r.content)
        with sqlite3.connect(path) as db:
            assert [r[0] for r in db.execute("SELECT id FROM docs")] == ids


def test_loading_another_bundle_resets_cached_catalog(client, tmp_path):
    client.get("/api/catalog")  # fill the cache
    country = tmp_path / "cod.sqlite"
    export_subset(server.state["path"], "DR Congo", country, ["demo-cod-water"])
    server.load(country, None)
    assert client.get("/").url.path == "/field"
    assert len(client.get("/api/field/docs").json()["documents"]) == 1
    assert client.get("/api/catalog").status_code == 404


def test_missing_bundle_does_not_create_a_file(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        server.load(missing, None)
    assert not missing.exists()


def test_source_file_boundary_and_html_sandbox(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    original = root / "data" / "original"
    original.mkdir(parents=True)
    html = original / "safe.html"
    html.write_text("<script>fetch('/api/gaps')</script>")
    outside = tmp_path / "outside.html"
    outside.write_text("private")
    (original / "symlink.html").symlink_to(outside)
    with sqlite3.connect(server.state["path"]) as db:
        db.execute("UPDATE docs SET source_file=? WHERE id='demo-cod-water'", ("data/original/safe.html",))
    monkeypatch.setattr(server, "ROOT", root)
    r = client.get("/api/doc/demo-cod-water/file")
    assert r.status_code == 200 and "sandbox" in r.headers["content-security-policy"]
    for source in ("../outside.html", "data/original/symlink.html"):
        with sqlite3.connect(server.state["path"]) as db:
            db.execute("UPDATE docs SET source_file=? WHERE id='demo-cod-water'", (source,))
        assert client.get("/api/doc/demo-cod-water/file").status_code == 404
    assert server.original_file("../outside") is None


def test_malformed_gap_rows_are_ignored(client):
    server.GAPS.write_text('{bad json}\n{}\nnull\n[]\n{"query": 2}\n')
    assert client.get("/api/gaps").json()["total"] == 0
