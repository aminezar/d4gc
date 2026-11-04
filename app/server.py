"""Server: Brussels and field interfaces over one SQLite bundle.

    python app/server.py --bundle bundles/full.sqlite     # Brussels + field
    python app/server.py --bundle bundles/cod.sqlite      # field only

A bundle without meta.country serves both pages; a country bundle defaults to field mode. Everything is local:
no CDN, no network calls. Refused queries are appended to gaps.jsonl in the repo root.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import Field
from starlette.background import BackgroundTask

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
GAPS = ROOT / "gaps.jsonl"
sys.path.insert(0, str(ROOT))

from pipeline.bundle import bundle_candidates, export_subset, split_countries  # noqa: E402

# Keyword-only search (dev bundles): share of the question's content words the best passage must hold.
MIN_COVERAGE = 0.5
# Cosine below which answerable and unanswerable test questions overlap (0.63-0.74, eval/): label such results "borderline".
BORDERLINE = 0.74
STOP = set((
    "the and for are was were has have had how what which who why when where with this that these those from into about does can "
    "les des une pour dans avec est sont que qui par sur aux pas plus quel quelle comment "
    "los las una por con para del que como cual cuál").split())

app = FastAPI(title="Response Bundle")
state: dict = {}


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(state["path"].as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def load(path: Path, mode: str | None) -> None:
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
        required = {"docs", "chunks", "chunks_fts", "tags", "removed"}
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
        if not required <= tables:
            raise ValueError("Not a Response Bundle: missing tables")
    db.close()
    state.clear()
    state.update(
        path=path, meta=meta,
        mode=mode or ("field" if meta.get("country") else "full"),  # country bundles carry meta.country
        semantic="qwen" in meta.get("model", "").lower(),
    )
    if state["semantic"]:
        from pipeline.embed import embed_query, embedding_signature
        if meta.get("embedding_signature") and meta["embedding_signature"] != embedding_signature():
            raise ValueError("Embedding runtime differs from this bundle. Use its locked environment or rebuild it.")
        embed_query("warm up")  # load the model now, not on the first question


# ---------- search: hybrid, with a refusal rule ----------

def tokens(query: str) -> list[str]:
    return [t for t in dict.fromkeys(re.findall(r"[\w']{3,}", query.lower())) if t not in STOP]


def keyword_hits(db: sqlite3.Connection, query: str, limit: int = 50) -> dict[int, float]:
    """chunk_id -> coverage in [0, 1]: share of the query's content words found in the passage."""
    toks = tokens(query)
    if not toks:
        return {}
    match = " OR ".join('"' + t.replace('"', "") + '"' for t in toks)
    try:
        rows = db.execute(
            "SELECT rowid, bm25(chunks_fts) AS s FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?", (match, limit)).fetchall()
    except sqlite3.OperationalError:
        return {}
    out = {}
    for r in rows:
        text = db.execute("SELECT c.text || ' ' || COALESCE(d.title, '') FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE c.chunk_id = ?", (r["rowid"],)).fetchone()[0].lower()
        out[r["rowid"]] = sum(1 for t in toks if t in text) / len(toks)
    return out


def log_gap(query: str) -> None:
    if state.get("no_gap_log"):
        return
    with GAPS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "query": query,
                            "bundle": state["meta"].get("name", "")}, ensure_ascii=False) + "\n")


def search_hybrid(query: str) -> dict:
    """Real bundles: the hybrid search and tuned refusal rule of search/, reduced to one entry per document."""
    from search import search as hybrid
    r = hybrid(str(state["path"]), query)
    if r["refused"]:
        log_gap(query)
        return {"refused": True, "results": []}
    by_id = {d["id"]: d for d in all_docs()}
    results, seen = [], set()
    for p in r["results"]:  # search/ ranks passages; the first one per document is its best
        doc_id = p["doc_id"]
        if doc_id in seen or doc_id not in by_id:
            continue
        seen.add(doc_id)
        d = by_id[doc_id]
        results.append({"doc_id": doc_id, "title": d["title"], "year": d["year"], "language": d["language"], "genre": d["genre"],
                        "countries": d["countries"], "can_open": True, "pdf": is_pdf(doc_id), "passage": p["passage"],
                        "score": p["score"], "borderline": p["score"] < BORDERLINE})
    return {"refused": False, "results": results, "engine": "hybrid"}


def search(query: str) -> dict:
    if state["semantic"]:
        return search_hybrid(query)
    # Keyword-only stand-in (dev bundles have no real embeddings): the best passage must hold at
    # least half of the question's content words, and at least two when the question has several.
    db = connect()
    kw = keyword_hits(db, query)
    need = max(MIN_COVERAGE, 2 / max(len(tokens(query)), 1)) if len(tokens(query)) > 1 else 1.0
    refuse = max(kw.values(), default=0) < min(need, 1.0)
    # A capitalised name the bundle never mentions ("Venezuela") means the question is about somewhere else.
    names = [w.lower() for w in re.findall(r"(?<!^)(?<![.?!] )\b[A-ZÀ-Ý][\w']{2,}", query)]
    for n in names:
        if n not in STOP and not db.execute("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1", ('"' + n + '"',)).fetchone():
            refuse = True
    if refuse:
        db.close()
        log_gap(query)
        return {"refused": True, "results": []}

    # One entry per document, best passage decides the order.
    by_id = {d["id"]: d for d in all_docs()}
    results, seen = [], set()
    for cid in sorted(kw, key=kw.get, reverse=True):
        doc_id, passage = db.execute("SELECT doc_id, text FROM chunks WHERE chunk_id = ?", (cid,)).fetchone()
        if doc_id in seen or doc_id not in by_id:
            continue
        seen.add(doc_id)
        d = by_id[doc_id]
        results.append({"doc_id": doc_id, "title": d["title"], "year": d["year"], "language": d["language"], "genre": d["genre"],
                        "countries": d["countries"], "can_open": True, "pdf": is_pdf(doc_id), "passage": passage})
        if len(results) == 8:
            break
    db.close()
    return {"refused": False, "results": results, "engine": "keyword"}


def doc_file(doc_id: str) -> Path | None:
    """The document's file, chosen by docs.source_file (original html/pdf), else its parsed markdown.

    Only files inside data/original or data/parsed are served, whatever the bundle says (no path traversal)."""
    db = connect()
    row = db.execute("SELECT source_file FROM docs WHERE id = ?", (doc_id,)).fetchone()
    db.close()
    if not row or not row["source_file"]:
        return None
    original = (ROOT / row["source_file"]).resolve()
    parsed = (ROOT / "data" / "parsed" / (Path(row["source_file"]).stem + ".md")).resolve()
    for p, folder in ((original, ROOT / "data" / "original"), (parsed, ROOT / "data" / "parsed")):
        if p.is_relative_to(folder.resolve()) and p.is_file():
            return p
    return None


def is_pdf(doc_id: str) -> bool:
    f = original_file(doc_id)
    return bool(f and f.suffix.lower() == ".pdf")


def original_file(doc_id: str) -> Path | None:
    p = doc_file(doc_id)
    return p if p and p.is_relative_to((ROOT / "data" / "original").resolve()) else None


@app.get("/api/doc/{doc_id}")
def api_doc(doc_id: str):
    d = next((x for x in all_docs() if x["id"] == doc_id), None)
    if not d:
        raise HTTPException(404, "Not in this bundle")
    db = connect()
    parts = [r[0] for r in db.execute("SELECT text FROM chunks WHERE doc_id = ? ORDER BY position", (doc_id,))]
    db.close()
    orig = original_file(doc_id)
    return {"id": doc_id, "title": d["title"], "genre": d["genre"], "year": d["year"], "language": d["language"], "countries": d["countries"],
            "text": "\n\n".join(parts), "pdf": bool(orig and orig.suffix.lower() == ".pdf")}


@app.get("/doc/{doc_id}")
def doc_page(doc_id: str):
    return FileResponse(STATIC / "doc.html")


@app.get("/api/doc/{doc_id}/file")
def api_doc_file(doc_id: str):
    p = doc_file(doc_id)
    if not p:
        raise HTTPException(404, "The document file is not on this laptop")
    if p.suffix == ".md":
        return FileResponse(p, media_type="text/plain; charset=utf-8")
    # Source HTML is untrusted: never give it access to the app's origin or scripts.
    return FileResponse(p, headers={
        "Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'",
        "X-Content-Type-Options": "nosniff",
    })


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.post("/api/search")
def api_search(q: Query):
    q.query = q.query.strip()
    if not q.query:
        raise HTTPException(400, "Empty question")
    return search(q.query)


# ---------- shared ----------

@app.get("/api/meta")
def api_meta():
    return {"meta": state["meta"], "mode": state["mode"], "size_bytes": state["path"].stat().st_size,
            "gap_logging": not state.get("no_gap_log", False)}


def read_gaps() -> list[dict]:
    if not GAPS.exists():
        return []
    out = []
    if state.get("no_gap_log"):
        return []
    with GAPS.open(encoding="utf-8") as f:
        for line in deque(f, maxlen=1000):
            try:
                gap = json.loads(line)
                if isinstance(gap, dict) and all(isinstance(gap.get(k), str) for k in ("query", "ts", "bundle")):
                    out.append(gap)
            except json.JSONDecodeError:
                pass
    return out[::-1]


@app.get("/api/gaps")
def api_gaps():
    gaps = read_gaps()
    counts: dict[str, int] = {}
    for g in gaps:
        counts[g["query"]] = counts.get(g["query"], 0) + 1
    seen, unique = set(), []
    for g in gaps:
        if g["query"] not in seen:
            seen.add(g["query"])
            unique.append({**g, "count": counts[g["query"]]})
    return {"gaps": unique, "total": len(gaps)}


def docs_rows(db, where="", args=()):
    return db.execute(
        "SELECT d.id, d.title, d.genre, d.language, d.year, d.countries, d.source_file, "
        "COALESCE(c.n, 0) AS chunks, t.crisis AS crisis FROM docs d "
        "LEFT JOIN (SELECT doc_id, COUNT(*) n FROM chunks GROUP BY doc_id) c ON c.doc_id = d.id "
        "LEFT JOIN (SELECT doc_id, group_concat(DISTINCT tag) crisis FROM tags WHERE tag != 'unsure' GROUP BY doc_id) t ON t.doc_id = d.id "
        f"{where} ORDER BY d.year DESC, d.id", args).fetchall()


def all_docs() -> list[dict]:
    if "docs" not in state:  # the bundle is read-only, so build the list once
        db = connect()
        state["docs"] = [doc_json(r) for r in docs_rows(db)]
        db.close()
    return state["docs"]


def doc_json(r) -> dict:
    return {"id": r["id"], "title": r["title"], "genre": r["genre"], "language": r["language"], "year": r["year"],
            "countries": split_countries(r["countries"]), "chunks": r["chunks"],
            "crisis": (r["crisis"] or "").split(",") if r["crisis"] else [], "has_file": r["source_file"] is not None}


@app.get("/api/field/docs")
def api_field_docs():
    return {"country": state["meta"].get("country", ""), "documents": [d for d in all_docs() if d["has_file"]]}


# ---------- Brussels ----------

def need_full():
    if state["mode"] != "full":
        raise HTTPException(404, "This bundle serves the field interface only")


@app.get("/api/crises")
def api_crises():
    need_full()
    counts: dict[str, int] = {}
    for d in all_docs():
        for t in d["crisis"]:
            counts[t] = counts.get(t, 0) + 1
    return {"crises": [{"tag": t, "documents": n} for t, n in sorted(counts.items(), key=lambda x: -x[1])]}


@app.get("/api/readiness")
def api_readiness(crisis: str = "", year_from: int = 0, year_to: int = 9999):
    need_full()
    docs = [d for d in all_docs() if d["has_file"]]
    counts: dict[str, int] = {}
    glob = 0
    for d in docs:
        if crisis and crisis not in d["crisis"]:
            continue
        if d["year"] and not (year_from <= int(d["year"]) <= year_to):
            continue
        if not d["countries"]:
            glob += 1
        for c in d["countries"]:
            counts[c] = counts.get(c, 0) + 1
    gap_by_country = {}
    return {"countries": sorted(({"country": c, "documents": n} for c, n in counts.items()), key=lambda x: -x["documents"]),
            "global": glob, "total": len(docs), "gaps": len(read_gaps()), "gap_by_country": gap_by_country}


@app.get("/api/catalog")
def api_catalog(q: str = "", country: str = "", genre: str = "", language: str = "", crisis: str = "", year_from: int = 0, year_to: int = 9999):
    need_full()
    rows = all_docs()
    out = []
    for d in rows:
        if q and q.lower() not in (d["title"] or "").lower() and q.lower() != d["id"].lower():
            continue
        if country and country not in d["countries"]:
            continue
        if genre and d["genre"] != genre or language and d["language"] != language or crisis and crisis not in d["crisis"]:
            continue
        if d["year"] and not (year_from <= int(d["year"]) <= year_to):
            continue
        out.append(d)
    genres = sorted({d["genre"] for d in rows if d["genre"]})
    langs = sorted({d["language"] for d in rows if d["language"]})
    return {"documents": [{**d, "can_open": d["chunks"] > 0, "pdf": is_pdf(d["id"])} for d in out[:400]], "total": len(out), "genres": genres, "languages": langs}


@app.get("/api/quality")
def api_quality():
    need_full()
    docs = all_docs()
    db = connect()
    removed = db.execute("SELECT reason, COUNT(*) n FROM removed GROUP BY reason ORDER BY n DESC").fetchall()
    db.close()
    withfile = [d for d in docs if d["has_file"]]
    return {
        "documents": len(docs), "with_file": len(withfile), "without_file": len(docs) - len(withfile),
        "no_year": sum(1 for d in withfile if not d["year"]),
        "no_country": sum(1 for d in withfile if not d["countries"]),
        "unsure_tags": sum(1 for d in withfile if not d["crisis"]),
        "removed": [{"reason": r["reason"], "documents": r["n"]} for r in removed],
    }


@app.get("/api/bundle/{country}")
def api_bundle(country: str, crisis: str = ""):
    need_full()
    wanted = {c for c in crisis.split(",") if c}
    docs = [d for d in all_docs() if d["has_file"]]
    own_all = [d for d in docs if country in d["countries"]]
    counts: dict[str, int] = {}
    for d in own_all:
        for t in d["crisis"]:
            counts[t] = counts.get(t, 0) + 1
    match = (lambda d: bool(wanted & set(d["crisis"]))) if wanted else (lambda d: True)
    own = [d for d in own_all if match(d)]
    glob = [d for d in docs if not d["countries"] and match(d)]
    db = connect()
    sizes = dict(db.execute("SELECT doc_id, SUM(LENGTH(text) + LENGTH(vector)) FROM chunks GROUP BY doc_id").fetchall())
    db.close()

    def row(d):
        return {"id": d["id"], "title": d["title"], "genre": d["genre"], "language": d["language"], "year": d["year"],
                "countries": d["countries"], "bytes": int(sizes.get(d["id"], 0)), "can_open": d["chunks"] > 0, "pdf": is_pdf(d["id"])}
    return {"country": country, "total": len(own_all), "crises": counts, "own": [row(d) for d in own], "global": [row(d) for d in glob]}


class Build(BaseModel):
    doc_ids: list[str] = Field(min_length=1, max_length=1000)


@app.post("/api/bundle/{country}/build")
def api_build(country: str, body: Build):
    need_full()
    candidates = bundle_candidates(state["path"], country)
    allowed = {d["id"] for group in candidates.values() for d in group}
    if not set(body.doc_ids) <= allowed:
        raise HTTPException(400, "Select documents belonging to this country or the global collection")
    slug = re.sub(r"[^a-z0-9]+", "-", country.lower()).strip("-")
    if not slug or len(slug) > 100:
        raise HTTPException(400, "Invalid country name")
    # Each request owns its output. Never overwrite the loaded bundle or another export.
    tmp = tempfile.TemporaryDirectory(prefix="response-bundle-")
    try:
        info = export_subset(state["path"], country, Path(tmp.name) / f"{slug}.sqlite", body.doc_ids)
        return FileResponse(info["path"], filename=f"{slug}.sqlite", media_type="application/vnd.sqlite3",
                            headers={"X-Document-Count": str(info["doc_count"])},
                            background=BackgroundTask(tmp.cleanup))
    except Exception:
        tmp.cleanup()
        raise


# ---------- pages ----------

@app.get("/")
def index():
    return RedirectResponse("/brussels" if state["mode"] == "full" else "/field")


@app.get("/brussels")
def brussels():
    if state["mode"] != "full":
        return RedirectResponse("/field")
    return FileResponse(STATIC / "brussels.html")


@app.get("/field")
def field():
    return FileResponse(STATIC / "field.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mode", choices=["full", "field"])
    ap.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 only for intentional hosting")
    ap.add_argument("--no-gap-log", action="store_true", help="disable storing and exposing unanswered queries")
    a = ap.parse_args()
    try:
        load(a.bundle, a.mode)
    except (OSError, sqlite3.Error, ValueError) as e:
        ap.error(f"Cannot load bundle: {e}")
    state["no_gap_log"] = a.no_gap_log
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
