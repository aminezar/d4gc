"""The bundle format (one SQLite file) shared by the pipeline, the search function and the app.

Tables:
    meta       (key, value)
    docs       (id, title, genre, language, year, countries, source_file)
               countries: canonical names joined by "; " (e.g. "DR Congo; Uganda"), "" for global docs.
               Listed-but-missing documents are kept in full.sqlite with source_file NULL and no chunks.
    tags       (doc_id, tag, evidence)   flexible tags; evidence is a verified verbatim quote
    chunks     (chunk_id, doc_id, position, text, vector)   vector: float32 x 256, L2-normalised
    chunks_fts FTS5 over (text, title), rowid = chunk_id
    removed    (id, reason)
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from pipeline.embed import DIMS, MODEL, MODEL_REVISION, embedding_signature, to_blob

# Refusal thresholds: tuned values from eval/thresholds.json (written by eval/run.py --write),
# placeholders until then.
DEFAULT_T_VEC = 0.5
DEFAULT_T_KW = 0.5
THRESHOLDS = Path(__file__).resolve().parent.parent / "eval" / "thresholds.json"
VALID_DAYS = 182

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE docs (
    id TEXT PRIMARY KEY, title TEXT, genre TEXT, language TEXT,
    year INTEGER, countries TEXT NOT NULL DEFAULT '', source_file TEXT
);
CREATE TABLE tags (doc_id TEXT NOT NULL REFERENCES docs(id), tag TEXT NOT NULL, evidence TEXT);
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES docs(id),
    position INTEGER NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, title, tokenize = 'unicode61 remove_diacritics 2');
CREATE TABLE removed (id TEXT PRIMARY KEY, reason TEXT NOT NULL);
CREATE INDEX chunks_doc ON chunks(doc_id);
CREATE INDEX tags_doc ON tags(doc_id);
CREATE INDEX tags_tag ON tags(tag);
"""


@dataclass
class BundleDoc:
    id: str
    title: str | None
    genre: str
    language: str
    year: int | None
    countries: list[str]
    source_file: str | None
    chunks: list[str] = field(default_factory=list)
    vectors: np.ndarray | None = None  # shape (len(chunks), DIMS)
    tags: list[tuple[str, str]] = field(default_factory=list)  # (tag, evidence)


COUNTRY_SEP = "; "


def join_countries(countries: list[str]) -> str:
    return COUNTRY_SEP.join(countries)


def split_countries(value: str | None) -> list[str]:
    return [c.strip() for c in (value or "").split(";") if c.strip()]


def _base_meta(name: str) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    return {
        "name": name,
        "created": now.isoformat(timespec="seconds"),
        "valid_until": (now.date() + timedelta(days=VALID_DAYS)).isoformat(),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "embedding_signature": embedding_signature(),
        "dims": str(DIMS),
        "T_vec": str(DEFAULT_T_VEC),
        "T_kw": str(DEFAULT_T_KW),
    } | (json.loads(THRESHOLDS.read_text()) if THRESHOLDS.exists() else {})


def write_bundle(path: Path, name: str, docs: list[BundleDoc], removed: dict[str, str],
                 extra_meta: dict[str, str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)

    chunk_id = 0
    for d in docs:
        db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?)", (
            d.id, d.title, d.genre, d.language, d.year,
            join_countries(d.countries), d.source_file,
        ))
        db.executemany("INSERT INTO tags VALUES (?,?,?)", [(d.id, t, e) for t, e in d.tags])
        if d.chunks:
            assert d.vectors is not None and d.vectors.shape == (len(d.chunks), DIMS), d.id
        for pos, text in enumerate(d.chunks):
            chunk_id += 1
            db.execute("INSERT INTO chunks VALUES (?,?,?,?,?)", (chunk_id, d.id, pos, text, to_blob(d.vectors[pos])))
            db.execute("INSERT INTO chunks_fts(rowid, text, title) VALUES (?,?,?)", (chunk_id, text, d.title or ""))
    db.executemany("INSERT INTO removed VALUES (?,?)", sorted(removed.items()))

    meta = _base_meta(name) | (extra_meta or {})
    meta["doc_count"] = str(sum(1 for d in docs if d.chunks))
    meta["chunk_count"] = str(chunk_id)
    db.executemany("INSERT INTO meta VALUES (?,?)", sorted(meta.items()))
    db.commit()
    db.execute("VACUUM")
    db.close()
    tmp.replace(path)


def bundle_candidates(full: Path, country: str) -> dict[str, list[dict]]:
    """Documents a country bundle would contain: about the country, and global (no country).

    The Brussels page shows both groups and lets staff untick documents before export.
    """
    db = sqlite3.connect(full)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS n_chunks "
        "FROM docs d WHERE d.source_file IS NOT NULL ORDER BY d.year DESC, d.id"
    ).fetchall()
    db.close()
    out = {"country": [], "global": []}
    for r in rows:
        countries = split_countries(r["countries"])
        if country in countries:
            out["country"].append(dict(r))
        elif not countries:
            out["global"].append(dict(r))
    return out


def export_subset(full: Path, country: str, out: Path, doc_ids: list[str] | None = None) -> dict:
    """Copy a country's documents (plus global ones) from the full bundle, without re-embedding.

    doc_ids overrides the default selection (e.g. after staff untick documents).
    Returns {"path", "doc_count", "chunk_count", "size_bytes"}.
    """
    if Path(full).resolve() == Path(out).resolve():
        raise ValueError("Export destination must differ from the source bundle")
    if not Path(full).is_file():
        raise FileNotFoundError(full)
    if doc_ids is None:
        c = bundle_candidates(full, country)
        doc_ids = [d["id"] for d in c["country"] + c["global"]]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)

    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)
    db.execute("ATTACH DATABASE ? AS src", (str(full),))
    db.execute("CREATE TEMP TABLE pick (id TEXT PRIMARY KEY)")
    db.executemany("INSERT OR IGNORE INTO pick VALUES (?)", [(i,) for i in doc_ids])
    db.execute("INSERT INTO docs SELECT * FROM src.docs WHERE id IN (SELECT id FROM pick)")
    db.execute("INSERT INTO tags SELECT * FROM src.tags WHERE doc_id IN (SELECT id FROM pick)")
    db.execute("INSERT INTO chunks SELECT * FROM src.chunks WHERE doc_id IN (SELECT id FROM pick) ORDER BY chunk_id")
    db.execute("INSERT INTO chunks_fts(rowid, text, title) "
               "SELECT c.chunk_id, c.text, COALESCE(d.title, '') FROM chunks c JOIN docs d ON d.id = c.doc_id")
    src_meta = dict(db.execute("SELECT key, value FROM src.meta").fetchall())
    db.commit()
    db.execute("DETACH DATABASE src")

    doc_count = db.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]
    chunk_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    meta = _base_meta(f"{country} bundle")
    # Keep the tuned thresholds and the model of the full bundle.
    for k, v in src_meta.items():
        if k in ("model", "dims", "model_revision", "embedding_signature", "demo", "notice") or k.startswith("T_"):
            meta[k] = v
    meta |= {"country": country, "source": src_meta.get("name", ""),
             "doc_count": str(doc_count), "chunk_count": str(chunk_count)}
    # Repackaging old documents must not make them appear fresh for another six months.
    if src_meta.get("valid_until"):
        meta["valid_until"] = src_meta["valid_until"]
    db.executemany("INSERT INTO meta VALUES (?,?)", sorted(meta.items()))
    db.commit()
    db.execute("VACUUM")
    db.close()
    tmp.replace(out)
    return {"path": str(out), "doc_count": doc_count, "chunk_count": chunk_count, "size_bytes": out.stat().st_size}
