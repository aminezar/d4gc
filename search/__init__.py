"""Hybrid search with a refusal rule over one bundle.

    search(bundle_path, query) -> {"refused": bool, "results": [...], "engine": "hybrid", ...}

1. The query is embedded with pipeline.embed.embed_query (same model and prompt as the passages).
2. Every passage gets a cosine score; FTS5 finds keyword candidates, each scored by keyword
   coverage: the share of the query's content words found in the passage (0..1).
3. Refuse when best cosine < T_vec AND best coverage < T_kw. Thresholds come from the bundle's
   meta; a per-language T_vec_<lang> (en, fr, es, ar) overrides T_vec when present.
4. Otherwise rank by reciprocal rank fusion of the two rankings and return the top 5 passages,
   at most 2 per document.
"""

import os
import re
import sqlite3
import unicodedata
from functools import lru_cache

import numpy as np

K = 5
PER_DOC = 2
CANDIDATES = 100
RRF_K = 60

STOPWORDS = set("""
a an and are as at be by can do does for from how i in is it its me my of on or our should that the their
there these this to us was we what when where which who why will with you your about into than then them
le la les un une des du de d l et ou en dans pour par sur au aux avec est sont que qui quoi quel quelle
quels quelles comment ce cette ces il elle ils elles on nous vous je mon ma mes son sa ses leur leurs
pas ne plus faut doit peut y a
el los las unos unas del al y o en para por con es son que qué cual cuál cuales como cómo se su sus
lo nos hay debe puede este esta estos estas mi mis
في من على إلى الى عن ما ماذا كيف هل هو هي هذا هذه ذلك التي الذي أو او و مع كل لا
""".split())

_ARABIC = re.compile(r"[؀-ۿ]")
_FR = {"le", "la", "les", "des", "du", "est", "une", "comment", "quel", "quelle", "pour", "dans", "faut", "doit"}
_ES = {"el", "los", "las", "del", "es", "una", "cómo", "como", "qué", "que", "para", "hay", "debe", "cuál"}


def _fold(text: str) -> str:
    """Casefold and strip diacritics, like the FTS5 tokenizer (remove_diacritics 2)."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokens(query: str) -> list[str]:
    """Distinct content words of a query, folded."""
    seen = []
    for t in re.findall(r"\w+", _fold(query)):
        if len(t) > 1 and t not in STOPWORDS and not t.isdigit() and t not in seen:
            seen.append(t)
    return seen


def query_language(query: str) -> str:
    if _ARABIC.search(query):
        return "ar"
    words = set(re.findall(r"\w+", query.lower()))
    fr, es = len(words & _FR), len(words & _ES)
    if fr > es and fr:
        return "fr"
    if es > fr and es:
        return "es"
    return "en"


def coverage(text: str, toks: list[str]) -> float:
    if not toks:
        return 0.0
    words = set(re.findall(r"\w+", _fold(text)))
    # A query word counts when the passage has it, or a longer form of it (plural, suffix).
    hit = sum(1 for t in toks if t in words or any(w.startswith(t) for w in words if len(t) >= 4))
    return hit / len(toks)


@lru_cache(maxsize=4)
def _matrix(bundle_path: str, mtime: float):
    con = sqlite3.connect(f"file:{bundle_path}?mode=ro", uri=True)
    rows = con.execute("SELECT chunk_id, vector FROM chunks ORDER BY chunk_id").fetchall()
    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    con.close()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 256), np.float32)
    return ids, mat, meta


def keyword_candidates(con, toks: list[str], limit: int = CANDIDATES) -> list[tuple[int, float]]:
    """(chunk_id, coverage) for the best FTS5 (bm25) matches, in bm25 order."""
    if not toks:
        return []
    match = " OR ".join(f'"{t}"*' if len(t) >= 4 else f'"{t}"' for t in toks)
    rows = con.execute(
        "SELECT f.rowid, c.text, d.title FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid "
        "JOIN docs d ON d.id = c.doc_id WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (match, limit),
    ).fetchall()
    return [(r[0], coverage(f"{r[2] or ''} {r[1]}", toks)) for r in rows]


def thresholds(meta: dict, lang: str) -> tuple[float, float]:
    t_vec = float(meta.get(f"T_vec_{lang}", meta.get("T_vec", 0.5)))
    return t_vec, float(meta.get("T_kw", 0.5))


def _passages(con, ranked: list[int], scores: dict[int, float], k: int = K) -> list[dict]:
    out, per_doc = [], {}
    for cid in ranked:
        r = con.execute(
            "SELECT c.chunk_id, c.doc_id, c.text, d.title, d.year, d.language, d.genre, d.countries, d.source_file "
            "FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE c.chunk_id = ?", (int(cid),)).fetchone()
        if per_doc.get(r[1], 0) >= PER_DOC:
            continue
        per_doc[r[1]] = per_doc.get(r[1], 0) + 1
        out.append({"chunk_id": r[0], "doc_id": r[1], "passage": r[2], "title": r[3], "year": r[4],
                    "language": r[5], "genre": r[6], "countries": r[7], "source_file": r[8],
                    "score": round(float(scores.get(cid, 0.0)), 3)})
        if len(out) == k:
            break
    return out


def score(bundle_path: str, query: str) -> dict:
    """Everything the refusal rule and the ranking need, without applying thresholds (used by eval)."""
    from pipeline.embed import embed_query, embedding_signature

    bundle_path = str(bundle_path)
    ids, mat, meta = _matrix(bundle_path, os.path.getmtime(bundle_path))
    if meta.get("embedding_signature") and meta["embedding_signature"] != embedding_signature():
        raise ValueError("Embedding runtime differs from the bundle. Use its locked environment or rebuild the bundle.")
    cos = mat @ embed_query(query) if len(ids) else np.zeros(0, np.float32)
    order = np.argsort(-cos)[:CANDIDATES]
    con = sqlite3.connect(f"file:{bundle_path}?mode=ro", uri=True)
    kw = keyword_candidates(con, tokens(query))
    con.close()
    return {
        "meta": meta,
        "language": query_language(query),
        "vec_ranked": [(int(ids[i]), float(cos[i])) for i in order],
        "kw_ranked": kw,
        "best_cos": float(cos[order[0]]) if len(order) else 0.0,
        "best_kw": max((c for _, c in kw), default=0.0),
    }


def fuse(s: dict) -> list[int]:
    fused: dict[int, float] = {}
    for rank, (cid, _) in enumerate(s["vec_ranked"]):
        fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + rank)
    for rank, (cid, _) in enumerate(s["kw_ranked"]):
        fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + rank)
    return sorted(fused, key=lambda c: -fused[c])


def refuse(s: dict, t_vec: float | None = None, t_kw: float | None = None) -> bool:
    dv, dk = thresholds(s["meta"], s["language"])
    t_vec = dv if t_vec is None else t_vec
    t_kw = dk if t_kw is None else t_kw
    return s["best_cos"] < t_vec and s["best_kw"] < t_kw


def search(bundle_path, query: str) -> dict:
    s = score(bundle_path, query)
    info = {"engine": "hybrid", "best_cosine": round(s["best_cos"], 3), "best_keyword": round(s["best_kw"], 3),
            "language": s["language"]}
    if not query.strip() or not s["vec_ranked"] and not s["kw_ranked"] or refuse(s):
        return {"refused": True, "results": [], **info}
    cos_by_id = dict(s["vec_ranked"])
    con = sqlite3.connect(f"file:{bundle_path}?mode=ro", uri=True)
    results = _passages(con, fuse(s), cos_by_id)
    con.close()
    return {"refused": False, "results": results, **info}


def keyword_search(bundle_path, query: str, t_kw: float = 0.5) -> dict:
    """Baseline: FTS5 bm25 only, refusing when no passage covers at least t_kw of the query words."""
    con = sqlite3.connect(f"file:{bundle_path}?mode=ro", uri=True)
    kw = keyword_candidates(con, tokens(query))
    best = max((c for _, c in kw), default=0.0)
    if best < t_kw:
        con.close()
        return {"refused": True, "results": [], "engine": "keyword", "best_keyword": round(best, 3)}
    results = _passages(con, [c for c, _ in kw], dict(kw))
    con.close()
    return {"refused": False, "results": results, "engine": "keyword", "best_keyword": round(best, 3)}
