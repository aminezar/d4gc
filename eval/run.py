"""Evaluate search against the keyword baseline, tune the refusal thresholds, write them into bundles.

    uv run python eval/run.py                          # evaluate on bundles/full.sqlite
    uv run python eval/run.py --bundle bundles/cod.sqlite --country "DR Congo"
    uv run python eval/run.py --write                  # write thresholds into the bundle (and eval/thresholds.json for full)

By default, evaluate the thresholds stored in the bundle. Use --tune to compare new thresholds,
or --write to tune and persist them. Tuning uses only the "tune" split.
A question "should answer" when it is answerable or partly answerable, and "should refuse" when unanswerable.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from search import K, PER_DOC, fuse, keyword_candidates, score, thresholds, tokens  # noqa: E402

THRESHOLDS = ROOT / "eval" / "thresholds.json"
LANGS = ("en", "fr", "es", "ar")


def top_docs(ranked_chunks: list[int], chunk_doc: dict[int, str]) -> list[str]:
    """Documents of the top K passages, with at most PER_DOC passages per document (as search() returns)."""
    out, per = [], {}
    for c in ranked_chunks:
        d = chunk_doc[c]
        if per.get(d, 0) >= PER_DOC:
            continue
        per[d] = per.get(d, 0) + 1
        out.append(d)
        if len(out) == K:
            break
    return out


def prepare(bundle: Path, questions: list[dict]) -> list[dict]:
    """Score every question once; thresholds are applied later, so tuning is cheap."""
    con = sqlite3.connect(bundle)
    chunk_doc = dict(con.execute("SELECT chunk_id, doc_id FROM chunks"))
    in_bundle = {r[0] for r in con.execute("SELECT DISTINCT doc_id FROM chunks")}
    rows = []
    for q in questions:
        s = score(bundle, q["query"])
        kw = keyword_candidates(con, tokens(q["query"]))
        rows.append({
            **q,
            "lang": q.get("language") or s["language"],
            "should_answer": q["category"] != "unanswerable",
            "best_cos": s["best_cos"],
            "best_kw": s["best_kw"],
            "ours_docs": top_docs(fuse(s), chunk_doc),
            "kw_docs": top_docs([c for c, _ in kw], chunk_doc),
            "expected": [d for d in q["expected_doc_ids"] if d in in_bundle],
        })
    con.close()
    return rows


def refuses_ours(r, t_vec, t_kw):
    tv = t_vec[r["lang"]] if isinstance(t_vec, dict) else t_vec
    return r["best_cos"] < tv and r["best_kw"] < t_kw


def refuses_kw(r, t_kw):
    return r["best_kw"] < t_kw


def metrics(rows, refused, docs_key):
    answer = [r for r in rows if r["should_answer"]]
    refuse = [r for r in rows if not r["should_answer"]]
    with_expected = [r for r in answer if r["expected"]]
    hits = [r for r in with_expected if not refused(r) and set(r["expected"]) & set(r[docs_key])]
    return {
        "recall@5": len(hits) / len(with_expected) if with_expected else float("nan"),
        "answered": sum(not refused(r) for r in answer) / len(answer) if answer else float("nan"),
        "false_pos": sum(not refused(r) for r in refuse),
        "n_refuse": len(refuse),
        "correct_refusal": sum(refused(r) for r in refuse) / len(refuse) if refuse else float("nan"),
    }


def objective(m):
    """Balanced: answer what can be answered, refuse what cannot. Ties go to fewer false positives."""
    return (m["answered"] + m["correct_refusal"], -m["false_pos"], m["recall@5"])


def _best_vec(rows, grid, t_kw):
    """Best objective for this keyword threshold, and the middle of the tied cosine range (widest margin)."""
    scored = [(objective(metrics(rows, lambda r: refuses_ours(r, tv, t_kw), "ours_docs")), float(tv)) for tv in grid]
    top = max(o for o, _ in scored)
    tied = [tv for o, tv in scored if o == top]
    return top, round(float(np.median(tied)), 3)


def tune(rows, per_language=False):
    cos_grid = np.round(np.arange(0.40, 0.86, 0.005), 3)
    # 1.01 = keywords never rescue a query on their own. Tried most conservative first: ties keep it.
    kw_grid = [1.01, 1.0, 0.9, 0.8, 0.75, 0.67, 0.6, 0.5]
    best = None
    for t_kw in kw_grid:
        obj, t_vec = _best_vec(rows, cos_grid, t_kw)
        if best is None or obj > best[0]:
            best = (obj, t_vec, t_kw)
    _, t_vec, t_kw = best
    if not per_language:
        return t_vec, t_kw
    per = {}
    for lang in LANGS:
        sub = [r for r in rows if r["lang"] == lang]
        # Needs both kinds of question, several of each, to tune on its own.
        if sum(r["should_answer"] for r in sub) < 3 or sum(not r["should_answer"] for r in sub) < 3:
            per[lang] = t_vec
        else:
            per[lang] = _best_vec(sub, cos_grid, t_kw)[1]
    return per, t_kw


def tune_kw(rows):
    grid = [1.0, 0.9, 0.8, 0.75, 0.67, 0.6, 0.5, 0.4, 0.34]  # ties keep the most conservative
    return max(grid, key=lambda t: objective(metrics(rows, lambda r: refuses_kw(r, t), "kw_docs")))


def fmt(m):
    return (f"{m['recall@5']:>9.0%} {m['answered']:>9.0%} {m['false_pos']:>4}/{m['n_refuse']:<3} "
            f"{m['correct_refusal']:>8.0%}")


def write_thresholds(t_vec, t_kw, bundle: Path):
    """Write the thresholds into the evaluated bundle. For full.sqlite also save eval/thresholds.json,
    which the pipeline uses for every bundle it builds (country exports inherit full.sqlite's meta)."""
    meta = {"T_kw": str(t_kw)}
    if isinstance(t_vec, dict):
        meta["T_vec"] = str(t_vec["en"])
        meta |= {f"T_vec_{lang}": str(v) for lang, v in t_vec.items()}
    else:
        meta["T_vec"] = str(t_vec)
    con = sqlite3.connect(bundle)
    con.execute("DELETE FROM meta WHERE key LIKE 'T\\_%' ESCAPE '\\'")
    con.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    con.commit()
    con.close()
    if bundle.name == "full.sqlite":
        THRESHOLDS.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=ROOT / "bundles" / "full.sqlite")
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.json")
    p.add_argument("--country", help="only questions for this country bundle, plus all unanswerable ones")
    p.add_argument("--per-language", action="store_true", help="tune one cosine threshold per language")
    p.add_argument("--tune", action="store_true", help="evaluate newly tuned thresholds without changing the bundle")
    p.add_argument("--write", action="store_true", help="write the tuned thresholds into the evaluated bundle")
    p.add_argument("--details", action="store_true", help="print every question")
    a = p.parse_args()
    if a.per_language and not (a.tune or a.write):
        p.error("--per-language needs --tune or --write")
    if not a.bundle.is_file():
        p.error("Bundle not found. Build the challenge collection first; see README.md.")

    questions = json.loads(a.questions.read_text())
    if a.country:
        questions = [q for q in questions if q["category"] == "unanswerable" or q.get("country") == a.country]
    rows = prepare(a.bundle, questions)
    tune_rows = [r for r in rows if r.get("split") == "tune"]
    test_rows = [r for r in rows if r.get("split") != "tune"]

    if a.tune or a.write:
        t_vec, t_kw = tune(tune_rows, a.per_language)
    else:
        with sqlite3.connect(a.bundle) as con:
            meta = dict(con.execute("SELECT key, value FROM meta"))
        t_vec = {lang: thresholds(meta, lang)[0] for lang in LANGS}
        t_kw = thresholds(meta, "en")[1]
    kw_t = tune_kw(tune_rows)

    print(f"\nBundle {a.bundle.name}: {len(tune_rows)} tuning and {len(test_rows)} held-out questions")
    configuration = "Tuned on the tuning split" if a.tune or a.write else "Stored bundle thresholds (unchanged)"
    print(f"{configuration}: ours T_vec={t_vec} T_kw={t_kw}; keyword baseline tuned T_kw={kw_t}\n")
    print(f"{'Held-out':<22}{'recall@5':>9} {'answered':>9} {'false pos':>8} {'refused ok':>9}")
    print(f"{'Keyword baseline':<22}{fmt(metrics(test_rows, lambda r: refuses_kw(r, kw_t), 'kw_docs'))}")
    print(f"{'Ours (hybrid+refusal)':<22}{fmt(metrics(test_rows, lambda r: refuses_ours(r, t_vec, t_kw), 'ours_docs'))}")
    print("\nrecall@5: answerable/partial questions with an expected document in the top 5 (and not refused).")
    print("false pos: unanswerable questions that got results instead of 'Not in this collection'.")

    print("\nBest cosine by language and category (all questions):")
    for lang in LANGS:
        for should in (True, False):
            v = [r["best_cos"] for r in rows if r["lang"] == lang and r["should_answer"] == should]
            if v:
                print(f"  {lang} {'answer' if should else 'refuse':6} n={len(v):<3} "
                      f"min {min(v):.3f}  median {float(np.median(v)):.3f}  max {max(v):.3f}")

    if a.details:
        print()
        for r in rows:
            ref = refuses_ours(r, t_vec, t_kw)
            ok = ref != r["should_answer"]
            hit = bool(set(r["expected"]) & set(r["ours_docs"]))
            print(f"{'ok ' if ok else 'BAD'} {r['id']:<4} {r['split']:<4} {r['category'][:6]:<6} {r['lang']} "
                  f"cos {r['best_cos']:.3f} kw {r['best_kw']:.2f} {'refused' if ref else 'answered'}"
                  f"{' hit' if hit else ''}  {r['query'][:60]}")

    if a.write:
        meta = write_thresholds(t_vec, t_kw, a.bundle)
        print(f"\nWrote {meta} into {a.bundle.name}" + (f" and {THRESHOLDS.relative_to(ROOT)}" if a.bundle.name == "full.sqlite" else ""))


if __name__ == "__main__":
    main()
