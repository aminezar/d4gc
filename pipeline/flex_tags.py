"""Flexible tags: crisis / emergency type per document.

Produced once, off-site, by a cloud LLM and frozen to ``data/annotations/flex_tags.json``.
Every tag carries a verbatim evidence quote; ``verify`` checks that quote against
the document text and ``load_tags`` turns anything unverifiable into "unsure".

Generate (resumable, concurrent):

    uv run python -m pipeline.flex_tags generate [--model opus] [--workers 8] [--limit N] [--ids a,b]

Backend: Anthropic SDK when ANTHROPIC_API_KEY is set, otherwise the Claude Code
CLI in headless mode (``claude -p``).

Report on the frozen file:

    uv run python -m pipeline.flex_tags stats
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = ROOT / "data" / "parsed"
OUT_PATH = ROOT / "data" / "annotations" / "flex_tags.json"

VOCAB: dict[str, str] = {
    "cholera": "cholera outbreaks, prevention, treatment (CTC/ORP)",
    "ebola": "Ebola / filovirus disease (incl. Marburg)",
    "measles": "measles outbreaks, measles vaccination campaigns",
    "malaria": "malaria diagnosis, treatment, prevention",
    "tuberculosis": "TB, MDR/XDR-TB",
    "hiv": "HIV/AIDS, antiretroviral therapy",
    "malnutrition": "acute/chronic malnutrition, nutrition crises, therapeutic feeding",
    "covid-19": "COVID-19 / SARS-CoV-2",
    "other-epidemic": "other outbreaks, epidemic-prone or neglected infectious diseases (meningitis, yellow fever, dengue, hepatitis, kala-azar, sleeping sickness, diphtheria, mpox, antimicrobial resistance, ...)",
    "conflict": "armed conflict, war, violence against civilians/health care, war surgery",
    "displacement": "refugees, IDPs, migrants, camps, people on the move",
    "natural-disaster": "earthquakes, floods, cyclones, tsunamis, drought and other natural hazards",
    "water-sanitation": "water supply, sanitation, hygiene (WASH), vector control as environmental health",
    "sexual-violence": "sexual and gender-based violence, rape, care for survivors",
    "maternal-health": "maternal, obstetric, reproductive health, safe abortion, neonatal care",
    "mental-health": "mental health, psychosocial support, trauma",
    "access-to-medicines": "drug/vaccine access, pricing, patents, R&D for neglected diseases, supply of medicines",
}
UNSURE = "unsure"
MAX_WORDS = 3000

SYSTEM_PROMPT = """You tag humanitarian / medical documents from Médecins Sans Frontières (MSF) with crisis or emergency types.

Allowed tags (use these exact strings, nothing else):
{vocab}

Rules:
- Tag only what the document is SUBSTANTIALLY about: a main topic, or a major section of it. Ask: would an MSF staff member searching for this crisis type want this document? A passing mention, a single sentence, a background statistic or a secondary outcome does not count. Most documents get 1 to 3 tags.
- Ignore reference lists, keywords lines, journal headers/footers and other articles that happen to share the page; tag the document itself.
- access-to-medicines means barriers to obtaining drugs, vaccines or diagnostics (price, patents, supply, registration, R&D gaps), not merely a cost-effectiveness figure.
- A document may get zero tags (e.g. general management, logistics, unrelated clinical topics). Return [] if none applies.
- For every tag give one piece of evidence: a quote of 5 to 30 consecutive words COPIED EXACTLY, character for character, from the document text.
  - Do not paraphrase, translate, summarise, reorder, or fix typos, spelling, spacing, or broken words. Keep the original language.
  - Do not join text across separate sections, and do not add or remove words or "...".
  - Prefer a single sentence from the main body that clearly shows the tag applies. If the text has broken words or odd spacing (e.g. "fi nding"), copy them as they are.
- Output ONLY a JSON array, no prose, no markdown fences:
[{{"tag": "<tag>", "evidence": "<exact quote>"}}]"""


def _system_prompt() -> str:
    vocab = "\n".join(f"- {t}: {d}" for t, d in VOCAB.items())
    return SYSTEM_PROMPT.format(vocab=vocab)


# ---------------------------------------------------------------------------
# Normalisation and verification
# ---------------------------------------------------------------------------

_QUOTES = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
        "´": "'", "`": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
        "«": '"', "»": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
        "―": "-", "−": "-", "﹘": "-", "﹣": "-", "－": "-",
        "­": None, "​": None, "‌": None, "‍": None, "﻿": None,
        " ": " ",
    }
)
_HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\r?\n[ \t]*(\w)")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC, unify quotes/dashes, rejoin hyphenated line breaks, casefold, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _WS.sub(" ", text.casefold()).strip()


def _alnum_key(text: str) -> str:
    """Letters and digits only: survives PDF artefacts ("Eﬀ ective", "multidrug-\\nresistant")
    and markdown stripping by a cleaner."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if ch.isalnum())


MIN_EVIDENCE_CHARS = 15  # letters+digits; guards against trivial quotes matching anywhere


def verify(quote: str, text: str) -> bool:
    """True when ``quote`` occurs in ``text`` after normalising both the same way.

    First tries the whitespace-collapsed form, then a letters-and-digits-only form
    (handles ligature spacing, hyphenation and markdown removed by a cleaner).
    """
    if not quote or not text:
        return False
    key = _alnum_key(quote)
    if len(key) < MIN_EVIDENCE_CHARS:
        return False
    if normalize(quote) in normalize(text):
        return True
    return key in _alnum_key(text)


# ---------------------------------------------------------------------------
# Loading for the bundle builder
# ---------------------------------------------------------------------------


def load_tags(
    texts: dict[str, str], path: Path | str = OUT_PATH
) -> dict[str, list[tuple[str, str]]]:
    """Verified (tag, evidence) per doc id, for the docs in ``texts``.

    A tag outside the vocabulary or whose evidence does not verify becomes
    ("unsure", evidence) -- or ("unsure", tag) when there is no evidence.
    Docs missing from ``texts`` are skipped; docs without tags map to [].
    """
    if not Path(path).exists():
        return {doc_id: [] for doc_id in texts}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tags_by_doc = data.get("tags", data) if isinstance(data, dict) else {}
    out: dict[str, list[tuple[str, str]]] = {}
    for doc_id, text in texts.items():
        result: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in tags_by_doc.get(doc_id) or []:
            if not isinstance(item, dict):
                continue
            tag = str(item.get("tag") or "").strip().lower()
            evidence = str(item.get("evidence") or "").strip()
            if tag in VOCAB and verify(evidence, text):
                pair = (tag, evidence)
            else:
                pair = (UNSURE, evidence or tag)
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
        out[doc_id] = result
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def prepare(text: str, max_words: int = MAX_WORDS) -> str:
    """NFKC-normalised, whitespace-collapsed, capped input for the LLM (title is at the top)."""
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1-\2", text)  # keep the hyphen; verify() tolerates either form
    words = text.split()
    return " ".join(words[:max_words])


def _parse_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in response: {raw[:200]!r}")
    arr = json.loads(raw[start : end + 1])
    if not isinstance(arr, list):
        raise ValueError("response is not a list")
    out = []
    for item in arr:
        if isinstance(item, dict) and "tag" in item:
            out.append({"tag": str(item["tag"]).strip(), "evidence": str(item.get("evidence", "")).strip()})
    return out


def _call_cli(doc: str, model: str, timeout: int, system: str | None = None) -> tuple[str, str]:
    cmd = [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--no-session-persistence",
        "--system-prompt", system or _system_prompt(),
    ]
    # Run outside the repo so no project CLAUDE.md is picked up.
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            cmd, input=f"<document>\n{doc}\n</document>", capture_output=True,
            text=True, timeout=timeout, cwd=cwd,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit {proc.returncode}: {(proc.stderr or proc.stdout)[:300]}")
    env = json.loads(proc.stdout)
    if env.get("is_error"):
        raise RuntimeError(f"claude error: {str(env.get('result'))[:300]}")
    used = ",".join((env.get("modelUsage") or {}).keys()) or model
    return env.get("result", ""), used


_sdk_client = None


def _call_sdk(doc: str, model: str, timeout: int, system: str | None = None) -> tuple[str, str]:
    global _sdk_client
    import anthropic  # optional dependency, only needed with an API key

    if _sdk_client is None:
        _sdk_client = anthropic.Anthropic(timeout=timeout)
    model_id = os.environ.get("ANTHROPIC_MODEL", model)
    if model_id in {"opus", "sonnet"}:
        raise ValueError("The SDK needs a full supported model ID: set ANTHROPIC_MODEL or pass --model <model-id>.")
    msg = _sdk_client.messages.create(
        model=model_id,
        max_tokens=2000,
        system=system or _system_prompt(),
        messages=[{"role": "user", "content": f"<document>\n{doc}\n</document>"}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return text, msg.model


def tag_document(text: str, model: str, backend: str, retries: int = 3, timeout: int = 300) -> tuple[list[dict], str]:
    doc = prepare(text)
    call = _call_sdk if backend == "sdk" else _call_cli
    last: Exception | None = None
    for attempt in range(retries):
        try:
            raw, used = call(doc, model, timeout)
            return _parse_json_array(raw), used
        except Exception as e:  # noqa: BLE001 - retry anything, record the last error
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last))


def _load_out(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("tags", {})
        data.setdefault("failures", {})
        data.setdefault("meta", {})
        return data
    return {"meta": {}, "tags": {}, "failures": {}}


def _save_out(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["tags"] = dict(sorted(data["tags"].items()))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def generate(model: str, workers: int, limit: int | None, ids: list[str] | None, out: Path, retries: int) -> None:
    backend = "sdk" if os.environ.get("ANTHROPIC_API_KEY") else "cli"
    data = _load_out(out)
    all_ids = sorted(p.stem for p in PARSED_DIR.glob("*.md"))
    todo = [i for i in (ids or all_ids) if i not in data["tags"]]
    if limit:
        todo = todo[:limit]
    data["meta"].update({"backend": backend, "model_requested": model, "vocabulary": list(VOCAB)})
    print(f"backend={backend} model={model} todo={len(todo)} done={len(data['tags'])}", file=sys.stderr)

    lock = threading.Lock()
    t0 = time.time()
    done = 0

    def work(doc_id: str):
        text = (PARSED_DIR / f"{doc_id}.md").read_text(encoding="utf-8", errors="replace")
        return doc_id, *tag_document(text, model, backend, retries=retries)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, i): i for i in todo}
        for fut in as_completed(futures):
            doc_id = futures[fut]
            with lock:
                done += 1
                try:
                    _, tags, used = fut.result()
                    data["tags"][doc_id] = tags
                    data["failures"].pop(doc_id, None)
                    data["meta"]["model_used"] = used
                    msg = f"{len(tags)} tags"
                except Exception as e:  # noqa: BLE001
                    data["failures"][doc_id] = str(e)[:500]
                    msg = f"FAILED {str(e)[:120]}"
                _save_out(data, out)
                print(f"[{done}/{len(todo)} {time.time() - t0:.0f}s] {doc_id}: {msg}", file=sys.stderr, flush=True)
    data["meta"]["last_run_seconds"] = round(time.time() - t0, 1)
    _save_out(data, out)


def stats(out: Path) -> None:
    data = _load_out(out)
    texts = {p.stem: p.read_text(encoding="utf-8", errors="replace") for p in PARSED_DIR.glob("*.md")}
    loaded = load_tags({k: v for k, v in texts.items() if k in data["tags"]}, out)
    raw_n = sum(len(v) for v in data["tags"].values())
    unsure = [(d, e) for d, v in loaded.items() for t, e in v if t == UNSURE]
    per_tag: dict[str, int] = {}
    for v in loaded.values():
        for t in {t for t, _ in v}:
            per_tag[t] = per_tag.get(t, 0) + 1
    print(f"docs with result: {len(data['tags'])}/{len(texts)}; failures: {len(data['failures'])}")
    print(f"raw tags: {raw_n}; verified: {raw_n - len(unsure)}; unsure: {len(unsure)}"
          f" ({100 * (raw_n - len(unsure)) / max(raw_n, 1):.1f}% pass)")
    print(f"docs with no tag: {sum(1 for v in loaded.values() if not v)}")
    for t, n in sorted(per_tag.items(), key=lambda x: -x[1]):
        print(f"  {t:22s} {n}")
    for d, e in unsure[:20]:
        print(f"  UNSURE {d}: {e[:140]}")
    print("meta:", json.dumps(data["meta"], ensure_ascii=False)[:400])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m pipeline.flex_tags")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="tag documents with the cloud LLM (resumable)")
    g.add_argument("--model", default="opus", help="model alias or id (default: opus)")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int, default=None, help="only tag the first N pending docs")
    g.add_argument("--ids", default=None, help="comma-separated doc ids")
    g.add_argument("--retries", type=int, default=3)
    g.add_argument("--out", type=Path, default=OUT_PATH)
    s = sub.add_parser("stats", help="verification report on the frozen JSON")
    s.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        ids = args.ids.split(",") if args.ids else None
        generate(args.model, args.workers, args.limit, ids, args.out, args.retries)
    else:
        stats(args.out)


if __name__ == "__main__":
    main()
