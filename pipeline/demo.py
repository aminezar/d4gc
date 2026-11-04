"""Build an original, redistributable demo without the challenge dataset or a model download.

    uv run python -m pipeline.demo
    uv run python -m pipeline.demo --semantic  # optional Qwen download, about 4 GB runtime RAM
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.bundle import BundleDoc, write_bundle
from pipeline.embed import DIMS, embed_passages

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "demo" / "documents.json"


def build_demo(out: Path, semantic: bool = False) -> None:
    docs = []
    for d in json.loads(SAMPLE.read_text(encoding="utf-8"))["documents"]:
        chunks = [d["text"]] if d["text"] else []
        vectors = (embed_passages([d["title"] + "\n\n" + text for text in chunks])
                   if semantic and chunks else np.zeros((len(chunks), DIMS), dtype=np.float32))
        docs.append(BundleDoc(d["id"], d["title"], d["genre"], d["language"], d["year"], d["countries"],
                              "demo/documents.json" if chunks else None, chunks, vectors,
                              [(tag, d["text"]) for tag in d["crisis"]]))
    meta = {"demo": "true", "notice": "Fictional sample documents; not medical or operational guidance."}
    if not semantic:
        meta["model"] = "demo-keyword-only"
    write_bundle(out, "Response Bundle · original demo", docs, {}, meta)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "bundles" / "demo.sqlite")
    p.add_argument("--semantic", action="store_true")
    a = p.parse_args()
    build_demo(a.out, a.semantic)
    print(f"Built {a.out}. Run: uv run python app/server.py --bundle {a.out}")


if __name__ == "__main__":
    main()
