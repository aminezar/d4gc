"""Passage and query embeddings. The search side must use embed_query from here so both match."""

import hashlib
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from importlib.metadata import version

import numpy as np

MODEL = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DIMS = 256
QUERY_PROMPT = "Instruct: Given a question, retrieve passages that answer it\nQuery: "
CACHE = Path(__file__).resolve().parent.parent / "bundles" / ".embedding_cache.sqlite"


def embedding_signature() -> str:
    """Invalidate cached vectors when the checkpoint, encoding configuration, or runtime changes."""
    runtime = "|".join(f"{p}={version(p)}" for p in ("sentence-transformers", "transformers", "torch"))
    return hashlib.sha256(f"{MODEL}|{MODEL_REVISION}|{DIMS}|1024|float32|{QUERY_PROMPT}|{runtime}".encode()).hexdigest()


@lru_cache(maxsize=2)
def _model(device: str | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    # The checkpoint defaults to bfloat16, which is slow on CPUs; float32 everywhere keeps
    # passage and query vectors consistent.
    m = SentenceTransformer(MODEL, revision=MODEL_REVISION, device=device, model_kwargs={"dtype": torch.float32})
    m.max_seq_length = 1024
    return m


def _truncate_normalize(v: np.ndarray) -> np.ndarray:
    """Matryoshka cut to DIMS, then L2-normalise, so a dot product is the cosine."""
    v = v[:, :DIMS].astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms != 0)


def _best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_passages(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Embed passages (no prompt). Results are cached on disk by model and text."""
    if not texts:
        return np.empty((0, DIMS), dtype=np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE)
    db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, vector BLOB)")
    signature = embedding_signature()
    keys = [hashlib.sha256(f"{signature}|{t}".encode()).hexdigest() for t in texts]
    found = {}
    for i in range(0, len(keys), 500):
        batch = keys[i:i + 500]
        q = f"SELECT key, vector FROM cache WHERE key IN ({','.join('?' * len(batch))})"
        found.update(db.execute(q, batch).fetchall())

    missing = sorted({k: t for k, t in zip(keys, texts) if k not in found}.items(), key=lambda kt: len(kt[1]))
    step = 512  # commit as we go, so an interrupted run keeps its work
    for i in range(0, len(missing), step):
        part = missing[i:i + step]
        vectors = _truncate_normalize(_model(_best_device()).encode(
            [t for _, t in part], batch_size=batch_size, convert_to_numpy=True,
        ))
        rows = [(k, v.tobytes()) for (k, _), v in zip(part, vectors)]
        db.executemany("INSERT OR REPLACE INTO cache VALUES (?, ?)", rows)
        db.commit()
        found.update(rows)
        print(f"embedded {min(i + step, len(missing))}/{len(missing)}", file=sys.stderr, flush=True)
    db.close()
    return np.stack([np.frombuffer(found[k], dtype=np.float32) for k in keys])


def embed_query(query: str) -> np.ndarray:
    """Embed one query on the CPU with the instruction prompt. Returns a float32 vector of DIMS."""
    return _truncate_normalize(_model("cpu").encode([QUERY_PROMPT + query], convert_to_numpy=True))[0]


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)
