"""Find documents to exclude from the index: navigation pages and duplicates.

Rules (thresholds checked against the 582-document corpus):

1. Exact duplicates: documents whose cleaned text is identical after case
   folding and keeping only word characters (so markdown escapes such as
   "\\[" vs "[" do not matter).
   - A group of NAV_GROUP_MIN (3) or more identical documents is a mirrored
     site navigation page: every member is removed as "navigation page".
     The corpus has exactly one such group, 18 copies of the MSF Access
     "Latest News" page.
   - A smaller group is a duplicate: one member is kept.
2. Near duplicates: 5-word-shingle Jaccard similarity >= NEAR_DUP_JACCARD
   (0.9). In the corpus the only pair above 0.86 is two editions of the
   Rwanda genocide case study (0.937); the next pairs (0.84-0.86) are en/fr/es
   translations of the same reference list, which are not duplicates.

Which member is kept: the most content (word characters, so markup does not
count), ties broken by the smallest id.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

__all__ = ["find_removed", "NAV_GROUP_MIN", "NEAR_DUP_JACCARD", "SHINGLE_WORDS"]

NAV_GROUP_MIN = 3
NEAR_DUP_JACCARD = 0.9
SHINGLE_WORDS = 5

_WORD = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _shingles(words: list[str]) -> set[int]:
    k = SHINGLE_WORDS
    if len(words) < k:
        return {hash(" ".join(words))} if words else set()
    return {hash(" ".join(words[i : i + k])) for i in range(len(words) - k + 1)}


def _keep_order(doc_id: str, size: dict[str, int]) -> tuple[int, str]:
    """Longest content first (markup does not count), then smallest id."""
    return (-size[doc_id], doc_id)


def find_removed(texts: dict[str, str]) -> dict[str, str]:
    """Map doc id -> reason ("navigation page" or "duplicate of <kept_id>").

    `texts` are cleaned texts keyed by doc id. Documents not in the result
    are kept.
    """
    removed: dict[str, str] = {}
    words = {i: _words(t) for i, t in texts.items()}
    size = {i: sum(len(w) + 1 for w in ws) for i, ws in words.items()}

    # 1. Exact duplicates on normalised text.
    groups: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(texts):
        key = hashlib.sha1(" ".join(words[doc_id]).encode("utf-8")).hexdigest()
        groups[key].append(doc_id)
    for members in groups.values():
        if len(members) >= NAV_GROUP_MIN:
            for doc_id in members:
                removed[doc_id] = "navigation page"
        elif len(members) > 1:
            keep, *rest = sorted(members, key=lambda i: _keep_order(i, size))
            for doc_id in rest:
                removed[doc_id] = f"duplicate of {keep}"

    # 2. Near duplicates among what is left. Jaccard >= t needs the smaller
    # shingle set to be at least t times the larger, which prunes most pairs.
    remaining = [i for i in texts if i not in removed and words[i]]
    sh = {i: _shingles(words[i]) for i in remaining}
    order = sorted(remaining, key=lambda i: (len(sh[i]), i))
    parent = {i: i for i in remaining}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_pos, a in enumerate(order):
        sa = sh[a]
        for b in order[a_pos + 1 :]:
            sb = sh[b]
            if len(sa) < NEAR_DUP_JACCARD * len(sb):
                break
            inter = len(sa & sb)
            if inter and inter / (len(sa) + len(sb) - inter) >= NEAR_DUP_JACCARD:
                parent[find(a)] = find(b)

    clusters: dict[str, list[str]] = defaultdict(list)
    for doc_id in remaining:
        clusters[find(doc_id)].append(doc_id)
    for members in clusters.values():
        if len(members) > 1:
            keep, *rest = sorted(members, key=lambda i: _keep_order(i, size))
            for doc_id in rest:
                removed[doc_id] = f"duplicate of {keep}"

    # A duplicate's kept twin may itself be a near-duplicate of a third doc:
    # point every reason at the document that is actually kept.
    for doc_id, reason in list(removed.items()):
        if reason.startswith("duplicate of "):
            target = reason[len("duplicate of ") :]
            seen = {doc_id}
            while target in removed and removed[target].startswith("duplicate of ") and target not in seen:
                seen.add(target)
                target = removed[target][len("duplicate of ") :]
            removed[doc_id] = f"duplicate of {target}"
    return dict(sorted(removed.items()))
