"""Split cleaned text into passages of about 200-300 words, respecting paragraphs."""

import re

TARGET_WORDS = 250
MAX_WORDS = 300
MIN_WORDS = 80  # a shorter tail is merged into the previous passage

_SENTENCE_END = re.compile(r"(?<=[.!?؟。])\s+")


def _split_long(paragraph: str) -> list[str]:
    """Split a paragraph over MAX_WORDS on sentence boundaries, then on words."""
    pieces, current = [], []
    for sentence in _SENTENCE_END.split(paragraph):
        words = sentence.split()
        while len(words) > MAX_WORDS:  # a single run-on "sentence"
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(" ".join(words[:TARGET_WORDS]))
            words = words[TARGET_WORDS:]
        if current and len(current) + len(words) > MAX_WORDS:
            pieces.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str) -> list[str]:
    """Greedy packing of paragraphs into passages; paragraphs are separated by blank lines."""
    paragraphs = []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())
        if p:
            paragraphs.extend(_split_long(p) if len(p.split()) > MAX_WORDS else [p])

    chunks: list[list[str]] = []
    count = 0
    for p in paragraphs:
        n = len(p.split())
        if chunks and count < TARGET_WORDS and count + n <= MAX_WORDS:
            chunks[-1].append(p)
            count += n
        else:
            chunks.append([p])
            count = n
    passages = ["\n\n".join(c) for c in chunks]

    if len(passages) > 1 and len(passages[-1].split()) < MIN_WORDS:
        tail = passages.pop()
        passages[-1] = passages[-1] + "\n\n" + tail
    return passages
