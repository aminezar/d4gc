"""Document language from its cleaned text, to correct metadata the CSV gets wrong (e.g. French text labelled en)."""

import re

_STOP = {
    "en": set("the and of to in is for that with are on by this from was were be have has".split()),
    "fr": set("le la les des du est une dans pour avec qui sont pas sur par au aux ont été être".split()),
    "es": set("el los las del es una en para con que por son al se como fue han ser está".split()),
}
_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")


def detect(text: str, fallback: str) -> str:
    """en/fr/es/ar when the text is clear about it, else the fallback (the CSV value)."""
    sample = text[:20000]
    arabic, latin = len(_ARABIC.findall(sample)), len(_LATIN.findall(sample))
    if arabic + latin < 200:
        return fallback
    if arabic > 0.6 * (arabic + latin):
        return "ar"
    if arabic > 0.2 * (arabic + latin):
        return fallback  # mixed, e.g. a bilingual page
    words = re.findall(r"[a-zà-ÿ]+", sample.lower())
    counts = {lang: sum(w in stop for w in words) for lang, stop in _STOP.items()}
    best = max(counts, key=counts.get)
    second = max(v for k, v in counts.items() if k != best)
    if counts[best] >= 20 and counts[best] > 2 * second:
        return best
    return fallback
