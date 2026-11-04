"""Core tags without an LLM: countries a document is about, and its year.

countries(title, text)          -> canonical country names, most relevant first
year(raw, genre, source_file)   -> publication year, or None when unsure
iso3(name)                      -> ISO 3166 alpha-3 code for a canonical name

Country names come from countries.json (English, French, Spanish and Arabic
names, plus a few unambiguous regions and cities such as Darfur or Gaza).
Both the aliases and the text are folded the same way before matching
(NFKC, accents and Arabic diacritics removed, curly quotes straightened).
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

COUNTRIES_JSON = Path(__file__).with_name("countries.json")

# --------------------------------------------------------------------------
# Text folding
# --------------------------------------------------------------------------

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'", "\u00b4": "'",
                         "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-",
                         "\u0649": "\u064a", "\u0629": "\u0647",  # Arabic ى -> ي, ة -> ه
                         "\u0640": None, "\u00a0": " ",  # tatweel, no-break space
                         "\u200b": None, "\u200c": None, "\u200d": None,
                         "\u200e": None, "\u200f": None, "\u00ad": None})


def fold(s: str) -> str:
    """Same-length-agnostic normalisation used for both aliases and text.

    NFKC (ligatures), straight quotes, Arabic-Indic digits to ASCII, accents
    and Arabic diacritics/hamza seats removed, Arabic ى/ة unified.
    """
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES).translate(_ARABIC_DIGITS)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", s)


_SPACES = re.compile(r"[ \t\f\v]+")


def _prep(text: str) -> str:
    return _SPACES.sub(" ", fold(text or ""))


# --------------------------------------------------------------------------
# Country gazetteer
# --------------------------------------------------------------------------

_ARABIC = re.compile(r"[\u0600-\u06ff]")

# Phrases that contain a country alias but do not refer to the country.
_BLOCKERS = [
    r"guinea[- ]?pigs?", r"guinea[- ]?worms?", r"ver de guinee",
    r"atlanta,? (?:ga|georgia)", r"georgia,? (?:usa|u\.s\.a?\.?|united states)",
    r"(?:university|state university|medical college) of georgia", r"georgia (?:state|institute|tech)",
    r"new mexico", r"new south wales", r"american samoa", r"new guinea pig",
    r"(?:latin|south|central|north|native|african|pan|meso)[- ]americ\w*",
    r"america latina", r"amerique latine", r"amerique du (?:sud|nord)",
    r"امريكا (?:اللاتينيه|الجنوبيه|الشماليه|الوسطي)",
    r"indian ocean", r"ocean indien", r"oceano indico", r"american indians?",
    r"(?:river|fleuve|rio) niger", r"niger river", r"aspergillus niger",
    r"(?:msf|medecins sans frontieres|medicos sin fronteras|doctors without borders|اطباء بلا حدود)"
    r"[ \-/(,]{0,3}(?:belgium|belgique|belgica|france|francia|spain|espagne|espana|switzerland|suisse|suiza"
    r"|holland|hollande|holanda|netherlands|luxembourg|germany|allemagne|uk|usa|japan|canada|australia"
    r"|brazil|brasil|italy|italia|sweden|norway|denmark|austria|ireland|south africa|east africa"
    r"|west africa|southern africa|latin america|india|hong kong|argentina|mexico|greece)",
    r"(?:paris|brussels|bruxelles|geneva|geneve|ginebra|amsterdam|barcelona|london|new york|berlin|luxembourg),"
    r" (?:france|belgium|belgique|switzerland|suisse|suiza|the netherlands|netherlands|spain|espana|uk|united kingdom|usa|germany|luxembourg)",
    r"(?:american|british|canadian|australian|french|german|swiss|dutch|indian|chinese|japanese|south african)"
    r" (?:national|medical|journal|heart|college|society|academy|association|thoracic|psychiatric|red cross"
    r"|institute|board|council|pharmacopoeia|formulary|agency|public health|dental|diabetes|cancer)",
    r"spanish flu", r"swiss tph", r"(?:republic|state) of (?:georgia|jordan) (?=university)",
    r"turkey (?:tail|meat|sandwich)", r"china (?:clay|ware)", r"us\$",
    r"ebola (?:river|virus)? ?\(?zaire", r"zaire (?:ebola\w*|species|strain|subtype|ebov)",
    r"sudan (?:ebola\w*|virus|species|strain|subtype)", r"ebola[- ]sudan",
    r"medecins du monde", r"west nile",
]
_BLOCK_RE = re.compile("|".join(f"(?:{b})" for b in _BLOCKERS), re.IGNORECASE)

# Names that are also common first names or surnames.
_PERSON_NAMES = {"Jordan", "Chad", "Georgia", "Kenya", "India"}
_FUNCTION_WORDS = {
    "In", "The", "And", "Of", "From", "To", "For", "With", "On", "At", "By", "As",
    "En", "Au", "Le", "La", "Les", "Du", "De", "Des", "Et", "El", "Los", "Las", "Y", "Del",
    "North", "South", "East", "West", "Northern", "Southern", "Eastern", "Western",
    "Central", "Kingdom", "Hashemite", "MSF", "Republic", "Rural", "Urban", "Upper",
}


def _trie_regex(words: list[str]) -> str:
    """Alternation of `words` as a prefix trie (much faster than a flat
    alternation of ~2000 aliases). Longer matches are tried first."""
    trie: dict = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = True

    def build(node: dict) -> str:
        end = "" in node
        branches = [re.escape(ch) + build(sub) for ch, sub in sorted(node.items()) if ch]
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        return "(?:" + body + ")?" if end else body

    return build(trie)


@lru_cache(maxsize=1)
def _gazetteer():
    data = json.loads(COUNTRIES_JSON.read_text(encoding="utf-8"))
    names = {c["name"]: c["iso3"] for c in data}
    # folded lowercase alias -> list of (canonical name, weak, spec form)
    table: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)
    for c in data:
        for weak, key in ((False, "aliases"), (True, "weak_aliases")):
            for a in c.get(key, []):
                fa = fold(a)
                forms = [fa]
                if _ARABIC.search(fa) and fa.startswith("ال"):
                    forms.append("لل" + fa[2:])  # ل + ال contracts to لل
                for f in forms:
                    entry = (c["name"], weak, f)
                    if entry not in table[f.lower()]:
                        table[f.lower()].append(entry)
    latin = sorted((a for a in table if not _ARABIC.search(a)), key=len, reverse=True)
    arabic = sorted((a for a in table if _ARABIC.search(a)), key=len, reverse=True)
    latin_re = re.compile(
        r"(?<![\w\-])(?P<a>" + _trie_regex(latin) + r")(?![\w])",
        re.IGNORECASE)
    arabic_re = re.compile(
        r"(?<![\w])(?:[وف]?[بلك]?)(?P<a>" + _trie_regex(arabic) + r")(?![\w])")
    return names, table, latin_re, arabic_re


def iso3(name: str) -> str:
    """ISO3 code for a canonical country name (KeyError if unknown).

    Non-ISO entities use the codes common in map data: Kosovo XKX,
    Somaliland SOL (Natural Earth).
    """
    return _gazetteer()[0][name]


def country_names() -> list[str]:
    return sorted(_gazetteer()[0])


def _accept(match: re.Match, text: str, spec: str) -> bool:
    """Case and context checks for one alias match."""
    got = match.group("a")
    if _ARABIC.search(spec):
        return True
    letters = [c for c in spec if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):  # acronyms: DRC, UK, CAR
        return got == spec
    if spec[0].isupper() and not got[0].isupper():  # "Chad" yes, "turkey" no
        return False
    if spec in _PERSON_NAMES:
        start, end = match.span("a")
        before = text[max(0, start - 25):start]
        after = text[end:end + 6]
        prev = re.search(r"([A-Z][\w'.-]*)\s+$", before)
        if prev and prev.group(1) not in _FUNCTION_WORDS:
            return False  # "Michael Jordan", "S. Chad"
        if re.match(r" [A-Z]{1,3}\b(?![a-z])", after) and not re.match(r" (?:MSF|US|UK)\b", after):
            return False  # "Jordan MR," in an author list
    return True


def _mentions(text: str) -> list[tuple[int, int, str, bool, list[str]]]:
    """All country mentions: (start, end, alias spec, weak, candidate names)."""
    _, table, latin_re, arabic_re = _gazetteer()
    masked = _BLOCK_RE.sub(lambda m: " " * len(m.group(0)), text)
    out = []
    for rx in (latin_re, arabic_re):
        for m in rx.finditer(masked):
            entries = table[m.group("a").lower()]
            spec = entries[0][2]
            if not _accept(m, masked, spec):
                continue
            weak = all(e[1] for e in entries)
            cands = list(dict.fromkeys(e[0] for e in entries))
            out.append((m.start("a"), m.end("a"), spec, weak, cands))
    out.sort()
    return out


# --------------------------------------------------------------------------
# Document zones
# --------------------------------------------------------------------------

_REF_HEAD = re.compile(
    r"^\W{0,6}(?:\d+\.?\s*)?(?:references|bibliography|literature cited|references and notes|"
    r"references?\s+cited|references bibliographiques|bibliographie|referencias|bibliografia|"
    r"المراجع|المصادر)\W{0,6}$",
    re.IGNORECASE | re.MULTILINE)

_AFFIL = re.compile(
    r"universit|institut|school|department|faculty|college|laborator|ministry|"
    r"epicentre|medecins sans frontieres|medicos sin fronteras|doctors without borders|\bmsf\b|"
    r"hospital|centre|center|\bunit\b|foundation|fondation|@|correspond|affiliation|"
    r"research|programme|\bprogram\b|\binc\b|corp|\bltd\b|gmbh|software|stata|\bsas\b|"
    r"\bstreet\b|\brue\b|\broad\b|\bbox\b|cedex|\b\d{4,6}\b|press\b|publish|edition|elsevier|wiley|springer",
    re.IGNORECASE)
_STATE_CODE = re.compile(r"\b[A-Z]{2}\s*,\s*$")  # "College Station, TX, USA"
_PROSE_IN = re.compile(
    r"(?:\b(?:in|en|a|au|of|near|from|de|del|desde|depuis|في|من)\s+[\w' .-]{1,30}|\d{1,2}\s+[^\W\d]+),\s*$", re.IGNORECASE)


def _cut_references(text: str) -> str:
    cut = None
    for m in _REF_HEAD.finditer(text):
        if m.start() > 0.3 * len(text):
            cut = m.start()
            break
    return text[:cut] if cut else text


def _in_affiliation(text: str, start: int, mention_ends: set[int]) -> bool:
    """True for an address-style "City, Country" (author affiliations, vendor
    addresses such as "College Station, TX, USA"), as opposed to a list of
    countries ("Kenya, Uganda").
    """
    before = text[max(0, start - 150):start]
    m = re.search(r",\s*$", before)
    if not m:
        return False
    comma = start - (len(before) - m.start())
    if comma in mention_ends or comma - 1 in mention_ends:
        return False  # previous item is itself a country: a list
    if _STATE_CODE.search(before):
        return True
    item = re.split(r"[,;\n()]", before[:m.start()])[-1]
    if _PROSE_IN.search(before[:m.start() - len(item)] + item + ","):
        return False  # "in Bentiu, South Sudan", "28 August, Gaza"
    return bool(_AFFIL.search(before))


# --------------------------------------------------------------------------
# Countries
# --------------------------------------------------------------------------

TITLE_W = 4.0      # per title mention (max 2 counted)
LEAD_W = 2.0       # per mention in the first LEAD_CHARS of the body (max 4)
BODY_W = 1.0       # per mention after that (log-damped)
WEAK_W = 0.5       # demonyms and ambiguous forms
LEAD_CHARS = 1500
MAX_COUNTRIES = 5
MANY_COUNTRIES = 5
MANY_KEEP = 3
TITLE_MAX_WORDS = 15
# floors grow with length: 5 + k * log(1 + chars/10000); the top country's is
# lower, since its later mentions are log-damped (a long trial report naming
# DR Congo a dozen times must still pass), secondary countries keep the strict one
TOP_FLOOR_PER_LOG_LEN = 1.0
# a document scoring >= 2 for this many countries, whose top score is within
# this factor of the 4th, is a world overview (annual or financial report)
BROAD_COUNTRIES = 15
BROAD_MARGIN = 1.25
FLOOR_PER_LOG_LEN = 2.0


_ABSTRACT = re.compile(r"\b(?:abstract|summary|background|resume|resumen|ملخص)\b", re.IGNORECASE)


def _zone_hits(title: str, text: str):
    """Weighted mention counts per country in the title (t), lead (l) and
    body (b): (all mentions, proper names only)."""
    title = _prep(title)
    body = _cut_references(_prep(text))
    hits = {z: defaultdict(float) for z in "tlb"}
    strong = {z: defaultdict(float) for z in "tlb"}
    ambiguous: list[tuple[str, float, list[str]]] = []

    def add(zone: str, cands: list[str], weak: bool):
        w = WEAK_W if weak else 1.0
        if len(cands) > 1:
            ambiguous.append((zone, w, cands))  # ambiguous forms are always weak
            return
        hits[zone][cands[0]] += w
        if not weak:
            strong[zone][cands[0]] += w

    for _, _, _, weak, cands in _mentions(title):
        add("t", cands, weak)
    # The lead is the first LEAD_CHARS, plus as much after an abstract or
    # summary heading (articles spend their first page on authors).
    m = _ABSTRACT.search(body, 0, 8000)
    anchor = m.start() if m else len(body)

    def zone(pos: int) -> str:
        return "l" if pos < LEAD_CHARS or anchor <= pos < anchor + LEAD_CHARS else "b"

    found = _mentions(body)
    ends = {e for _, e, _, _, _ in found}
    for pos, _, _, weak, cands in found:
        if not _in_affiliation(body, pos, ends):
            add(zone(pos), cands, weak)

    # An ambiguous form ("Congo", "Congolese", "Korea") goes to the candidate
    # with the most proper-name evidence, and then counts as a full mention
    # (it is that country once resolved). Ties favour the earlier candidate in
    # _AMBIGUOUS_DEFAULT (DR Congo: in this collection a bare "Congo" with a
    # DR Congo signal, or with no signal at all, means DR Congo). Without any
    # evidence and no default, the mention is dropped.
    def evidence(c):
        return sum(strong[z][c] for z in "tlb")

    for zn, w, cands in ambiguous:
        best = max(cands, key=lambda c: (evidence(c), c in _AMBIGUOUS_DEFAULT))
        if evidence(best) > 0:
            hits[zn][best] += 1.0
        else:
            default = [c for c in cands if c in _AMBIGUOUS_DEFAULT]
            if default:
                hits[zn][default[0]] += w
    return hits, strong


_AMBIGUOUS_DEFAULT = {"DR Congo"}


def _score(hits) -> dict[str, float]:
    t, l, b = hits["t"], hits["l"], hits["b"]
    return {c: TITLE_W * min(t[c], 2) + LEAD_W * min(l[c], 4) + BODY_W * 3 * math.log1p(b[c])
            for c in set(t) | set(l) | set(b)}


def country_scores(title: str, text: str) -> dict[str, float]:
    """Relevance score per canonical country name (before thresholding)."""
    return _score(_zone_hits(title, text)[0])


def countries(title: str, text: str) -> list[str]:
    """Canonical names of the countries the document is about, most relevant first.

    Score = title mentions x4 (max 2) + mentions in the first 1500 characters
    x2 (max 4) + 3*log(1 + later mentions); demonyms ("Israeli") and ambiguous
    forms ("Congo") count half; references and address-style "City, Country"
    (author affiliations, vendors) are ignored.

    - A document naming 15+ countries whose top score is within 25% of the
      4th is a world overview: only title countries are returned.
    - The top country is kept when it clearly dominates (score >= 2 and more
      than twice the runner-up's proper-name score, and named in the title or
      first 1500 characters: a short abstract naming one country once), or
      when it reaches its floor (5 + log(1 + chars/10000)).
    - Other countries need a stricter floor (5 + 2 log(...)) counting proper names only (so "Israeli
      forces" in a Gaza update does not tag Israel) and 40% of the top score.
    - A country named in a title of at most 15 words is always kept.
    - Israel is not tagged on documents about Palestine (_CONFLICT_PARTY).
    - When more than 5 countries reach the floor on proper names (annual
      overviews, financial reports, ratification lists),
      only those within 70% of the top survive, and none if that is still >3.
    Generic documents (guideline chapters) return [].
    """
    hits, strong_hits = _zone_hits(title, text)
    scores = _score(hits)
    # A long "title" is usually the first sentence standing in for a missing
    # title; it still counts in the score but does not force a tag.
    title_set = _title_countries(title) if len((title or "").split()) <= TITLE_MAX_WORDS else []
    if not scores and not title_set:
        return []
    strong = _score(strong_hits)
    for main, party in _CONFLICT_PARTY:
        if main in scores and party in scores and scores[main] >= 0.5 * scores[party]:
            scores[main] = max(scores[main], scores[party])
            strong[main] = max(strong.get(main, 0.0), strong.get(party, 0.0))
            del scores[party]
    loglen = math.log1p(len(text or "") / 10000)
    top_floor = 5.0 + TOP_FLOOR_PER_LOG_LEN * loglen
    floor = 5.0 + FLOOR_PER_LOG_LEN * loglen
    # ties (capped lead counts) are broken by the total number of mentions
    ranked = sorted(scores, key=lambda c: (-scores[c], -sum(hits[z][c] for z in "tlb"), c))
    keep = []
    if ranked:
        top = scores[ranked[0]]
        second = max((strong.get(c, 0.0) for c in ranked[1:]), default=0.0)
        early = hits["t"][ranked[0]] + hits["l"][ranked[0]] > 0
        fourth = scores[ranked[3]] if len(ranked) > 3 else 0.0
        if sum(v >= 2.0 for v in scores.values()) >= BROAD_COUNTRIES and top < BROAD_MARGIN * fourth:
            # world overview (annual or financial report): no country stands out
            return title_set[:MAX_COUNTRIES]
        if top >= top_floor or (top >= 2.0 and second * 2 < top and early):
            keep.append(ranked[0])
        keep += [c for c in ranked[1:] if strong.get(c, 0) >= floor and scores[c] >= 0.4 * top]
        passing = [c for c in ranked if strong.get(c, 0) >= floor]
        if len(passing) > MANY_COUNTRIES:
            # Overview of many countries (annual or financial report): only
            # those that stand out from the crowd, if few enough.
            keep = [c for c in passing if scores[c] >= 0.7 * top]
            if len(keep) > MANY_KEEP:
                keep = []
    keep = title_set + [c for c in keep if c not in title_set]
    return keep[:MAX_COUNTRIES]


# (main country, other party): documents about the first name the second as a
# party to the conflict on nearly every line; that alone is not "about" it.
# When both are present and the main one has at least half the party's score,
# the party's score is folded into the main country.
_CONFLICT_PARTY = {("Palestine", "Israel")}


def _title_countries(title: str) -> list[str]:
    """Countries named (not just as a demonym) in the title, in order."""
    out: list[str] = []
    for _, _, _, weak, cands in _mentions(_prep(title)):
        if len(cands) == 1 and not weak and cands[0] not in out:
            out.append(cands[0])
    return out


# --------------------------------------------------------------------------
# Year
# --------------------------------------------------------------------------

MIN_YEAR, MAX_YEAR = 1971, 2026

_MONTHS = {
    # en
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # fr (accents folded)
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
    # es
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    # ar (folded: hamza seats removed, ة->ه, ى->ي)
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5, "يونيو": 6, "يوليو": 7,
    "اغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
    "كانون الثاني": 1, "شباط": 2, "اذار": 3, "نيسان": 4, "ايار": 5, "حزيران": 6,
    "تموز": 7, "اب": 8, "ايلول": 9, "تشرين الاول": 10, "تشرين الثاني": 11, "كانون الاول": 12,
    "جانفي": 1, "فيفري": 2, "افريل": 4, "جوان": 6, "جويليه": 7, "اوت": 8,
}
_MONTH_ALT = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))
_Y = r"(?P<y>(?:19|20)\d\d)(?!\d)"
# "23 February 2013", "February 23, 2013", "Feb. 1, 2019", "28 أكتوبر/تشرين الأول 2024",
# "le 3 mars 2021", "15 de mayo de 2026", "June 2025"
_DATE_RE = re.compile(
    r"(?<![\w])(?:(?P<d>\d{1,2})(?:er|st|nd|rd|th)?\s+(?:de\s+)?)?"
    r"(?P<m>" + _MONTH_ALT + r")\.?"
    r"(?:\s*/\s*(?:" + _MONTH_ALT + r"))?"
    r"(?:\s+(?P<d2>\d{1,2})(?:st|nd|rd|th)?)?,?\s+(?:de\s+|del\s+)?" + _Y,
    re.IGNORECASE)
_NUMERIC_DATE = re.compile(r"(?<![\d/.])(?:\d{1,2})[/.](?:\d{1,2})[/.]" + _Y)

# Optional day/month (any language, or numeric) between a marker and the year.
_DATE_TAIL = (
    r"(?:\s*(?:online|on|:|le|el|en|-|in))*\s*"
    r"(?:\d{1,2}(?:st|nd|rd|th|er)?[\s/.-]+(?:de\s+)?)?"
    r"(?:(?:" + _MONTH_ALT + r")\.?(?:\s*/\s*(?:" + _MONTH_ALT + r"))?[\s,/.-]*"
    r"(?:\d{1,2}(?:st|nd|rd|th)?,?\s*)?(?:de\s+)?)?(?:\d{1,2}[/.-])?")


def _marker(words: str) -> re.Pattern:
    return re.compile(r"(?<![\w])(?:" + words + r")" + _DATE_TAIL + _Y, re.IGNORECASE)


# Strong publication markers, in order of preference.
_PUB_PATTERNS = [
    ("published", _marker(r"published(?: online| date)?|first published|first posted online|"
                          r"epub(?: ahead of print)?|publie|publicado|date of publication|تاريخ النشر|نشر")),
    ("updated", _marker(r"last updated|last update|updated|mise a jour|mis a jour|actualizacion|"
                        r"actualizado|ultima actualizacion|اخر تحديث|تم التحديث|edition|edicion|طبعه")),
    # Journal citation headers: "BMJ Global Health 2020;5:e003236", "(2025) 19:7",
    # "Vol 381 February 23, 2013", "Lancet 2010; 376:"
    ("journal", re.compile(
        r"(?:vol(?:ume)?\.?\s*\d+[^\n]{0,30}?" + _Y + r")"
        r"|(?:(?<![\d(])" + _Y.replace("?P<y>", "?P<y2>") + r"\s*;\s*\d+\s*[(:])"
        r"|(?:\(" + _Y.replace("?P<y>", "?P<y3>") + r"\)\s*\d+\s*[:(])",
        re.IGNORECASE)),
    ("copyright", re.compile(
        r"(?:©|\(c\)|copyright)\s*(?:the authors?\(?s?\)?\s*)?[^\n\d]{0,40}?" + _Y, re.IGNORECASE)),
    ("accepted", _marker(r"accepted(?: for publication)?|accepte|aceptado")),
    ("received", _marker(r"received|recu|recibido")),
]

# Genres that describe a calendar year ("Activity report 2019").
REPORT_GENRES = {"Activity Report Chapter", "Financial Report"}
# Genres with no reliable publication date in the text.
UNDATED_GENRES = {"Legal Encyclopedia Entry"}


def _ok(y) -> int | None:
    try:
        y = int(y)
    except (TypeError, ValueError):
        return None
    return y if MIN_YEAR <= y <= MAX_YEAR else None


def _group_year(m: re.Match) -> int | None:
    for g in ("y", "y2", "y3"):
        try:
            v = m.group(g)
        except IndexError:
            continue
        if v:
            return _ok(v)
    return None


# A date right after these bounds a range or dates an event, not the document.
_SINCE = re.compile(
    r"(?:since|between|from|until|till|by|depuis|entre|desde|hasta|jusqu'?(?:en|au)?|منذ|بين|حتي)"
    r"\s+(?:de\s+|le\s+|el\s+|the\s+)?$", re.IGNORECASE)
_EVENT_PREP = re.compile(
    r"\b(?:in|on|during|after|before|en|durant|apres|avant|el|durante|despues|antes|في|خلال|بعد|قبل)\s+$",
    re.IGNORECASE)


def _dates(text: str) -> list[tuple[int, int, bool]]:
    """(offset, year, has_day) for every month-name or numeric date, except
    range or bound dates ("since May 2021", "entre junio 2019 y ...")."""
    out = []
    for m in _DATE_RE.finditer(text):
        y = _ok(m.group("y"))
        if y and not _SINCE.search(text[max(0, m.start() - 20):m.start()]):
            out.append((m.start(), y, bool(m.group("d") or m.group("d2"))))
    for m in _NUMERIC_DATE.finditer(text):
        y = _ok(m.group("y"))
        if y:
            out.append((m.start(), y, True))
    out.sort()
    return out


def year(raw: str, genre: str, source_file: str | Path | None = None) -> int | None:
    """Publication year, or None when unsure.

    When `source_file` is an original HTML page, its publication metadata
    (schema.org datePublished, article:published_time, or the page's
    publication-date <time>) wins: on news pages the text itself usually
    only dates the events, which precede publication. Otherwise the year
    comes from the text.

    Order: explicit markers (published, last updated or edition, journal
    citation header, copyright, accepted, received) in the first 8000
    characters; then a dated header line in the first 400 characters; then
    for report genres, the year named most often in their header. Dates that
    bound a range ("since May 2021", "entre junio 2019 y ...") are not used,
    nor are event dates deeper in the text.
    Years outside 1971-2026 are rejected.
    """
    if source_file is not None:
        y = html_year(source_file)
        if y:
            return y
    return _year_detail(raw, genre)[0]


_ROOT = Path(__file__).resolve().parents[1]
_HTML_DATE = [
    re.compile(r'"datePublished"\s*:\s*"((?:19|20)\d\d)-'),
    re.compile(r'article:published_time"\s+content="((?:19|20)\d\d)-'),
    re.compile(r'class="[^"]*(?:publication-date|published-at)[^"]*"[^<]*(?:<[^>]*>[^<]*){0,3}?'
               r'<time[^>]*datetime="((?:19|20)\d\d)-'),
]


def html_year(source_file: str | Path) -> int | None:
    """Publication year from an original HTML page's metadata, else None."""
    path = Path(source_file)
    if not path.is_absolute():
        path = _ROOT / path
    if path.suffix.lower() not in (".html", ".htm") or not path.exists():
        return None
    html = path.read_text(encoding="utf-8", errors="ignore")
    for rx in _HTML_DATE:
        m = rx.search(html)
        if m:
            return _ok(m.group(1))
    return None


def _journal_header(text: str, m: re.Match) -> bool:
    """A journal citation is the article's own header (or running footer),
    not an entry of its reference list: near the top, a "Vol N" form, or
    alone on its line rather than after "Author. Title. Journal"."""
    if m.start() < 2000 or m.group(0)[:3].lower() == "vol":
        return True
    line = text[text.rfind("\n", 0, m.start()) + 1:m.start()]
    return ". " not in line and not re.search(r"\bet al\b|\(PDF\)", line)


def _year_detail(raw: str, genre: str) -> tuple[int | None, str, int]:
    """(year, rule that fired, offset in the folded text) for debugging."""
    text = _prep(raw)
    if not text.strip() or genre in UNDATED_GENRES:
        return None, "undated genre", -1
    head = text[:8000]

    for kind, rx in _PUB_PATTERNS:
        found = [(m.start(), _group_year(m)) for m in rx.finditer(head)
                 if kind != "journal" or _journal_header(head, m)]
        found = [(p, y) for p, y in found if y]
        if found:
            if kind in ("journal", "updated"):
                p, y = found[0]  # the header comes first
            else:
                p, y = max(found, key=lambda x: x[1])
            return y, kind, p

    # A dated header line: "22 May 2018" right under the title.
    for p, y, has_day in _dates(text[:400]):
        if _EVENT_PREP.search(text[max(0, p - 12):p]):
            continue  # "In November 2025, ..." dates an event
        if genre not in REPORT_GENRES and (has_day or p < 150):
            return y, "header date", p

    if genre in REPORT_GENRES:
        ys = [(m.start(), int(m.group(1)))
              for m in re.finditer(r"(?<!\d)((?:19|20)\d\d)(?!\d)", text[:1500]) if _ok(m.group(1))]
        if ys:
            # the report year is the one named most often near the top
            only = [y for _, y in ys]
            best = max(set(only), key=lambda y: (only.count(y), y))
            return best, "report year", next(p for p, y in ys if y == best)

    # Dates deeper in the text are events, not publication: checked against
    # the HTML metadata, "latest dated event" was wrong half the time
    # (usually one year early), so no guess is made from them.
    return None, "no marker", -1
