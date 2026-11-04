"""Turn parsed document text (data/parsed/<id>.md) into clean paragraphs.

`clean_text` returns paragraphs separated by a blank line, with no hard line
breaks inside a paragraph. The rules below were derived from the corpus:

PDF-derived text
- Pages are separated by a blank-line run; lines are hard-wrapped every few words.
- Ligatures (U+FB00-FB06), sometimes followed by a stray space ("Eﬀ ective").
- Running headers/footers repeat at page edges ("www.thelancet.com Vol 381 ...",
  "Bull World Health Organ 2025;103:418–428| doi: ...") with page numbers.
- C0 control characters and private-use glyphs stand in for lost symbols/bullets.

HTML-derived text
- Paragraphs are already one per line; the noise is site chrome. Three site
  families carry it (counted on distinct html docs):
  * MSF Medical Guidelines (38 docs): app banner + language switcher header.
  * MSF Science Portal (14 docs): header/footer menus, "Similar Content" list.
  * msf.org resource pages (4 docs): 18 identical unit/centre blurbs.
  Lines are only removed when they belong to one of these recognised blocks,
  or match an exact line from BOILERPLATE_LINES (each found in >= 4 docs).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

__all__ = ["clean_text", "extract_title", "recurring_lines"]


# ---------------------------------------------------------------------------
# Character-level normalisation
# ---------------------------------------------------------------------------

_LIGATURE_SPACE = re.compile(r"([ﬀ-ﬆ]) (?=[^\W\d_])")
# C0 controls (except \t \n), DEL, C1 controls: lost symbols (±, ≤, bullets).
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_INVISIBLE = re.compile(r"[­​⁠﻿]")  # soft hyphen, zero-width space/joiner-free
_PRIVATE_USE = re.compile(r"[-]")


def _normalise_chars(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n\n")
    # "Eﬀ ective" -> "Effective": in this corpus a ligature followed by a space
    # and a letter is always a split word (checked: offspring, offers, findings...).
    text = _LIGATURE_SPACE.sub(r"\1", text)
    text = unicodedata.normalize("NFKC", text)
    # \x07 and private-use glyphs at line start are bullets in the source PDFs.
    text = re.sub(r"(?m)^[ \t]*(?:\x07|[-])[ \t]*", "• ", text)
    # A soft hyphen before a line break is a typesetting break: join the word.
    text = re.sub(r"\u00ad[ \t]*\n[ \t]*", "", text)
    text = _CONTROL.sub(" ", text)
    text = _PRIVATE_USE.sub(" ", text)
    text = _INVISIBLE.sub("", text)
    text = text.replace("\t", " ")
    return text


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", line)).strip()


# ---------------------------------------------------------------------------
# HTML boilerplate (derived with `recurring_lines` on the corpus)
# ---------------------------------------------------------------------------

# Exact lines (after normalisation) that occur verbatim in >= 4 distinct html
# docs and are site chrome, never document content.
_BOILERPLATE_RAW = ['.هذه الصفحة غير متاحة باللغة العربية',
 'Esta página no está disponible en este idioma.',
 'Cette page n’est pas disponible dans cette langue.',
 'This page is not available in this language.',
 'Skip to main content',
 'Aller au contenu principal',
 'Pasar al contenido principal',
 'Permalink',
 'Share',
 'Loading...']

# Hashes preserve exact matching of long site-chrome lines without source quotations.
_BOILERPLATE_HASHES = frozenset(
['34b50b808bf2f392da274c1c50b07c480b2ff2ef5ebeaade8d40443774bed36a',
 'cf736c73b1234ed335d414cb7450c4007f210cca16a136149d39ec7bab3b9351',
 'c5a353b061dc1585a7b205b10b499085cf465459af4c4cfbdb59268370c03ba0',
 '80e542c3dc77e90b6b9fa5281c8a03879d9fb68b479fd191d32b5665af6563f8',
 '4af5b58a2b8f82c0750076ea0c4225d80b05cc5509925222a8e34bc206d89595',
 'e0b14622485eac96e7acf6becb91b964ec33ab60d5c7ef04bfa68fe5db2acf18',
 'e9b09f97591b968d1350e74c4a46b6abe0af10ab9c31e137a0795a7029911cce',
 '779eebe0e3ffa6f801ae3805abecd4910d5a04f4e12217b1cdf695e9d3d1953b',
 'e16e00fb113ecedeb8d118f574920bdaf888394bcac9f9802c4cc8f3a09df1f5',
 '8540515700bee55046098807b6059e4ee78cd147ab2aa424a33329333f6d8495',
 '1509a3285fdef2b93117e448962bf6b641c421039845db60bac802eb1fd26101',
 'e77ff7c3f90be30b3e7e77d8cb5d116cc093614154016f4e52367b25740ebe30',
 '5036dbb35475cce9c4b98ffb8d6a432c2bf86f800de7239a9a1ef8e6a0e823ee',
 '0b0edc97af0f5de65c4ab42ec759ef102d020565996df0a7c26611747b59f2f2',
 'ba0d79318c83998a9d84d067f0f698dcce77d63d12cb2ef423cc2ab58943687f',
 'f2ff131222992e807fa4ca1592a548fa4cf3edaa375b67a04ec29904c6720ddb',
 '4710ca15d83881c209ce6ac1976734135744ae38128f56d755139631012926d0',
 '6d552b340999faa0b228f9671d5ad59dc24f427824ec2d8b79bec03e6c478241',
 'd9db5c991dfe240f54bc363063528fa5e0a62fe661aff8f59b58060b49d78dbf',
 '83ff8a0318f95536d6aa6386c6e95856326d2bb2f80795b6eecff12495bfafe8',
 '3fbc4576bee3aba8b35aa7b92d7369d00bd70d7467a5e3096a2bc643741432f4',
 '67cd9701d981d197b9155d5fb45cf64bc52ce9d420f28aa921562942d75d0df7',
 'c8b2ba28e68dbf82bdba2329845c8e902056a00371a9f2dbd9e02c1a63ffc8b0']
)
BOILERPLATE_LINES = frozenset(_norm_line(s) for s in _BOILERPLATE_RAW)


def _is_boilerplate(line: str) -> bool:
    normalized = _norm_line(line)
    return (normalized in BOILERPLATE_LINES
            or hashlib.sha256(normalized.encode()).hexdigest() in _BOILERPLATE_HASHES)


# Science Portal chrome: only removed inside Science Portal pages.
_PORTAL_CHROME = frozenset([
    "Science Portal", "Copyright © Médecins Sans Frontières", "v", "2.1", ".", "prod",
    "produseast1", "About MSF Science Portal", "About", "Contact Us",
    "Frequently Asked Questions (FAQs)", "Privacy Policy", "Terms of Use",
    "Export All Citations", "Save", "Cite", "Share", "Loading...", "Dimensions Badge",
    "View HTML", "Menu", "Home", "Conferences", "Collections", "Calls for Papers",
    "MSF Research Ethics", "Clinical Case Reporting", "My Saved Content",
    "Advanced Search", "Browse All", "English",
])

# Per-record widgets that also appear mid-page on Science Portal pages.
_PORTAL_WIDGETS = frozenset(["Save", "Cite", "Share", "Loading...", "Dimensions Badge",
                             "View HTML", "Export All Citations"])

_GUIDELINES_NOTICE = "This page is not available in this language."


def recurring_lines(texts: dict[str, str], min_docs: int = 4, min_len: int = 0) -> Counter:
    """Count, per normalised line, the number of distinct documents containing it.

    Identical documents are counted once, so mirrored pages do not inflate the
    counts. Used to derive BOILERPLATE_LINES; kept for re-deriving on new data.
    """
    seen: set[str] = set()
    counts: Counter = Counter()
    for text in texts.values():
        if text in seen:
            continue
        seen.add(text)
        counts.update({_norm_line(l) for l in text.split("\n") if len(l.strip()) > min_len})
    return Counter({l: n for l, n in counts.items() if n >= min_docs and l})


def _strip_html_blocks(lines: list[str]) -> list[str]:
    """Remove recognised site-chrome blocks. `lines` are stripped, may be ''."""
    nonblank = [i for i, l in enumerate(lines) if l]
    head = [lines[i] for i in nonblank[:40]]

    # Medical Guidelines header: banner + language notices, then the page
    # title, then "Select language: ... Enter".
    if _GUIDELINES_NOTICE in head:
        cut = nonblank[head.index(_GUIDELINES_NOTICE)]
        lines = [""] * (cut + 1) + lines[cut + 1:]
        for i, l in enumerate(lines):
            if l == "Select language:":
                end = next((j for j in range(i + 1, min(i + 10, len(lines))) if lines[j] == "Enter"), None)
                if end is not None:
                    for j in range(i, end + 1):
                        lines[j] = ""
                break

    # Science Portal pages.
    if "produseast1" in head or (head[:1] == ["Menu"] and "Science Portal" in head):
        stop = 0
        for k, l in enumerate(head):
            if l == "produseast1" or (head[0] == "Menu" and l in _PORTAL_CHROME):
                stop = nonblank[k] + 1
            elif head[0] == "Menu":
                break
        lines = [""] * stop + lines[stop:]
        # "Linked Content" / "Similar Content" are lists of *other* records.
        for marker in ("Linked Content", "Similar Content"):
            if marker in lines:
                lines = lines[: lines.index(marker)]
        while lines and (not lines[-1] or lines[-1] in _PORTAL_CHROME):
            lines.pop()
        lines = ["" if l in _PORTAL_WIDGETS else l for l in lines]
    return lines


# msf.org attachment links: "<name>pdf — 1.89 MBDownload".
_DOWNLOAD = re.compile(r"\s*(?:pdf|docx?|xlsx?|pptx?|zip)?\s*—\s*[\d.,]+\s*[KMG]B\s*Download\s*$", re.I)
_MD_HEADING = re.compile(r"^#{1,6}\s+")
_MD_BOLD = re.compile(r"\*\*(.*?)\*\*")
_MD_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\((?:[^()\s]|\([^()]*\))*\)")
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>~])")
_HTML_TAG = re.compile(r"</?(?:sup|sub|br|span|em|strong|b|i|u|p|div)\b[^>]*>", re.I)
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def _strip_markdown(line: str) -> str:
    line = _MD_HEADING.sub("", line)
    line = _DOWNLOAD.sub("", line)
    line = _MD_LINK.sub(r"\1", line)
    line = _HTML_TAG.sub("", line)
    line = _MD_BOLD.sub(r"\1", line)
    line = _MD_ITALIC.sub(r"\1", line)
    line = _MD_ESCAPE.sub(r"\1", line)
    line = line.replace("**", "")
    if line.startswith("|") and line.endswith("|") and len(line) > 1:
        cells = [c.strip() for c in line.strip("|").split("|")]
        line = " | ".join(c for c in cells if c)
    return re.sub(r"\s+", " ", line).strip()


def _has_word_char(s: str) -> bool:
    return any(ch.isalnum() for ch in s)


def _clean_html(text: str) -> list[str]:
    lines = [l.strip() for l in text.split("\n")]
    lines = _strip_html_blocks(lines)
    paras: list[str] = []
    prev_blank = True
    for raw in lines:
        if not raw:
            prev_blank = True
            continue
        if _TABLE_SEP.match(raw):
            continue
        line = _strip_markdown(raw)
        if not line or _is_boilerplate(line) or _is_boilerplate(raw):
            continue
        if not _has_word_char(line):
            # Lone separators ("," between author names, "|", "-", "×", ".").
            if line in {",", ";"} and paras:
                paras[-1] += line
            continue
        if line in {"-", "*", "•"}:
            continue
        line = re.sub(r"^[-*]\s+(?=\S)", "• ", line) if raw[:2] in ("- ", "* ") else line
        cont = (
            paras
            and not prev_blank
            and line[0].islower()
            and not paras[-1].endswith((".", "!", "?", ":"))
        ) or (paras and paras[-1].endswith(","))
        if cont:
            paras[-1] += " " + line
        else:
            paras.append(line)
        prev_blank = False
    return paras


# ---------------------------------------------------------------------------
# PDF: headers/footers, page numbers, hyphenation, paragraph reflow
# ---------------------------------------------------------------------------

_EDGE = 3  # lines inspected at the top and bottom of each page
_PAGE_NUM = re.compile(r"^(?:page\s+)?[-–—]?\s*(\d{1,4})\s*[-–—]?(?:\s*(?:/|of|\|)\s*\d{1,4})?$", re.I)
_BULLET = re.compile(r"^(?:[•▪◦●○■□➢►✓-]\s+|\(?[0-9]{1,2}[.)]\s+|\(?[a-z][.)]\s+)")
_SENT_END = (".", "!", "?", ":", "…", ".”", '."', ".)", "?”", "!”")
# Halves that make "x-\ny" a real compound when the doc gives no evidence
# either way (from auditing the corpus' line-break hyphens).
_COMPOUND_LEFT = frozenset(
    "post non anti multi low high middle long short self well cross semi intra extra "
    "pan mid ill full half all one two three four five six seven eight ten first "
    "second third community facility hospital home cost evidence".split()
)
_COMPOUND_RIGHT = frozenset(
    "based term line income course wage held led related free scale up care level "
    "positive negative specific confirmed resistant associated induced dependent "
    "risk month months year years week weeks day days old time old going making "
    "like wide oriented centred centered driven saving threatening acquired "
    "controlled sensitive susceptible naive infected exposed treated year-old".split()
)
_JOIN_SUFFIXES = (
    "tion", "tions", "sion", "sions", "ment", "ments", "ing", "ings", "ity", "ities",
    "ness", "ance", "ence", "ances", "ences", "ive", "ives", "ous", "ally", "ically",
    "ated", "ating", "ate", "ates", "ised", "ized", "ise", "ize", "ible", "able", "ary",
    "ers", "er", "ed", "es", "ly", "al", "ic", "ics", "ist", "ists", "ism", "logy",
)


def _sig(line: str) -> str:
    """Signature of a header/footer line: digits collapsed, case folded."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line)).strip().casefold()


def _strip_headers_footers(pages: list[list[str]]) -> list[list[str]]:
    n = len(pages)
    if n < 2:
        return pages
    edge_counts: Counter = Counter()
    for p in pages:
        edges = set(p[:_EDGE]) | set(p[-_EDGE:])
        edge_counts.update({_sig(l) for l in edges if _has_word_char(l)})
    # A running header must recur on several pages. Chapter-level running heads
    # in long books recur less often than journal headers, so the bar is low but
    # never below 2 occurrences (3 for docs over 6 pages).
    need = 2 if n <= 6 else 3
    running = {
        s for s, k in edge_counts.items()
        if k >= need and k >= 0.05 * n and len(s) <= 160
        # Number-only edge lines are table values, not headers (page numbers
        # are handled below by their consistent offset to the page index).
        and sum(ch.isalpha() for ch in s) >= 3
    }

    # Bare page numbers: edge lines holding a number that continues a
    # sequence with a nearby page (n on one page, n+1..n+4 within the next few
    # pages). Blank-line runs are not always page breaks, so the sequence test
    # is local rather than a fixed offset to the block index.
    cand: list[tuple[int, int, int]] = []  # (page, line index, number)
    for pi, p in enumerate(pages):
        idxs = set(range(min(_EDGE, len(p)))) | set(range(max(0, len(p) - _EDGE), len(p)))
        for li in sorted(idxs):
            m = _PAGE_NUM.match(p[li])
            if m:
                cand.append((pi, li, int(m.group(1))))
    drop_num: set[tuple[int, int]] = set()
    for a_i, (pa, la, na) in enumerate(cand):
        for pb, lb, nb in cand[a_i + 1 : a_i + 12]:
            if 0 < pb - pa <= 4 and 0 < nb - na <= 4 and nb - na <= (pb - pa) + 1:
                drop_num.update({(pa, la), (pb, lb)})
                break

    out: list[list[str]] = []
    kept_once: set[str] = set()
    for pi, p in enumerate(pages):
        edge_idx = set(range(min(_EDGE, len(p)))) | set(range(max(0, len(p) - _EDGE), len(p)))
        new = []
        for li, l in enumerate(p):
            if (pi, li) in drop_num:
                continue
            if li in edge_idx:
                s = _sig(l)
                if s in running:
                    # Keep the first occurrence: it often carries the journal,
                    # volume and date (used for year tagging).
                    if s in kept_once:
                        continue
                    kept_once.add(s)
            new.append(l)
        out.append(new)
    return out


_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")


_LINE_BREAK_HYPHEN = re.compile(r"-[ \t]*\n\s*")


def _drop_broken_words(text: str) -> str:
    """Blank out words hyphenated across a line break ("multidrug-\nresistant")."""
    out, pos = [], 0
    for m in _LINE_BREAK_HYPHEN.finditer(text):
        a, b = m.start(), m.end()
        if a < pos:
            continue
        while a > pos and text[a - 1].isalpha():
            a -= 1
        while b < len(text) and text[b].isalpha():
            b += 1
        out.append(text[pos:a])
        out.append(" ")
        pos = b
    out.append(text[pos:])
    return "".join(out)


def _vocab(text: str) -> tuple[Counter, Counter]:
    """Counts of whole tokens (incl. hyphenated compounds) and of their parts.

    Words broken across lines are left out, so a break never votes for itself.
    """
    text = _drop_broken_words(text)
    whole: Counter = Counter(_WORD.findall(text.casefold()))
    parts: Counter = Counter()
    for w, k in whole.items():
        for part in w.split("-"):
            parts[part] += k
    return whole, parts


def _join_hyphen(left: str, right: str, whole: Counter, parts: Counter) -> bool:
    """True if `left-` + newline + `right` is a typesetting break (drop hyphen)."""
    lw = re.search(r"([^\W\d_]+)$", left)
    rw = re.match(r"([^\W\d_]+)", right)
    if not lw or not rw or not rw.group(1)[0].islower():
        return False  # "MDR-\nTB", "2013-\n2014": keep the hyphen
    l, r = lw.group(1).casefold(), rw.group(1).casefold()
    joined, hyph = l + r, f"{l}-{r}"
    if whole[hyph] > 0 and whole[hyph] >= whole[joined]:
        return False
    if whole[joined] > 0:
        return True
    if l in _COMPOUND_LEFT or r in _COMPOUND_RIGHT:
        return False
    if r.endswith(_JOIN_SUFFIXES) and parts[r] == 0:
        return True
    # Both halves are words used elsewhere in the doc: a real compound.
    if parts[l] > 0 and parts[r] > 0:
        return False
    return True


def _reflow_pages(pages: list[list[str]], text_for_vocab: str) -> list[str]:
    whole, parts = _vocab(text_for_vocab)
    lengths = sorted(len(l) for p in pages for l in p)
    if not lengths:
        return []
    typical = lengths[int(len(lengths) * 0.75)] or 1  # long-line length of the doc
    paras: list[str] = []
    for p in pages:
        if not p:
            continue
        page_paras = _reflow_lines(p, whole, parts, typical)
        if (
            paras
            and page_paras
            and not paras[-1].endswith(_SENT_END)
            and (page_paras[0][:1].islower() or paras[-1].endswith((",", "-")))
        ):
            # Paragraph continues across the page break.
            last = paras.pop()
            if last.endswith("-") and _join_hyphen(last[:-1], page_paras[0], whole, parts):
                page_paras[0] = last[:-1] + page_paras[0]
            else:
                page_paras[0] = last + ("" if last.endswith("-") else " ") + page_paras[0]
        paras.extend(page_paras)
    return paras


_REF_NUM = re.compile(r"^\d{1,3}\.?$")


def _reflow_lines(lines: list[str], whole: Counter, parts: Counter, typical: float) -> list[str]:
    """Merge hard-wrapped lines into paragraphs."""
    paras: list[str] = []
    cur = ""
    last_len = 0  # length of the last physical line appended to `cur`
    for line in lines:
        if not cur:
            cur, last_len = line, len(line)
            continue
        if cur.endswith("-") and not cur.endswith(("--", " -")) and line[:1].isalpha():
            cur = (cur[:-1] if _join_hyphen(cur[:-1], line, whole, parts) else cur) + line
            last_len = len(line)
            continue
        if _REF_NUM.match(cur):
            # "4" on its own line followed by the reference text.
            cur, last_len = cur + " " + line, len(line)
            continue
        short_prev = last_len < 0.7 * typical
        new_para = (
            _BULLET.match(line) is not None
            or (cur.endswith(_SENT_END) and _REF_NUM.match(line) is not None)
            # Safety valve for layouts where paragraph ends are not short lines.
            or (cur.endswith(_SENT_END) and cur.count(" ") >= 250)
            or cur.count(" ") >= 600
            or (short_prev and cur.endswith(_SENT_END))
            or (
                # A short line followed by a capitalised one: heading, table
                # cell, author line... keep it as its own paragraph.
                short_prev
                and not cur.endswith((",", ";", "(", "–", "/", "&", "and", "or", "of", "the", "to", "in"))
                and not line[:1].islower()
            )
        )
        if new_para:
            paras.append(cur)
            cur = line
        else:
            cur = cur + " " + line
        last_len = len(line)
    if cur:
        paras.append(cur)
    return paras


def _clean_pdf(text: str) -> list[str]:
    pages = []
    for block in re.split(r"\n[ \t]*\n+", text):
        lines = [" ".join(l.split()) for l in block.split("\n")]
        lines = [l for l in lines if l]
        if lines:
            pages.append(lines)
    pages = _strip_headers_footers(pages)
    paras = _reflow_pages(pages, text)
    return [p for p in paras if _has_word_char(p)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_text(raw: str, ext: str) -> str:
    """Clean parsed document text.

    ext is "pdf" or "html" (the format of data/original/<id>.<ext>). Returns
    paragraphs separated by "\\n\\n"; no newlines inside a paragraph.
    """
    text = _normalise_chars(raw or "")
    paras = _clean_pdf(text) if ext == "pdf" else _clean_html(text)
    paras = [" ".join(p.split()) for p in paras]
    return "\n\n".join(p for p in paras if p)


_GENERIC_TITLES = {
    s.casefold()
    for s in [
        "abstract", "résumé", "resumen", "ملخص", "ملخّص", "summary", "latest news", "sommaire",
        "contents", "contenido", "table of contents", "correspondence", "research", "article",
        "original article", "research article", "open access", "editorial", "comment",
        "letter", "review", "introduction", "background", "methods", "menu", "home",
        "permalink", "report", "press release", "news", "msf", "médecins sans frontières",
        "journal article", "conference material", "video", "slide presentation", "poster",
        "research letter", "brief report", "perspective", "policy brief", "lessons from the field",
    ]
}
_TITLE_JUNK = re.compile(
    r"(www\.|https?://|doi[:.\s]|\bdoi\b|issn|isbn|©|copyright|\bvol\.?\s*\d|\bvolume\s+\d|"
    r"\d{4};\s*\d+|received:|accepted:|published:|citation:|journal of|^page\s+\d|"
    r"creative commons|licen[cs]e|\bet al\b|provisional pdf|charity reg|registered charity|"
    r"jour ?nal of medicine|^presented by|^laurence binet$|speaking out case studies|"
    r"^(?:the )?lancet(?: [a-z]+){0,3}$|@|\bpage \d+ of \d+|full text|^item type|"
    r"^published online|^permits use|all rights reserved|^see online|"
    r"^(?:bmc [\w ]+|plos [\w ]+|malaria journal|conflict and health|trials|tropical medicine and international health)$)",
    re.I,
)
_CONNECTORS = {"of", "and", "the", "in", "for", "to", "a", "an", "with", "through", "on", "at",
               "from", "by", "de", "la", "le", "des", "du", "et", "en", "y", "el", "los", "las",
               "versus", "vs", "among", "during", "after", "before", "under", "into",
               "towards", "between", "across", "against", "within"}


def _title_ok(s: str) -> bool:
    s = s.strip()
    if not (4 <= len(s) <= 250) or s.casefold().rstrip(":.") in _GENERIC_TITLES:
        return False
    if _TITLE_JUNK.search(s):
        return False
    letters = sum(ch.isalpha() for ch in s)
    if letters < 0.6 * len(s.replace(" ", "")):
        return False
    return len(s.split()) >= 2 or letters >= 12


def _truncate(s: str, n: int = 120) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:–- ") + "…"


def _first_sentence(cleaned: str) -> str:
    for para in cleaned.split("\n\n"):
        para = para.strip()
        if _has_word_char(para) and len(para) >= 20 and para.casefold().rstrip(":. ") not in _GENERIC_TITLES:
            m = re.match(r"(.+?[.!?؟])(\s|$)", para)
            return _truncate(m.group(1) if m else para)
    return ""


def _html_title(raw: str) -> str:
    text = _normalise_chars(raw)
    lines = [l.strip() for l in text.split("\n")]
    nonblank = [l for l in lines if l]
    # Guidelines pages: the title follows the language notices.
    if _GUIDELINES_NOTICE in nonblank[:40]:
        i = nonblank.index(_GUIDELINES_NOTICE) + 1
        if i < len(nonblank):
            cand = _strip_markdown(nonblank[i])
            if 2 <= len(cand) <= 200 and cand != "Select language:" and _has_word_char(cand):
                return cand
    # Science Portal: "<type>", "|", "<subtype>", "<title>".
    if "Science Portal" in nonblank[:20]:
        for k in range(len(nonblank) - 3):
            if nonblank[k + 1] == "|" and k < 60:
                cand = _strip_markdown(nonblank[k + 3])
                if _title_ok(cand):
                    return cand
                break
    # First markdown heading of level 1-3 that is not a generic label.
    for l in nonblank[:80]:
        m = re.match(r"^#{1,3}\s+(.*)$", l)
        if m:
            cand = _strip_markdown(m.group(1))
            if _title_ok(cand) and len(cand) <= 200:
                return cand
    return ""


def _authorish(line: str) -> bool:
    """Author/affiliation lines: name lists with superscripts or roles."""
    return bool(line.count(",") >= 2 or re.search(r"[^\W\d_]\d|\d,|[*†‡]| - ", line))


def _pdf_title(raw: str) -> str:
    text = _normalise_chars(raw)
    first = [re.sub(r"\s+", " ", l).strip() for l in text.split("\n")[:60]]
    first = [l for l in first if l]
    whole = parts = None
    for i, l in enumerate(first[:40]):
        if not _title_ok(l) or len(l.split()) < 2 or l[:1].islower() or _authorish(l):
            continue
        title = l
        j = i + 1
        while j < len(first) and len(title) < 140 and not title.endswith((".", "?", "!")):
            nxt = first[j]
            last_word = title.rsplit(" ", 1)[-1].casefold()
            words = [w for w in re.findall(r"[^\W\d_]+", nxt)]
            title_case = words and sum(w[0].isupper() or w.casefold() in _CONNECTORS for w in words) >= 0.8 * len(words)
            authorish = _authorish(nxt)
            if (nxt[:1].islower() or last_word in _CONNECTORS or title.endswith(("-", ":", ","))
                    or (title_case and len(words) >= 2 and not title.isupper())) \
                    and _title_ok(nxt) and len(nxt) < 120 and not authorish:
                if title.endswith("-"):
                    if whole is None:
                        whole, parts = _vocab(text)
                    title = (title[:-1] if _join_hyphen(title[:-1], nxt, whole, parts) else title) + nxt
                else:
                    title = title + " " + nxt
                j += 1
            else:
                break
        return _truncate(title, 150)
    return ""


_SITE_SUFFIXES = {"MSF", "أطباء بلا حدود", "MSF Science Portal", "Epicentre", "MSF Access"}
_GUIDE_SITES = {"MSF Medical Guidelines", "Guides médicaux MSF", "Guías de práctica clínica MSF",
                "The Practical Guide to Humanitarian Law"}
_CRUMB_START = {"Breadcrumb", "Sobrescribir enlaces de ayuda a la navegación"}
_CRUMB_HOME = {"Home", "Accueil", "Inicio", "الرئيسية"}


def _breadcrumb_root(original: str) -> str:
    """First breadcrumb entry after "Home" on a guidelines page: the guide the page belongs to."""
    import html as _html

    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", original, flags=re.S | re.I)
    lines = [_html.unescape(l).strip() for l in re.sub(r"<[^>]+>", "\n", body).split("\n")]
    lines = [l for l in lines if l]
    for k in range(len(lines) - 2):
        if lines[k] in _CRUMB_START and lines[k + 1] in _CRUMB_HOME:
            return lines[k + 2]
    return ""


def html_page_title(original: str) -> str:
    """Title from the original page's <title> tag, without the site name.

    Guidelines pages are often a section ("Annexes", "Tuberculose pulmonaire"), so the guide
    they belong to is added from the breadcrumb: "Annexes (Prise en charge d'une ÉPIDÉMIE DE CHOLÉRA)".
    """
    import html as _html

    m = re.search(r"<title[^>]*>(.*?)</title>", original or "", re.S | re.I)
    if not m:
        return ""
    parts = [p.strip() for p in _html.unescape(re.sub(r"\s+", " ", m.group(1))).split(" | ")]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    site = parts[-1] if len(parts) > 1 else ""
    if site == "MSF Science Portal":
        parts = parts[:1]  # "<title> | <article type> | MSF Science Portal"
    elif site in _SITE_SUFFIXES or site in _GUIDE_SITES:
        parts = parts[:-1]
    page = re.sub(r"\s*-\s*Cloned$", "", ": ".join(parts)).strip()
    if site in _GUIDE_SITES:
        guide = _breadcrumb_root(original)
        if guide and guide.casefold() not in page.casefold():
            page = f"{page} ({guide})"
    return page if _has_word_char(page) else ""


def extract_title(raw: str, ext: str, cleaned: str, original: str | None = None) -> str:
    """Human-readable title for a document. Never empty.

    ``original`` is the source HTML; its <title> tag is the most reliable title when present.
    """
    if ext == "html" and original:
        page = html_page_title(original)
        if page:
            # Country pages are titled with the country alone ("Nigeria | MSF"): add the heading.
            heading = _html_title(raw or "")
            if (len(page.split()) <= 2 and heading and len(heading) <= 80
                    and not heading.endswith(".") and page.casefold() not in heading.casefold()
                    and heading.casefold() not in page.casefold()):
                page = f"{page}: {heading}"
            return _truncate(page, 200)
    title = ""
    try:
        title = _pdf_title(raw or "") if ext == "pdf" else _html_title(raw or "")
    except Exception:  # title is best effort; fall back below
        title = ""
    if not title:
        for para in cleaned.split("\n\n")[:5]:
            if _title_ok(para) and len(para) <= 150:
                title = para
                break
    if not title:
        title = _first_sentence(cleaned)
    if not title:
        title = _truncate(re.sub(r"\s+", " ", raw or "").strip()) or "Untitled document"
    title = re.sub(r"^[•\-*]\s*", "", title.strip())
    return _truncate(title.strip(" -–:|"), 200) or "Untitled document"


def truncate(s: str, n: int = 120) -> str:
    """Collapse whitespace and cut at a word boundary with an ellipsis."""
    return _truncate(s, n)
