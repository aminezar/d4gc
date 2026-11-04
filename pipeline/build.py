"""Build bundles/full.sqlite (or the 10-document sample) from data/.

    uv run python -m pipeline.build                 # bundles/full.sqlite
    uv run python -m pipeline.build --sample        # bundles/sample.sqlite (10 docs)
    uv run python -m pipeline.build --country "DR Congo" --out bundles/cod.sqlite
                                                    # one country plus global docs, built directly
                                                    # (same content as pipeline.export from full.sqlite)
"""

import argparse
import sys
import time
from pathlib import Path

from pipeline import clean, core_tags, flex_tags, hygiene, langid, titles
from pipeline.bundle import BundleDoc, write_bundle
from pipeline.chunk import chunk_text
from pipeline.embed import embed_passages
from pipeline.load import ROOT, load_docs


def prepare(sample: bool = False, country: str | None = None) -> tuple[list[BundleDoc], dict[str, str]]:
    """Load, clean, dedupe, chunk and tag. Embeddings are added by build()."""
    docs = load_docs()
    texts = {d.id: clean.clean_text(d.raw, d.ext) for d in docs if d.raw is not None}
    removed = hygiene.find_removed(texts)
    # Evidence is checked against the source text: cleaning may strip boilerplate a quote overlaps.
    flex = flex_tags.load_tags({d.id: d.raw for d in docs if d.raw is not None and d.id not in removed})
    pdf_titles = titles.load_titles({d.id: d.raw for d in docs if d.ext == "pdf" and d.raw is not None})

    out = []
    for d in docs:
        if d.id in removed:
            continue
        if d.raw is None:
            out.append(BundleDoc(d.id, None, d.genre, d.language, None, [], None))
            continue
        text = texts[d.id]
        original = (ROOT / d.source_file).read_text(encoding="utf-8", errors="ignore") if d.ext == "html" else None
        title = clean.truncate(pdf_titles[d.id], 200) if d.id in pdf_titles else clean.extract_title(d.raw, d.ext, text, original)
        out.append(BundleDoc(
            id=d.id, title=title, genre=d.genre, language=langid.detect(text, d.language),  # fixes CSV errors
            year=core_tags.year(d.raw, d.genre, d.source_file),
            countries=core_tags.countries(title, text),
            source_file=d.source_file,
            chunks=chunk_text(text),
            tags=flex.get(d.id, []),
        ))
    if sample:
        out = pick_sample(out)
    if country:  # same selection as bundle.bundle_candidates
        out = [d for d in out if d.source_file and (country in d.countries or not d.countries)]
    return out, removed


def pick_sample(docs: list[BundleDoc]) -> list[BundleDoc]:
    """10 real documents that exercise the demo: DR Congo (French first), global guidelines,
    Venezuela, Arabic and Spanish."""
    have = sorted((d for d in docs if 0 < len(d.chunks) <= 15), key=lambda d: d.id)  # small, so it builds fast
    picked: list[BundleDoc] = []

    def take(pred, n):
        for d in have:
            if n == 0:
                return
            if d not in picked and pred(d):
                picked.append(d)
                n -= 1

    take(lambda d: "DR Congo" in d.countries and d.language == "fr", 2)
    take(lambda d: "DR Congo" in d.countries, 4 - len(picked))
    take(lambda d: not d.countries and d.genre == "Clinical Guideline Chapter", 2)
    take(lambda d: "Venezuela" in d.countries, 1)
    take(lambda d: d.language == "ar", 1)
    take(lambda d: d.language == "es", 1)
    take(lambda d: True, 10 - len(picked))
    return picked


def build(out: Path, name: str, sample: bool = False, country: str | None = None) -> None:
    t0 = time.time()
    docs, removed = prepare(sample, country)
    print(f"prepared {len(docs)} docs, {sum(len(d.chunks) for d in docs)} chunks, "
          f"{len(removed)} removed ({time.time() - t0:.0f}s)", file=sys.stderr)

    inputs = []
    for d in docs:
        for c in d.chunks:
            inputs.append(f"{d.title}\n\n{c}")  # the title travels with every passage
    vectors = embed_passages(inputs)
    i = 0
    for d in docs:
        d.vectors = vectors[i:i + len(d.chunks)]
        i += len(d.chunks)

    if country:
        write_bundle(out, f"{country} bundle", docs, {}, {"country": country, "source": "MSF full collection"})
    else:
        write_bundle(out, name, docs, removed)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="store_true", help="10-document sample bundle")
    p.add_argument("--country", help='build one country bundle directly, e.g. "DR Congo"')
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.country and not a.out:
        p.error("--country needs --out")
    out = a.out or ROOT / "bundles" / ("sample.sqlite" if a.sample else "full.sqlite")
    build(out, "MSF sample (10 documents)" if a.sample else "MSF full collection", a.sample, a.country)


if __name__ == "__main__":
    main()
