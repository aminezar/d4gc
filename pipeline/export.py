"""Export a country bundle from the full bundle (the "Build bundle" button runs the same function).

    uv run python -m pipeline.export "DR Congo" --out bundles/cod.sqlite
"""

import argparse
import json

from pipeline.bundle import export_subset
from pipeline.load import ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("country", help='canonical name, e.g. "DR Congo"')
    p.add_argument("--full", default=ROOT / "bundles" / "full.sqlite")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(export_subset(a.full, a.country, a.out)))


if __name__ == "__main__":
    main()
