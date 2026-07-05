#!/usr/bin/env python3
"""Step 3a — locate every node label on the map.

pdftotext -bbox gives each text label with a bounding box in the PDF's
coordinate space, which is identical to the SVG viewBox (2328 x 1599). We use
that to record the (x, y) centre of every backbone node referenced by the
traffic index, so the front-end can draw an overlay link between the right two
points.

Output: web/data/nodes.json  ->  { "<code>": {"x":.., "y":.., "label":..} }
"""

import html
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PDF = ROOT / "network-energy-efficiency-research" / "switch-network-maps" / "optopo-SWITCHlan-A1-redacted.pdf"
OUT = HERE / "data" / "nodes.json"

# Traffic-log endpoint codes that are not the literal map label.
ALIASES = {
    "LZ": "LUZ",     # Luzern
    "ENSI": "ENS",   # ENSI site near Zurich cluster
    "FHSG": "FH",    # FH/SG (St. Gallen) — label drawn as "FH" / "SG"
    "GL": "GLI",     # Glarus
    "GO": "GOS",     # Gossau
    "HEPVD": "HEP",  # HEP Vaud near Lausanne
}

# Codes with no clear label on this (redacted) optical map. Left unmapped;
# the front-end simply won't draw these links and will report them.
UNMAPPED = {"CHU", "GR", "SLF"}

# The SWITCH legend (bottom-left) reuses node codes (EZ, RA, IX, CR, ...) as
# type examples. Ignore any label whose centre falls inside this box.
LEGEND_BOX = (170, 470, 950, 1210)  # xmin, xmax, ymin, ymax

# When a code still has several candidate positions, prefer the one nearest
# this hand-checked anchor (in SVG coords).
PREFERRED = {
    "EL": (612.3, 874.4),    # EL yellow static-core box (lower of the pair)
    "SG": (1953.1, 491.7),   # St. Gallen static-core box (lower of the pair)
}

def in_legend(x, y):
    xa, xb, ya, yb = LEGEND_BOX
    return xa <= x <= xb and ya <= y <= yb

NEEDED = set("AG BA BE BLU BU CE CHU CR EHL EL ENSI EZ FHSG FR GE GL GO GR "
             "HEPVD IBM IX IXG KR LG LS LZ NE PS RA SA SG SGD SGE SI SLF TO "
             "WI WSL YV ZH".split())

WORD_RE = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>')


def words_from_pdf():
    out = subprocess.run(["pdftotext", "-bbox", str(PDF), "-"],
                         capture_output=True, text=True, check=True).stdout
    words = []
    for m in WORD_RE.finditer(out):
        x0, y0, x1, y1, t = m.groups()
        words.append((html.unescape(t).strip(),
                      (float(x0) + float(x1)) / 2.0,
                      (float(y0) + float(y1)) / 2.0))
    return words


def main():
    words = words_from_pdf()
    # Collect all positions per literal token.
    by_token = {}
    for t, x, y in words:
        by_token.setdefault(t, []).append((x, y))

    nodes = {}
    report = {"mapped": [], "ambiguous": [], "missing": []}

    for code in sorted(NEEDED):
        if code in UNMAPPED:
            report["missing"].append(code)
            continue
        label = ALIASES.get(code, code)
        positions = [p for p in by_token.get(label, []) if not in_legend(*p)]
        if not positions:
            report["missing"].append(code)
            continue
        if len(positions) > 1:
            report["ambiguous"].append((code, label, len(positions)))
            if code in PREFERRED:
                ax, ay = PREFERRED[code]
                positions.sort(key=lambda p: (p[0] - ax) ** 2 + (p[1] - ay) ** 2)
        x, y = positions[0]
        nodes[code] = {"x": round(x, 1), "y": round(y, 1), "label": label}
        report["mapped"].append(code)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(nodes, f, indent=2)

    print(f"mapped   ({len(report['mapped'])}): {report['mapped']}")
    print(f"ambiguous ({len(report['ambiguous'])}): {report['ambiguous']}")
    print(f"missing  ({len(report['missing'])}): {report['missing']}")
    print(f"out: {OUT}")


if __name__ == "__main__":
    main()
