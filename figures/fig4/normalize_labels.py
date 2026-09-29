#!/usr/bin/env python3
"""Normalize the two dORF labels used by the current paper."""

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = ROOT / "figures/figure_subplot_check/fig4/panels"
REPLACEMENTS = {"RT8LA": "RTL8A", "PPPC2A": "PPP2CA"}
FILES = ("F4B_0423.pdf", "F4D_E_0423.pdf")


def main() -> None:
    for name in FILES:
        path = PANEL_DIR / name
        document = fitz.open(path)
        replacements = []
        for page in document:
            spans = [
                span
                for block in page.get_text("dict")["blocks"]
                if "lines" in block
                for line in block["lines"]
                for span in line["spans"]
            ]
            for old, new in REPLACEMENTS.items():
                for rect in page.search_for(old):
                    size = next((span["size"] for span in spans
                                 if fitz.Rect(span["bbox"]).intersects(rect)), 8)
                    replacements.append((page, rect, new, size))
                    page.add_redact_annot(rect, fill=(1, 1, 1))
        for page in document:
            page.apply_redactions()
        for page, rect, new, size in replacements:
            page.insert_text((rect.x0, rect.y1 - 1.5), new, fontsize=size,
                             fontname="helv", color=(0, 0, 0), overlay=True)
        document.save(path.with_suffix(".normalized.pdf"), garbage=4, deflate=True)
        document.close()
        path.unlink()
        path.with_suffix(".normalized.pdf").rename(path)
        print(f"updated {path}: {len(replacements)} labels")


if __name__ == "__main__":
    main()
