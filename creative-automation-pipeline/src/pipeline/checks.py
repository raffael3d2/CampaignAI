"""Nice-to-have checks: brand compliance + legal content flagging.

These are intentionally lightweight but real:
  - brand_color_check: does the creative actually contain the brand palette?
  - legal_check: does the campaign message contain prohibited/regulated words?
Both return structured results that feed the run report.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

# A small illustrative blocklist. In production this would be a
# per-market, per-category compliance dictionary (e.g. no "cure", "guaranteed",
# "#1" without substantiation, etc.).
PROHIBITED_WORDS = {
    "guaranteed", "miracle", "cure", "risk-free", "100% safe",
    "best in the world", "clinically proven",
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _close(a, b, tol=40) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def brand_color_check(image_path: str | Path, brand_colors: list[str]) -> dict:
    """Sample the image and report which brand colors are present."""
    if not brand_colors:
        return {"checked": False, "reason": "no brand colors defined"}
    img = Image.open(image_path).convert("RGB").resize((80, 80))
    pixels = list(img.getdata())
    present = {}
    for hexc in brand_colors:
        target = _hex_to_rgb(hexc)
        hits = sum(1 for p in pixels if _close(p, target))
        present[hexc] = round(hits / len(pixels), 3)
    found = [c for c, frac in present.items() if frac > 0.005]
    return {
        "checked": True,
        "passed": len(found) > 0,
        "colors_present": found,
        "coverage": present,
    }


def legal_check(message: str) -> dict:
    lowered = message.lower()
    flagged = sorted({w for w in PROHIBITED_WORDS if w in lowered})
    return {"checked": True, "passed": not flagged, "flagged_terms": flagged}
