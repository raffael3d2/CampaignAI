"""Compositing: turn one hero image into finished creatives per aspect ratio.

Steps per ratio:
  1. Smart center-crop the hero to the target ratio (no distortion).
  2. Overlay the campaign message with an auto-contrast scrim so text is
     always legible regardless of the underlying image.
  3. Optionally stamp the brand logo.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fonts")
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")

# Named aspect ratios -> (width, height) render targets.
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1x1": (1080, 1080),
    "9x16": (1080, 1920),
    "16x9": (1920, 1080),
}


def smart_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop `img` to the target aspect ratio, then resize to exact size."""
    src_ratio = img.width / img.height
    tgt_ratio = target_w / target_h
    if src_ratio > tgt_ratio:
        new_w = int(img.height * tgt_ratio)
        left = (img.width - new_w) // 2
        box = (left, 0, left + new_w, img.height)
    else:
        new_h = int(img.width / tgt_ratio)
        top = (img.height - new_h) // 2
        box = (0, top, img.width, top + new_h)
    return img.crop(box).resize((target_w, target_h), Image.LANCZOS)


def _fit_font(draw, text, max_w, start_size) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > 12:
        font = ImageFont.truetype(FONT_BOLD, size)
        if draw.textlength(text, font=font) <= max_w:
            return font
        size -= 2
    return ImageFont.truetype(FONT_BOLD, 12)


def _wrap(draw, text, font, max_w) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def overlay_message(img: Image.Image, message: str) -> Image.Image:
    """Draw the campaign message in the lower third with a gradient scrim."""
    img = img.convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    margin = int(W * 0.06)
    max_text_w = W - 2 * margin
    base = _fit_font(draw, message, max_text_w, start_size=int(H * 0.075))
    lines = _wrap(draw, message, base, max_text_w)

    line_h = base.size + int(base.size * 0.25)
    block_h = line_h * len(lines)

    # Scrim: darken the bottom band so light text stays legible on any image.
    scrim_top = H - block_h - int(margin * 1.6)
    scrim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(scrim)
    for y in range(scrim_top, H):
        alpha = int(180 * (y - scrim_top) / max(1, H - scrim_top))
        sdraw.line([(0, y), (W, y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, scrim)

    draw = ImageDraw.Draw(img)
    y = scrim_top + int(margin * 0.4)
    for line in lines:
        draw.text((margin, y), line, font=base, fill=(255, 255, 255, 255))
        y += line_h
    return img


def stamp_logo(img: Image.Image, logo_path: str | Path) -> Image.Image:
    """Place the brand logo top-left, scaled to ~14% of the canvas width."""
    logo_path = Path(logo_path)
    if not logo_path.exists():
        return img
    img = img.convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")
    target_w = int(img.width * 0.14)
    ratio = target_w / logo.width
    logo = logo.resize((target_w, int(logo.height * ratio)), Image.LANCZOS)
    margin = int(img.width * 0.04)
    img.alpha_composite(logo, (margin, margin))
    return img


def _remove_flat_background(product: Image.Image, tol: int = 32) -> Image.Image:
    """Best-effort background knockout using Pillow only (no heavy deps).

    Samples the four corners; pixels close to that background color become
    transparent. Works well for product shots on clean/near-solid backgrounds.
    Falls back to the original image if corners disagree (busy background)."""
    im = product.convert("RGBA")
    px = im.load()
    w, h = im.size
    corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
    # If corners are wildly different, the background isn't flat — skip knockout.
    r0 = sum(c[0] for c in corners) // 4
    g0 = sum(c[1] for c in corners) // 4
    b0 = sum(c[2] for c in corners) // 4
    spread = max(max(c[i] for c in corners) - min(c[i] for c in corners)
                 for i in range(3))
    if spread > 60:
        return im  # busy background; caller will place it on a card instead

    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    opx = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if abs(r - r0) <= tol and abs(g - g0) <= tol and abs(b - b0) <= tol:
                opx[x, y] = (r, g, b, 0)
            else:
                opx[x, y] = (r, g, b, a)
    return out


def overlay_product(background: Image.Image, product: Image.Image,
                    scale: float = 0.55) -> Image.Image:
    """Composite a real product image onto a generated background scene.

    Knocks out a flat product background when possible, then centers the
    product in the lower-middle of the scene. This is the 'reference-composite'
    path: the finished ad contains the ACTUAL product pixels."""
    bg = background.convert("RGBA")
    W, H = bg.size
    knocked = _remove_flat_background(product)

    # If knockout failed (fully opaque still), drop the product on a soft card
    # so it still reads as intentional rather than a hard rectangle.
    alpha = knocked.split()[3]
    if alpha.getextrema() == (255, 255):
        card = Image.new("RGBA", knocked.size, (255, 255, 255, 230))
        card.alpha_composite(knocked)
        knocked = card

    target_w = int(W * scale)
    ratio = target_w / knocked.width
    knocked = knocked.resize((target_w, int(knocked.height * ratio)), Image.LANCZOS)
    x = (W - knocked.width) // 2
    y = int(H * 0.52) - knocked.height // 2
    y = max(0, min(y, H - knocked.height))
    bg.alpha_composite(knocked, (x, y))
    return bg


def build_creative(
    hero: Image.Image,
    ratio_name: str,
    message: str,
    logo_path: Optional[str] = None,
) -> Image.Image:
    w, h = ASPECT_RATIOS[ratio_name]
    canvas = smart_crop(hero, w, h)
    canvas = overlay_message(canvas, message)
    if logo_path:
        canvas = stamp_logo(canvas, logo_path)
    return canvas
