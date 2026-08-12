"""Pipeline orchestrator.

For each product:
  1. Resolve a hero image  -> reuse an input asset if present, else generate.
  2. For each aspect ratio -> smart-crop, overlay message, stamp logo.
  3. Run brand + legal checks and record everything in a run report.

Reruns are idempotent: a generated hero is cached under output/<campaign>/
_cache so a second run doesn't re-hit the GenAI API.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .brief import CampaignBrief
from .checks import brand_color_check, legal_check
from .compositor import ASPECT_RATIOS, build_creative
from .providers import ImageProvider
from .storage import LocalStorage

log = logging.getLogger("pipeline")


class Pipeline:
    def __init__(self, provider: ImageProvider, storage: LocalStorage):
        self.provider = provider
        self.storage = storage

    def _resolve_hero(self, brief: CampaignBrief, product) -> tuple[Image.Image, str]:
        """Reuse an input asset if available; otherwise generate one (cached)."""
        existing = self.storage.find_asset(product.folder)
        if existing:
            log.info("Reusing input asset for '%s': %s", product.name, existing)
            return Image.open(existing).convert("RGBA"), "reused"

        cache_dir = self.storage.output_dir / brief.slug / "_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{product.folder}.png"
        if cache_file.exists():
            log.info("Using cached generated hero for '%s'", product.name)
            return Image.open(cache_file).convert("RGBA"), "generated (cached)"

        prompt = product.image_prompt or (
            f"Premium social ad hero image of {product.name}, "
            f"for {brief.target_audience} in {brief.target_region}, "
            f"clean studio lighting, high-end commercial photography"
        )
        log.info("Generating hero image for '%s' via %s",
                 product.name, type(self.provider).__name__)
        hero = self.provider.generate(prompt, aspect_ratio="1:1") \
            if getattr(self.provider, "supports_references", False) \
            else self.provider.generate(prompt)
        hero.convert("RGB").save(cache_file)
        return hero, "generated"

    def _ratio_base(self, hero: Image.Image, ratio_name: str, source: str) -> Image.Image:
        """For a freshly GENERATED hero, render non-square ratios natively via
        the provider (passing the square as a reference) so subjects aren't
        cropped. Otherwise, return the hero for a normal smart-crop.
        """
        import io as _io
        _AR = {"1x1": "1:1", "9x16": "9:16", "16x9": "16:9"}
        if source != "generated" or ratio_name == "1x1":
            return hero
        if not getattr(self.provider, "supports_references", False):
            return hero  # provider can't reframe; fall back to crop
        ar = _AR[ratio_name]
        orient = "vertical" if ratio_name == "9x16" else "widescreen horizontal"
        prompt = (
            f"Reframe this advertising image to a {orient} {ar} composition. "
            f"Keep the person, product, and logo fully visible and uncropped; "
            f"naturally extend the scene and background to fill the new format. "
            f"Preserve the subject's appearance, the product, and the brand style."
        )
        buf = _io.BytesIO(); hero.convert("RGBA").save(buf, format="PNG")
        try:
            return self.provider.generate(prompt, refs=[buf.getvalue()], aspect_ratio=ar)
        except Exception as e:
            log.warning("Reframe to %s failed (%s); using crop.", ar, e)
            return hero

    def run(self, brief: CampaignBrief, locale: str | None = None) -> dict:
        message = brief.localized_messages.get(locale, brief.campaign_message) if locale \
            else brief.campaign_message
        logo = brief.brand.logo

        report = {
            "campaign": brief.campaign_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "region": brief.target_region,
            "audience": brief.target_audience,
            "message": message,
            "locale": locale or "en",
            "legal_check": legal_check(message),
            "products": [],
        }
        if not report["legal_check"]["passed"]:
            log.warning("Legal check flagged terms: %s",
                        report["legal_check"]["flagged_terms"])

        for product in brief.products:
            hero, source = self._resolve_hero(brief, product)
            product_entry = {"product": product.name, "hero_source": source, "creatives": []}

            for ratio_name in ASPECT_RATIOS:
                base = self._ratio_base(hero, ratio_name, source)
                creative = build_creative(base, ratio_name, message, logo)
                rel = f"{brief.slug}/{product.folder}/{ratio_name}/creative.jpg"
                saved = self.storage.save_image(creative, rel)
                brand = brand_color_check(saved, brief.brand.colors)
                log.info("Saved %s (brand_pass=%s)", rel, brand.get("passed"))
                product_entry["creatives"].append({
                    "aspect_ratio": ratio_name,
                    "path": str(saved.relative_to(self.storage.output_dir)),
                    "brand_check": brand,
                })
            report["products"].append(product_entry)

        report_path = self.storage.output_dir / brief.slug / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("Report written: %s", report_path)
        return report
