"""Campaign brief schema and loader.

The brief is the typed contract for the whole pipeline. Using Pydantic turns
"accept a brief in JSON/YAML" into validation + self-documentation: a malformed
brief fails loudly at the boundary instead of deep inside image generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    """A single product to generate creatives for."""

    name: str
    # Optional slug overrides the auto-derived folder name (from `name`).
    slug: Optional[str] = None
    # Prompt hint fed to the GenAI model when a hero image must be generated.
    image_prompt: Optional[str] = None

    @property
    def folder(self) -> str:
        return self.slug or self.name.lower().replace(" ", "_").replace("-", "_")


class Brand(BaseModel):
    """Brand assets/rules used for compositing and compliance checks."""

    # Path to a logo PNG (relative to repo root or absolute). Optional.
    logo: Optional[str] = None
    # Brand colors as hex strings, e.g. ["#0A7E8C", "#F4B41A"].
    colors: list[str] = Field(default_factory=list)


class CampaignBrief(BaseModel):
    """Top-level campaign brief. At least two products are required by the task."""

    campaign_name: str
    target_region: str
    target_audience: str
    # Default English message shown on every creative.
    campaign_message: str
    # Optional locale -> message map for the localization bonus.
    localized_messages: dict[str, str] = Field(default_factory=dict)
    products: list[Product]
    brand: Brand = Field(default_factory=Brand)

    @field_validator("products")
    @classmethod
    def _at_least_two(cls, v: list[Product]) -> list[Product]:
        if len(v) < 2:
            raise ValueError("Campaign brief must include at least two products.")
        return v

    @property
    def slug(self) -> str:
        return self.campaign_name.lower().replace(" ", "_").replace("-", "_")


def load_brief(path: str | Path) -> CampaignBrief:
    """Load and validate a brief from a JSON or YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Brief not found: {p}")
    raw = p.read_text(encoding="utf-8")
    data = json.loads(raw) if p.suffix.lower() == ".json" else yaml.safe_load(raw)
    return CampaignBrief.model_validate(data)
