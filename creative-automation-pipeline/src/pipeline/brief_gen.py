"""Turn a free-text creative brief into a structured, validated CampaignBrief.

The user types something like "launch two summer skincare products in Spain for
young women, message about natural radiance" and we ask Claude to expand it into
the exact JSON our Pydantic schema expects. The schema is the safety net: if the
model returns something malformed, validation fails loudly rather than breaking
deep in the pipeline.

Requires ANTHROPIC_API_KEY. Falls back with a clear error otherwise.
"""
from __future__ import annotations

import json
import os

from .brief import CampaignBrief

# Kept in sync with brief.py. Described to the model so it emits valid JSON.
_SCHEMA_HINT = """
Return ONLY a JSON object (no markdown, no prose) with exactly these fields:
{
  "campaign_name": "short campaign title",
  "target_region": "region or market, e.g. 'Spain (ES)'",
  "target_audience": "one sentence describing the audience",
  "campaign_message": "the English tagline shown on every creative (<= 12 words)",
  "localized_messages": { "es": "optional localized tagline", ... },
  "brand": { "colors": ["#RRGGBB", "#RRGGBB"] },
  "products": [
    { "name": "Product name",
      "slug": "lower_snake_case_slug",
      "image_prompt": "a vivid prompt for a GenAI hero image of this product" }
  ]
}
Rules:
- At least TWO products.
- slug: lowercase, words joined by underscores, no spaces.
- localized_messages may be empty {} if no locale is implied.
- brand.colors: 1-3 plausible hex colors for the brand; omit logo.
- Keep it realistic and concise.
"""


def generate_brief(prompt: str, model: str = "claude-sonnet-4-6") -> CampaignBrief:
    """Expand a free-text description into a validated CampaignBrief."""
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. `pip install anthropic` to use prompt-to-brief."
        ) from e
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Set it to use the prompt box, "
            "or pick a ready-made brief instead."
        )

    client = Anthropic()
    system = (
        "You are a marketing operations assistant. You convert a short campaign "
        "description into a structured creative brief as strict JSON. " + _SCHEMA_HINT
    )
    resp = client.messages.create(
        model=model,
        max_tokens=1200,
        system=system,
        messages=[{"role": "user", "content": prompt.strip()}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    # Strip any accidental code fences before parsing.
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    data = json.loads(text)
    return CampaignBrief.model_validate(data)  # raises on malformed output
