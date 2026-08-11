"""GenAI image provider abstraction.

The pipeline hides the model behind an `ImageProvider` interface so it never
depends on a specific vendor. This build uses Google's Gemini "Nano Banana"
image model exclusively — it supports reference-image-conditioned generation
(e.g. a person image + a product image composed into one finished ad photo),
which text-to-image models cannot do.
"""
from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod

from PIL import Image


class ImageProvider(ABC):
    """Generates a hero image (returned as a PIL RGBA Image).

    generate() takes the text prompt and, optionally, a list of reference
    images (raw bytes) the model should condition on — e.g. a person and a
    product to combine into one ad — plus an optional aspect ratio.
    """

    supports_references: bool = False

    @abstractmethod
    def generate(self, prompt: str, size: int = 1024,
                 refs: list[bytes] | None = None,
                 aspect_ratio: str | None = None) -> Image.Image: ...


class GeminiProvider(ImageProvider):
    """Google Gemini "Nano Banana" image generation via the google-genai SDK.

    Reference-conditioned: pass person/product images in `refs` and the model
    composes them into the scene described by the prompt. Renders at 2K.

    Auth: set GEMINI_API_KEY (or GOOGLE_API_KEY) — get one free at
    https://aistudio.google.com/apikey . Model overridable via GEMINI_IMAGE_MODEL.
    """

    supports_references = True

    def __init__(self, model: str | None = None):
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "google-genai not installed. Run: pip install google-genai"
            ) from e
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/apikey , then add it in Setup & keys "
                "(or export GEMINI_API_KEY=...)."
            )
        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=key)
        self._model = model or os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")

    def generate(self, prompt: str, size: int = 1024,
                 refs: list[bytes] | None = None,
                 aspect_ratio: str | None = None) -> Image.Image:
        # Build the multimodal input: the prompt text, then any reference images.
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for raw in (refs or []):
            parts.append({
                "type": "image",
                "data": base64.b64encode(raw).decode("utf-8"),
                "mime_type": "image/png",
            })

        response_format = {"type": "image", "image_size": "2K"}
        if aspect_ratio:
            response_format["aspect_ratio"] = aspect_ratio

        interaction = self._client.interactions.create(
            model=self._model,
            input=parts,
            response_format=response_format,
        )
        img_b64 = interaction.output_image.data
        raw = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(raw)).convert("RGBA")


def get_provider(name: str = "gemini") -> ImageProvider:
    """Only Gemini (Nano Banana) is available in this build."""
    name = (name or "gemini").lower()
    if name in ("gemini", "nano-banana", "nano_banana", "google"):
        return GeminiProvider()
    raise ValueError(
        f"Unknown provider: {name}. This build uses Gemini (Nano Banana) only."
    )
