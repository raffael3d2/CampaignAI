"""Storage abstraction.

The task allows Azure/AWS/Dropbox; for a local POC we use the filesystem but
keep it behind a `Storage` interface so the production backend is a drop-in
replacement (see README "Production architecture"). Also handles asset
resolution: reuse an existing asset before spending a GenAI call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from PIL import Image

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


class Storage(ABC):
    @abstractmethod
    def find_asset(self, product_folder: str) -> Optional[Path]: ...

    @abstractmethod
    def save_image(self, img: Image.Image, rel_path: str) -> Path: ...


class LocalStorage(Storage):
    def __init__(self, assets_dir: str | Path, output_dir: str | Path):
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)

    def find_asset(self, product_folder: str) -> Optional[Path]:
        """Return the first existing input asset for a product, else None."""
        product_dir = self.assets_dir / product_folder
        if not product_dir.is_dir():
            return None
        for ext in _IMAGE_EXTS:
            hits = sorted(product_dir.glob(f"*{ext}"))
            if hits:
                return hits[0]
        return None

    def save_image(self, img: Image.Image, rel_path: str) -> Path:
        dest = self.output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(dest, quality=92)
        return dest
