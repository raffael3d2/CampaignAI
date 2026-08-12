"""Local web UI server for the creative automation pipeline.

Wraps the existing pipeline and streams each stage to the browser over
Server-Sent Events (SSE), so you watch the run happen live: brief parsed,
asset resolved (reused vs generated), each aspect ratio composited, checks run.

This is a thin presentation layer — all real work still lives in src/pipeline.
Run:  python -m src.web.server   then open http://localhost:5000
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import queue
import sys
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.brief import load_brief  # noqa: E402
from src.pipeline.checks import brand_color_check, legal_check  # noqa: E402
from src.pipeline.compositor import ASPECT_RATIOS, build_creative  # noqa: E402
from src.pipeline.providers import get_provider  # noqa: E402
from src.pipeline.storage import LocalStorage  # noqa: E402

log = logging.getLogger("web")
app = Flask(__name__, static_folder=None)

STATIC_DIR = Path(__file__).resolve().parent / "static"
BRIEFS_DIR = ROOT / "briefs"

# Local settings file so a recruiter can paste keys once and have them persist.
# It is gitignored (keys never get committed) and lives beside the project.
CONFIG_PATH = ROOT / ".studio_config.json"
DEFAULT_OUTPUT = ROOT / "output"


def load_config() -> dict:
    """Read persisted settings. Missing/corrupt file -> empty config."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def apply_config_to_env(cfg: dict) -> None:
    """Push saved tokens into the environment for this process, so the
    providers (which read os.environ) pick them up without any other change."""
    if cfg.get("gemini_key"):
        os.environ["GEMINI_API_KEY"] = cfg["gemini_key"].strip()
    if cfg.get("anthropic_key"):
        os.environ["ANTHROPIC_API_KEY"] = cfg["anthropic_key"].strip()


def output_dir_from_config(cfg: dict) -> Path:
    """Resolve the output folder: user setting if present, else default."""
    folder = (cfg.get("output_dir") or "").strip()
    return Path(folder).expanduser() if folder else DEFAULT_OUTPUT


# Where builder-picked / uploaded / generated hero images live.
WORKSPACE = ROOT / "assets" / "_workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _list_asset_images() -> list[dict]:
    """All usable images under assets/ (excluding the brand logo), for the pickers."""
    items = []
    for p in sorted(ROOT.glob("assets/**/*")):
        if p.suffix.lower() in _IMG_EXTS and p.name != "logo.png":
            rel = p.relative_to(ROOT).as_posix()
            items.append({"id": rel, "name": p.parent.name + "/" + p.name})
    return items


def _resolve_image_id(image_id: str) -> Path | None:
    """Map an image id (a repo-relative path under assets/) to a real file,
    guarding against path traversal."""
    if not image_id:
        return None
    p = (ROOT / image_id).resolve()
    try:
        p.relative_to((ROOT / "assets").resolve())
    except ValueError:
        return None
    return p if p.exists() else None


def _thumb(img: Image.Image, max_side: int = 480) -> str:
    """Return a base64 data URL for a downscaled preview of the image."""
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _img_to_png_bytes(img: Image.Image) -> bytes:
    """Serialize a PIL image to PNG bytes (for passing as a Gemini reference)."""
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/test_connection")
def test_connection():
    """Fire one tiny Gemini generation to verify the API key works, and report
    a clear ✓/✗ so a user can confirm setup before running a full campaign."""
    apply_config_to_env(load_config())
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return jsonify({
            "ok": False,
            "message": "No Gemini API key saved. Paste one above and Save first.",
        }), 400
    try:
        provider = get_provider("gemini")
        img = provider.generate("a single red apple on a plain white background")
        _ = img.size  # force materialization
        return jsonify({"ok": True, "message": "Connection works — Gemini generated a test image."})
    except Exception as e:
        msg = str(e)
        hint = ""
        low = msg.lower()
        if "401" in msg or "403" in msg or "unauthorized" in low or "permission" in low or "api key" in low or "invalid" in low:
            hint = " The API key was rejected — check it at https://aistudio.google.com/apikey."
        elif "429" in msg or "quota" in low or "rate" in low or "exhausted" in low:
            hint = " Rate limit / quota hit — wait a moment and try again."
        elif "billing" in low or "payment" in low:
            hint = " This model may require billing enabled on your Google project."
        return jsonify({"ok": False, "message": "Test failed: " + msg[:180] + hint}), 200


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/assets")
def list_assets():
    """Images the pickers can choose from (folder option)."""
    return jsonify(_list_asset_images())


@app.get("/api/asset_image")
def asset_image():
    """Serve a thumbnail data URL for a given asset id, for previews."""
    p = _resolve_image_id(request.args.get("id", ""))
    if not p:
        return jsonify({"error": "not found"}), 404
    return jsonify({"thumb": _thumb(Image.open(p))})


@app.post("/api/upload_image")
def upload_image():
    """Accept an uploaded image (multipart) and store it in the workspace.
    Returns an image id the pickers/run can use."""
    slot = (request.form.get("slot") or "asset").strip()
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file uploaded."}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in _IMG_EXTS:
        return jsonify({"error": "Please upload a PNG, JPG, or WEBP image."}), 400
    try:
        img = Image.open(f.stream).convert("RGBA")
    except Exception:
        return jsonify({"error": "That file isn't a readable image."}), 400
    dest = WORKSPACE / f"{slot}_upload.png"
    img.save(dest)
    rel = dest.relative_to(ROOT).as_posix()
    return jsonify({"id": rel, "thumb": _thumb(img)})


@app.post("/api/generate_asset")
def generate_asset():
    """Generate ONE image via Gemini for a slot (person/product) from a prompt,
    store it in the workspace, and return its id + preview. Returns a clear
    error if the Gemini key is missing/invalid."""
    apply_config_to_env(load_config())
    payload = request.get_json(silent=True) or {}
    slot = (payload.get("slot") or "asset").strip()
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Describe what to generate first."}), 400
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return jsonify({"error": "No Gemini API key set — add one in Setup & keys, then Save."}), 400
    try:
        provider = get_provider("gemini")
        img = provider.generate(prompt)
    except Exception as e:
        return jsonify({"error": "Gemini generation failed: " + str(e)[:180]}), 200
    dest = WORKSPACE / f"{slot}_generated.png"
    img.convert("RGB").save(dest)
    rel = dest.relative_to(ROOT).as_posix()
    return jsonify({"id": rel, "thumb": _thumb(img)})


@app.post("/api/clear_cache")
def clear_cache():
    """Delete cached generated heroes (and prior output) so the next run
    regenerates fresh images instead of reusing what's on disk."""
    import shutil
    cfg = load_config()
    out_dir = output_dir_from_config(cfg)
    removed = 0
    # Remove every campaign's _cache folder, plus whole prior outputs.
    try:
        if out_dir.exists():
            for cache in out_dir.glob("**/_cache"):
                if cache.is_dir():
                    shutil.rmtree(cache, ignore_errors=True)
                    removed += 1
            # also clear finished creatives so stale renders don't linger
            for child in out_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
    except Exception as e:
        return jsonify({"ok": False, "message": f"Could not clear cache: {e}"}), 200
    return jsonify({"ok": True, "message": f"Cache cleared — next run regenerates fresh. ({removed} folder(s) removed)"})


@app.get("/api/config")
def get_config():
    """Return current settings for the UI. Tokens are masked, never sent raw:
    the UI only needs to know whether each is set, not its value."""
    cfg = load_config()
    def mask(v: str | None) -> str:
        if not v:
            return ""
        v = v.strip()
        return (v[:6] + "…" + v[-4:]) if len(v) > 12 else "set"
    return jsonify({
        "gemini_key_set": bool(cfg.get("gemini_key")),
        "gemini_key_masked": mask(cfg.get("gemini_key")),
        "anthropic_key_set": bool(cfg.get("anthropic_key")),
        "anthropic_key_masked": mask(cfg.get("anthropic_key")),
        "output_dir": cfg.get("output_dir", ""),
        "default_output": str(DEFAULT_OUTPUT),
    })


@app.post("/api/config")
def set_config():
    """Persist settings from the UI. Only overwrites a token when a new,
    non-empty value is supplied — so re-saving other fields won't wipe a key.
    Sending the sentinel '__clear__' explicitly clears a field."""
    incoming = request.get_json(silent=True) or {}
    cfg = load_config()

    for key, field in (("gemini_key", "gemini_key"),
                        ("anthropic_key", "anthropic_key")):
        val = incoming.get(field)
        if val is None:
            continue
        val = val.strip()
        if val == "__clear__":
            cfg.pop(key, None)
        elif val:
            cfg[key] = val

    if "output_dir" in incoming:
        cfg["output_dir"] = (incoming.get("output_dir") or "").strip()

    save_config(cfg)
    apply_config_to_env(cfg)
    return get_config()


@app.get("/api/briefs")
def list_briefs():
    """List available briefs with a short summary for the picker."""
    items = []
    for p in sorted(BRIEFS_DIR.glob("*")):
        if p.suffix.lower() not in (".yaml", ".yml", ".json"):
            continue
        try:
            brief = load_brief(p)
            items.append({
                "file": p.name,
                "campaign": brief.campaign_name,
                "region": brief.target_region,
                "audience": brief.target_audience,
                "message": brief.campaign_message,
                "products": [pr.name for pr in brief.products],
                "locales": list(brief.localized_messages.keys()),
            })
        except Exception as e:
            log.warning("Skipping %s: %s", p.name, e)
    return jsonify(items)


# In-memory store for briefs generated from a prompt this session.
_GENERATED: dict[str, "CampaignBrief"] = {}


@app.post("/api/generate_brief")
def generate_brief_endpoint():
    """Turn a free-text prompt into a structured brief and return it as JSON.

    The brief is cached under a token the client passes back to /api/run, so the
    same validated object is used for the actual run (no re-generation, no drift).
    """
    from src.pipeline.brief_gen import generate_brief  # local import: optional dep

    apply_config_to_env(load_config())
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Please enter a description of the campaign."}), 400
    try:
        brief = generate_brief(prompt)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    token = f"gen_{abs(hash(prompt)) % 10_000_000}"
    _GENERATED[token] = brief
    return jsonify({
        "token": token,
        "campaign": brief.campaign_name,
        "region": brief.target_region,
        "audience": brief.target_audience,
        "message": brief.campaign_message,
        "products": [p.name for p in brief.products],
        "locales": list(brief.localized_messages.keys()),
    })


import yaml as _yaml  # for parsing the builder brief textarea

_BUILDER: dict[str, dict] = {}  # token -> {brief, heroes}


@app.post("/api/builder_brief")
def builder_brief():
    """Build a 2-product campaign from the builder page: region/audience/message
    (typed or pasted as JSON/YAML) + a person image and a product image chosen
    as the two heroes. Returns a token /api/run consumes."""
    from src.pipeline.brief import CampaignBrief, Product

    payload = request.get_json(silent=True) or {}

    # The brief fields may arrive as discrete values or as a pasted blob.
    region = (payload.get("target_region") or "").strip()
    audience = (payload.get("target_audience") or "").strip()
    message = (payload.get("campaign_message") or "").strip()
    blob = (payload.get("brief_text") or "").strip()
    if blob:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            try:
                parsed = _yaml.safe_load(blob) or {}
            except Exception:
                return jsonify({"error": "Brief text isn't valid JSON or YAML."}), 400
        if isinstance(parsed, dict):
            region = region or str(parsed.get("target_region", "")).strip()
            audience = audience or str(parsed.get("target_audience", "")).strip()
            message = message or str(parsed.get("campaign_message", "")).strip()

    if not message:
        return jsonify({"error": "Campaign message is required."}), 400

    person_id = payload.get("person_image") or ""
    product_id = payload.get("product_image") or ""
    logo_id = payload.get("logo_image") or ""
    person_path = _resolve_image_id(person_id)
    product_path = _resolve_image_id(product_id)
    logo_path_ref = _resolve_image_id(logo_id)  # optional
    if not person_path or not product_path:
        return jsonify({"error": "Pick or generate both a person and a product image first."}), 400

    # Plain-language descriptions of what each image shows — sharpen the prompt.
    person_desc = (payload.get("person_desc") or "the person").strip()
    product_desc = (payload.get("product_desc") or "the product").strip()

    # Gemini (Nano Banana) generates a finished ad, conditioned on the person
    # and product (and brand logo, if provided) reference images.
    logo_clause = (
        " Incorporate the brand logo from the final reference image tastefully "
        "in a corner." if logo_path_ref else ""
    )
    ad_prompt = (
        f"Professional commercial advertising photograph combining the reference images: "
        f"{person_desc} naturally using and presenting {product_desc}. "
        f"Polished lifestyle ad scene for {audience or 'a broad audience'} in "
        f"{region or 'a global market'}. Studio-grade lighting, sharp focus, premium brand "
        f"aesthetic, clean composition with space for a headline. High-end product photography." + logo_clause
    )

    brief = CampaignBrief(
        campaign_name=(payload.get("campaign_name") or "Custom Campaign").strip(),
        target_region=region or "—",
        target_audience=audience or "—",
        campaign_message=message,
        products=[
            Product(name="Ad (person + product)", slug="ad", image_prompt=ad_prompt),
            Product(name="Product only", slug="product"),
        ],
    )
    # Reference images fed to Gemini for the composed ad (logo appended if present).
    ad_refs = [str(person_path), str(product_path)]
    if logo_path_ref:
        ad_refs.append(str(logo_path_ref))
    token = f"build_{abs(hash((region, audience, message, person_id, product_id, logo_id))) % 10_000_000}"
    _BUILDER[token] = {
        "brief": brief,
        # 'ad' has no hero override → Gemini generates it from ad_prompt + refs.
        # 'product' uses the picked product image directly.
        "heroes": {"product": str(product_path)},
        "ad_prompt": ad_prompt,
        "references": {"ad": ad_refs},
    }
    return jsonify({
        "token": token,
        "campaign": brief.campaign_name,
        "region": brief.target_region,
        "audience": brief.target_audience,
        "message": brief.campaign_message,
    })


@app.get("/api/run")
def run_stream():
    """Run the pipeline for a brief, streaming each stage as an SSE event.

    Accepts either ?brief=<file> (a saved brief) or ?token=<id> (a brief just
    generated from a prompt).
    """
    brief_file = request.args.get("brief", "")
    token = request.args.get("token", "")
    builder_token = request.args.get("builder", "")
    provider_name = request.args.get("provider", "gemini")
    locale = request.args.get("locale") or None

    def generate():
        hero_overrides: dict[str, str] = {}
        references: dict[str, list[str]] = {}
        # Resolve the brief from a builder token, a generated token, or a file.
        if builder_token:
            entry = _BUILDER.get(builder_token)
            if entry is None:
                yield _sse("error", {"message": "Builder session expired; rebuild and run again."})
                return
            brief = entry["brief"]
            hero_overrides = entry["heroes"]
            references = entry.get("references", {})
        elif token:
            brief = _GENERATED.get(token)
            if brief is None:
                yield _sse("error", {"message": "Generated brief expired; please re-generate."})
                return
        else:
            try:
                brief = load_brief(BRIEFS_DIR / Path(brief_file).name)  # no path traversal
            except Exception as e:
                yield _sse("error", {"message": f"Could not load brief: {e}"})
                return

        message = brief.localized_messages.get(locale, brief.campaign_message) \
            if locale else brief.campaign_message

        yield _sse("brief", {
            "campaign": brief.campaign_name,
            "region": brief.target_region,
            "audience": brief.target_audience,
            "message": message,
            "locale": locale or "en",
            "products": [p.name for p in brief.products],
            "ratios": list(ASPECT_RATIOS.keys()),
        })

        # Legal check is campaign-wide.
        legal = legal_check(message)
        yield _sse("legal", legal)

        try:
            provider = get_provider(provider_name)
        except Exception as e:
            yield _sse("error", {"message": str(e)})
            return

        cfg = load_config()
        apply_config_to_env(cfg)
        out_dir = output_dir_from_config(cfg)
        storage = LocalStorage(assets_dir=ROOT / "assets", output_dir=out_dir)
        logo = brief.brand.logo
        logo_path = str(ROOT / logo) if logo else None

        for product in brief.products:
            yield _sse("product_start", {"product": product.name})

            # --- Stage: resolve hero (builder override > reuse > generate) ---
            override = hero_overrides.get(product.folder)
            if override and Path(override).exists():
                hero = Image.open(override).convert("RGBA")
                source = "chosen"
                detail = Path(override).name
            else:
                existing = storage.find_asset(product.folder)
                if existing:
                    hero = Image.open(existing).convert("RGBA")
                    source = "reused"
                    detail = str(existing.relative_to(ROOT))
                else:
                    cache = out_dir / brief.slug / "_cache" / f"{product.folder}.png"
                    if cache.exists():
                        hero = Image.open(cache).convert("RGBA")
                        source, detail = "cached", "previously generated"
                    else:
                        prompt = product.image_prompt or (
                            f"Premium social ad hero image of {product.name}, "
                            f"for {brief.target_audience} in {brief.target_region}"
                        )
                        yield _sse("hero_generating", {"product": product.name, "prompt": prompt})
                        # Load reference images (person + product) for this slot,
                        # so Gemini composes them into the ad.
                        ref_bytes = []
                        for rp in references.get(product.folder, []):
                            try:
                                ref_bytes.append(Path(rp).read_bytes())
                            except Exception:
                                pass
                        try:
                            hero = provider.generate(prompt, refs=ref_bytes or None,
                                                     aspect_ratio="1:1")
                        except Exception as e:
                            yield _sse("error", {
                                "message": f"Image generation failed for '{product.name}': {e}. "
                                           f"Check your Gemini API key in Setup & keys, or try again."
                            })
                            return
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        hero.convert("RGB").save(cache)
                        source, detail = "generated", prompt

            yield _sse("hero_ready", {
                "product": product.name,
                "source": source,
                "detail": detail,
                "thumb": _thumb(hero),
            })

            # Gemini aspect-ratio strings for native per-ratio rendering.
            _GEMINI_AR = {"1x1": "1:1", "9x16": "9:16", "16x9": "16:9"}

            # --- Stage: composite each aspect ratio ---
            # For a GENERATED ad, render each ratio natively with Gemini so the
            # person/product/logo are never cropped: the 1:1 hero is the base,
            # and 9:16 / 16:9 are re-composed from it (passed as a reference)
            # with a reframe prompt that extends the scene instead of cropping.
            ratio_refs = references.get(product.folder, [])
            for ratio_name in ASPECT_RATIOS:
                base = hero  # default: the hero we already have
                if source == "generated":
                    if ratio_name == "1x1":
                        base = hero  # hero is already the square master
                    else:
                        ar = _GEMINI_AR[ratio_name]
                        orient = "vertical" if ratio_name == "9x16" else "widescreen horizontal"
                        reframe_prompt = (
                            f"Reframe this advertising image to a {orient} {ar} composition. "
                            f"Keep the person, product, and logo fully visible and uncropped; "
                            f"naturally extend the scene and background to fill the new format. "
                            f"Preserve the subject's appearance, the product, and the brand style."
                        )
                        yield _sse("hero_generating", {
                            "product": product.name,
                            "prompt": f"Reframing to {ar} (keeping subject in frame)…",
                        })
                        try:
                            # Pass the 1:1 master as the primary reference, plus
                            # the original references so identity stays consistent.
                            reframe_refs = [_img_to_png_bytes(hero)]
                            for rp in ratio_refs:
                                try:
                                    reframe_refs.append(Path(rp).read_bytes())
                                except Exception:
                                    pass
                            base = provider.generate(
                                reframe_prompt, refs=reframe_refs,
                                aspect_ratio=ar,
                            )
                        except Exception as e:
                            # If a reframe fails, fall back to the crop so the run
                            # still completes rather than aborting.
                            log.warning("Reframe to %s failed (%s); using crop.", ar, e)
                            base = hero

                creative = build_creative(base, ratio_name, message, logo_path)
                rel = f"{brief.slug}/{product.folder}/{ratio_name}/creative.jpg"
                saved = storage.save_image(creative, rel)
                brand = brand_color_check(saved, brief.brand.colors)
                yield _sse("creative", {
                    "product": product.name,
                    "ratio": ratio_name,
                    "thumb": _thumb(creative),
                    "path": rel,
                    "brand_check": brand,
                })

            yield _sse("product_done", {"product": product.name})

        yield _sse("done", {"output_dir": str(out_dir / brief.slug)})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    apply_config_to_env(load_config())  # pick up saved keys on launch
    print("\n  Creative Automation Studio  →  http://localhost:5000\n")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
