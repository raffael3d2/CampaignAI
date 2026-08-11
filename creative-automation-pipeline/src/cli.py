"""Command-line entrypoint for the creative automation pipeline.

Examples:
    python -m src.cli run briefs/skincare_launch.yaml
    python -m src.cli run briefs/skincare_launch.yaml --provider openai --locale es
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

# Allow `python -m src.cli` and `python src/cli.py` to both work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.brief import load_brief  # noqa: E402
from src.pipeline.pipeline import Pipeline  # noqa: E402
from src.pipeline.providers import get_provider  # noqa: E402
from src.pipeline.storage import LocalStorage  # noqa: E402

app = typer.Typer(add_completion=False, help="GenAI creative automation pipeline.",
                  pretty_exceptions_show_locals=False)

ROOT = Path(__file__).resolve().parent.parent


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def run(
    brief_path: str = typer.Argument(..., help="Path to campaign brief (.yaml or .json)"),
    provider: str = typer.Option("gemini", help="Image provider: 'gemini' (Nano Banana, requires GEMINI_API_KEY)"),
    assets: str = typer.Option(str(ROOT / "assets"), help="Input assets directory"),
    output: str = typer.Option(str(ROOT / "output"), help="Output directory"),
    locale: str = typer.Option(None, help="Locale key for localized message, e.g. 'es'"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate creatives for every product x aspect ratio in the brief."""
    _setup_logging(verbose)
    log = logging.getLogger("cli")

    try:
        brief = load_brief(brief_path)
    except Exception as e:
        log.error("Failed to load brief: %s", e)
        raise typer.Exit(code=1)

    log.info("Loaded campaign '%s' with %d product(s)",
             brief.campaign_name, len(brief.products))

    try:
        img_provider = get_provider(provider)
    except Exception as e:
        log.error("Provider error: %s", e)
        raise typer.Exit(code=1)

    storage = LocalStorage(assets_dir=assets, output_dir=output)
    pipeline = Pipeline(provider=img_provider, storage=storage)

    try:
        report = pipeline.run(brief, locale=locale)
    except Exception as e:
        log.error("Pipeline failed: %s", e)
        raise typer.Exit(code=1)

    n_creatives = sum(len(p["creatives"]) for p in report["products"])
    log.info("Done. %d creatives across %d products -> %s",
             n_creatives, len(report["products"]), Path(output) / brief.slug)
    if not report["legal_check"]["passed"]:
        log.warning("REVIEW NEEDED: legal check flagged %s",
                    report["legal_check"]["flagged_terms"])


@app.command(hidden=True)
def version():
    """Print version (also forces `run` to remain a named subcommand)."""
    typer.echo("creative-automation-pipeline 1.0.0")


if __name__ == "__main__":
    app()
