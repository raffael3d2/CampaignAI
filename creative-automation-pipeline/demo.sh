#!/usr/bin/env bash
# Demo script for the 2-3 minute walkthrough video.
# Runs the full pipeline on both briefs and shows the key behaviors.
set -e

echo "=================================================="
echo " Creative Automation Pipeline — Demo"
echo "=================================================="

echo
echo ">>> 1. Clean previous output"
rm -rf output && echo "output/ cleared"

echo
echo ">>> 2. Run the YAML campaign (mock GenAI, offline, no API key)"
echo ">>>    Aura Glow REUSES a supplied asset; Pure Hydrate is GENERATED"
python3 -m src.cli run briefs/skincare_launch.yaml --provider mock

echo
echo ">>> 3. Same campaign, localized to Spanish (--locale es)"
python3 -m src.cli run briefs/skincare_launch.yaml --provider mock --locale es

echo
echo ">>> 4. JSON brief — legal check flags a prohibited word"
python3 -m src.cli run briefs/energy_drink.json --provider mock || true

echo
echo ">>> 5. Output organized by product and aspect ratio:"
find output -type f -name "*.jpg" | sort

echo
echo ">>> 6. Sample run report:"
cat output/peak_fuel_q3/report.json | head -30

echo
echo "=================================================="
echo " Demo complete. Open output/ to view creatives."
echo "=================================================="
