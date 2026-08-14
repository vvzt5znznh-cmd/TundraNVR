#!/usr/bin/env bash
# Pull code updates without re-cloning or re-downloading models.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .git ]]; then
  echo "This folder is not a git clone. Clone once, then run this script from that copy." >&2
  exit 1
fi

echo "Pulling latest code in $ROOT"
git pull --ff-only

if [[ -x .venv/bin/python ]]; then
  echo "Keeping existing .venv, YOLO weights (*.pt), and data/"
  echo "If requirements.txt changed, run:"
  echo "  source .venv/bin/activate && pip install -r requirements.txt"
else
  echo "No .venv yet — follow README Setup once."
fi

echo
echo "Restart the app (models stay on disk):"
echo "  source .venv/bin/activate"
echo "  python -m app.main"
echo
echo "Source/model tweaks: use the live page Apply button, not a new clone."
