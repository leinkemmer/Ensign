#!/bin/bash
# Builds the API reference (Doxygen) and serves the documentation locally.
# Usage: docs/serve.sh [build]   ("build" only writes the static site to site/)
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${DOCS_VENV:-$HOME/.cache/ensign-docs-venv}"
cd "$ROOT"

if ! command -v doxygen >/dev/null; then
  echo "doxygen not found. Install it with: sudo apt install doxygen" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/mkdocs" ]; then
  echo "Creating Python environment in $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r docs/requirements.txt
fi

doxygen Doxyfile

if [ "$1" = "build" ]; then
  "$VENV/bin/mkdocs" build --strict
  echo "Site written to $ROOT/site (open site/index.html)"
else
  "$VENV/bin/mkdocs" serve
fi
