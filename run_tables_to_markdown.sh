#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root based on this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_FILE="$SCRIPT_DIR/utils/tables_to_markdown.py"

if [[ ! -f "$PY_FILE" ]]; then
  echo "Error: $PY_FILE was not found. Update the script if the file moved." >&2
  read -rp "Press Enter to close..."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not available in PATH." >&2
  read -rp "Press Enter to close..."
  exit 1
fi

python3 "$PY_FILE"

# Keep the terminal window open when launched via double-click
read -rp "\nRun complete. Press Enter to close..."