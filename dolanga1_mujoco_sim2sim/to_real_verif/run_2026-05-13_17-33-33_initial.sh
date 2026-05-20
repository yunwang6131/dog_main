#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/sen/miniconda3/envs/dolang1/bin/python}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/verify_bundle.py" "$@"
