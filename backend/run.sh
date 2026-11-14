#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run.sh  –  Start the RAG Research Assistant backend
#
# Usage:
#   chmod +x run.sh
#   ./run.sh              # development (hot-reload)
#   ./run.sh prod         # production  (4 workers)
# ─────────────────────────────────────────────────────────────

# Always resolve to the directory this script lives in,
# so it works regardless of where you call it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add project root to PYTHONPATH so `api`, `core`, `models` are importable.
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "PYTHONPATH set to: $PYTHONPATH"

if [ "$1" = "prod" ]; then
    echo "🚀 Starting in PRODUCTION mode..."
    uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 4 \
        --app-dir "$SCRIPT_DIR"
else
    echo "🔧 Starting in DEVELOPMENT mode (hot-reload)..."
    uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --app-dir "$SCRIPT_DIR"
fi
