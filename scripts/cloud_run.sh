#!/usr/bin/env bash
# Wrapper script for GitHub Cloud Notebook Runner
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$DIR")"

# Prefer virtual environment python if present
if [ -f "$REPO_ROOT/venv/bin/python3" ]; then
    PY="$REPO_ROOT/venv/bin/python3"
else
    PY="python3"
fi

exec "$PY" "$DIR/cloud_run.py" "$@"
