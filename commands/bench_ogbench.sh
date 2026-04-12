#!/bin/bash
# OGBench benchmark — wraps bench/ogbench/run.sh for consistency with commands/ interface
#
# Requires external OGBench repo (seohongpark/ogbench) separate from this repo.
# Set OGBENCH_DIR env var to specify location (default: $HOME/ogbench).
#
# Usage:
#   bash commands/bench_ogbench.sh                                      # defaults
#   bash commands/bench_ogbench.sh antmaze-large-navigate-v0 agents/gciql.py
#   OGBENCH_DIR=/path/to/ogbench bash commands/bench_ogbench.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OGBENCH_DIR="${OGBENCH_DIR:-$HOME/ogbench}"
ENV_NAME="${1:-antmaze-large-navigate-v0}"
AGENT="${2:-agents/gciql.py}"

if [ ! -d "$OGBENCH_DIR/impls" ]; then
    echo "ERROR: OGBench repo not found at \$OGBENCH_DIR=$OGBENCH_DIR"
    echo "Install with:"
    echo "  git clone https://github.com/seohongpark/ogbench.git \$OGBENCH_DIR"
    echo "  cd \$OGBENCH_DIR && uv venv && uv pip install -e '.[all]'"
    echo "  cd impls && uv pip install -r requirements.txt"
    echo "See CLAUDE.md 'OGBench Setup' section for details."
    exit 1
fi

echo "=== OGBench Benchmark ==="
echo "OGBENCH_DIR: $OGBENCH_DIR"
echo "Env: $ENV_NAME"
echo "Agent: $AGENT"

# Delegate to the existing run.sh (already integrated)
OGBENCH_DIR="$OGBENCH_DIR" bash bench/ogbench/run.sh "$ENV_NAME" "$AGENT"
