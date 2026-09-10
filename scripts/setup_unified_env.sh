#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$UV_BIN" ]]; then
  python3.10 -m pip install --user uv
fi

cd "$ROOT"
if [[ ! -d .venv ]]; then
  "$UV_BIN" venv .venv --python /usr/bin/python3.10
fi

"$UV_BIN" pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match -r requirements-unified.txt
"$UV_BIN" pip install --python .venv/bin/python --no-deps \
  -e models/co-tracker \
  -e models/visualnav-transformer/train \
  -e models/habitat-lab/habitat-lab

# Habitat-Sim 0.3.3 uses old CMake policies and invokes pip internally.
HEADLESS=1 WITH_BULLET=0 CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-8}" \
  CMAKE_ARGS='-DCMAKE_POLICY_VERSION_MINIMUM=3.5' \
  "$UV_BIN" pip install --python .venv/bin/python --no-build-isolation \
  models/habitat-sim

echo "Unified environment ready: $ROOT/.venv"
