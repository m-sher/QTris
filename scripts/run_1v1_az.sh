#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"
log="logs/1v1_az_${ts}.log"
pidfile="logs/1v1_az_${ts}.pid"

# Rebuild the C extension so the .so matches source (editable installs do not
# recompile b2b_search.c on checkout).
echo "Rebuilding tetrisenv C extension..."
uv pip install --force-reinstall --no-deps -e tetrisenv >/dev/null 2>&1

setsid uv run train placement --mode 1v1 --algo az \
  --num-games 16 --horizon 64 --num-simulations 256 --leaves-per-round 8 \
  --batch-size 512 --max-game-steps 512 --eval-interval 10 --w-b2b 0.0 \
  --td-lambda 0.9 --num-epochs 2 --wandb \
  >"$log" 2>&1 </dev/null &

pid=$!
echo "$pid" >"$pidfile"

echo "Running 1v1 AZ training"
echo "  PID:  $pid   (pidfile: $pidfile)"
echo "  Log:  $log"
