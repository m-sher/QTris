#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"
log="logs/1v1_az_${ts}.log"
pidfile="logs/1v1_az_${ts}.pid"

setsid uv run train placement --mode 1v1 --algo az \
  --num-games 16 --horizon 64 --num-simulations 256 --leaves-per-round 8 \
  --mini-batch-size 512 --max-game-steps 512 --eval-interval 10 --w-b2b 0.06 \
  --td-lambda 0.9 --num-epochs 10 --wandb \
  >"$log" 2>&1 </dev/null &

pid=$!
echo "$pid" >"$pidfile"

echo "Running 1v1 AZ training"
echo "  PID:  $pid   (pidfile: $pidfile)"
echo "  Log:  $log"
