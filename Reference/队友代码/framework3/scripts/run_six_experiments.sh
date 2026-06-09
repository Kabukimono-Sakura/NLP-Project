#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] run_six_experiments.sh is kept for compatibility."
echo "[INFO] The project now uses the eight-experiment protocol with baseline + three frameworks."

exec bash framework3/scripts/run_eight_experiments.sh "$@"
