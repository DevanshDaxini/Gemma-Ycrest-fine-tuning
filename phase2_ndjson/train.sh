#!/usr/bin/env bash
# Phase 2 training: workout log → NDJSON
# Usage: bash train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "data/train.jsonl" ]; then
    echo "[error] data/train.jsonl not found. Run: python generate_data.py"
    exit 1
fi

N_TRAIN=$(wc -l < data/train.jsonl)

PYTHON="../venv/bin/python"

$PYTHON -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}/../shared')
from mlx_utils import print_training_estimate
print_training_estimate(${N_TRAIN}, 4, 120, 'gemma-3-1b-it', lora_rank=8)
"

echo ""
echo "Starting Phase 2 training..."
echo "Adapters will be saved to: ${SCRIPT_DIR}/adapters/"
echo ""

$PYTHON -m mlx_lm.lora --config lora_config.yaml

echo ""
echo "Training complete. Adapter saved to ./adapters/"
echo "Next: python infer.py --prompt 'Alex 45min...' --mode greedy"
