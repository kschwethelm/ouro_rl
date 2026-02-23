#!/usr/bin/env bash
# Evaluate Ouro-Thinking on code benchmarks.
#
# All tasks:     ./shells/eval-code.sh
# Smoke test:    LIMIT=20 ./shells/eval-code.sh

set -euo pipefail
source "$(dirname "$0")/eval-common.sh"

SYSTEM_INSTRUCTION="You are a helpful assistant."
TASKS="${TASKS:-mbpp}"

echo "Model:          $MODEL"
echo "Tasks:          $TASKS"
echo "Max gen tokens: $MAX_GEN_TOKS"
echo ""

for task in $TASKS; do
    run_eval "$task" --system_instruction "$SYSTEM_INSTRUCTION"
done
