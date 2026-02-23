#!/usr/bin/env bash
# Evaluate Ouro-Thinking on math benchmarks.
#
# All tasks:     ./shells/eval-math.sh
# Single task:   TASKS=gsm8k_thinking ./shells/eval-math.sh
# Smoke test:    LIMIT=20 ./shells/eval-math.sh
# Fine-tuned:    MODEL=path/to/checkpoint ./shells/eval-math.sh

set -euo pipefail
source "$(dirname "$0")/eval-common.sh"

SYSTEM_INSTRUCTION="Solve the math problem. Put your final answer in \\boxed{}."
TASKS="${TASKS:-gsm8k_thinking minerva_math500 aime24}"

echo "Model:          $MODEL"
echo "Tasks:          $TASKS"
echo "Max gen tokens: $MAX_GEN_TOKS"
echo ""

for task in $TASKS; do
    run_eval "$task" --system_instruction "$SYSTEM_INSTRUCTION"
done
