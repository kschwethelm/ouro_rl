#!/usr/bin/env bash
# Evaluate Ouro-Thinking on QA benchmarks.
#
# All tasks:     ./shells/eval-qa.sh
# Single task:   TASKS=arc_challenge_chat ./shells/eval-qa.sh
# Smoke test:    LIMIT=20 ./shells/eval-qa.sh

set -euo pipefail
source "$(dirname "$0")/eval-common.sh"

SYSTEM_INSTRUCTION="You are a helpful assistant."
TASKS="${TASKS:-arc_challenge_chat mmlu_flan_cot_zeroshot_stem gpqa_diamond_cot_zeroshot}"

echo "Model:          $MODEL"
echo "Tasks:          $TASKS"
echo "Max gen tokens: $MAX_GEN_TOKS"
echo ""

for task in $TASKS; do
    run_eval "$task" --system_instruction "$SYSTEM_INSTRUCTION"
done
