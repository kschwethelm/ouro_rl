#!/usr/bin/env bash
# Evaluate Ouro-Thinking on math benchmarks.
#
# All tasks:     ./eval/shells/eval-math.sh
# Pick tasks:    TASKS=gsm8k_thinking,aime24_thinking ./eval/shells/eval-math.sh
# Smoke test:    LIMIT=20 ./eval/shells/eval-math.sh
# Fine-tuned:    MODEL=path/to/checkpoint ./eval/shells/eval-math.sh
# + pass@10:     TASKS=gsm8k_thinking,aime24_thinking,aime24_thinking_pass10 ./eval/shells/eval-math.sh

set -euo pipefail
source "$(dirname "$0")/eval-common.sh"

SYSTEM_INSTRUCTION="Solve the math problem. Put your final answer in \\boxed{}."
TASKS="${TASKS:-gsm8k_thinking,math500_thinking,aime24_thinking}"

echo "Model:          $MODEL"
echo "Tasks:          $TASKS"
echo ""

run_all "$TASKS" --system_instruction "$SYSTEM_INSTRUCTION"
