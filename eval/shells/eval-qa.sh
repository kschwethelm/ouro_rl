#!/usr/bin/env bash
# Evaluate Ouro-Thinking on QA benchmarks.
#
# All tasks:     LOG_SAMPLES=true ./eval/shells/eval-qa.sh
# Pick tasks:    TASKS=gpqa_main_thinking ./eval/shells/eval-qa.sh
# Smoke test:    LIMIT=20 ./eval/shells/eval-qa.sh

set -euo pipefail
source "$(dirname "$0")/eval-common.sh"

SYSTEM_INSTRUCTION="Answer the multiple choice question. Output only the correct option letter, e.g. (C). No explanation."
TASKS="${TASKS:-gpqa_main_thinking}"

echo "Model:          $MODEL"
echo "Tasks:          $TASKS"
echo ""

run_all "$TASKS" --system_instruction "$SYSTEM_INSTRUCTION"
