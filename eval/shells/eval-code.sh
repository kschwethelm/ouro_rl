#!/usr/bin/env bash
# Evaluate Ouro-Thinking on code benchmarks.
#
# All tasks:     ./eval/shells/eval-code.sh
# Smoke test:    LIMIT=20 ./eval/shells/eval-code.sh

set -euo pipefail
source "$(dirname "$0")/eval-common.sh"

SYSTEM_INSTRUCTION="You are a helpful assistant."
TASKS="${TASKS:-mbpp}"

echo "Model:          $MODEL"
echo "Tasks:          $TASKS"
echo ""

run_all "$TASKS" --system_instruction "$SYSTEM_INSTRUCTION"
