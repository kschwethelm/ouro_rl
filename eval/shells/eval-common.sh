# Shared config for Ouro-Thinking evaluation scripts.
# Source this from eval-math.sh, eval-qa.sh, etc. — do not run directly.
#
# Override any variable via environment:
#   MODEL=path/to/checkpoint ./eval/shells/eval-math.sh
#   TP=2 LIMIT=50 ./eval/shells/eval-math.sh

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-ByteDance/Ouro-1.4B-Thinking}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/outputs}"
LIMIT="${LIMIT:-}"
LOG_SAMPLES="${LOG_SAMPLES:-true}"

# --- Patched tokenizer ---
TOKENIZER_DIR="models/tokenizer"
if [[ ! -d "$TOKENIZER_DIR" ]]; then
    echo "Setting up patched tokenizer..."
    uv run python eval/scripts/setup_tokenizer.py --model "$MODEL" --output "$TOKENIZER_DIR"
fi

# --- vLLM model args ---
MODEL_ARGS="pretrained=$MODEL,tokenizer=$TOKENIZER_DIR,trust_remote_code=True,dtype=bfloat16,max_model_len=$MAX_MODEL_LEN,tensor_parallel_size=$TP,enable_thinking=True,think_end_token=</think>"

LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

LOG_SAMPLES_ARG=""
if [[ "$LOG_SAMPLES" == "true" ]]; then
    LOG_SAMPLES_ARG="--log_samples"
fi

mkdir -p "$OUTPUT_DIR"

# --- Run a single lm-eval task ---
# Usage: run_eval <task> [extra lm_eval args...]
run_eval() {
    local task="$1"
    shift
    echo "=== $task ==="

    uv run lm_eval \
        --model vllm \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size auto \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --include_path eval/tasks \
        --output_path "$OUTPUT_DIR" \
        $LOG_SAMPLES_ARG \
        $LIMIT_ARG \
        "$@"
}

# --- Run tasks, with automatic pass@k aggregation ---
# Usage: run_all <comma-separated-tasks> [extra lm_eval args...]
# Tasks matching *_pass<k> are detected automatically: run with --log_samples
# and compute_pass_at_k.py is called after.
run_all() {
    local tasks="$1"
    shift

    # Split into regular tasks and pass@k tasks.
    local regular=() pass_tasks=()
    IFS=',' read -ra task_list <<< "$tasks"
    for task in "${task_list[@]}"; do
        if [[ "$task" =~ _pass([0-9]+)$ ]]; then
            pass_tasks+=("$task")
        else
            regular+=("$task")
        fi
    done

    # Run regular tasks.
    if [[ ${#regular[@]} -gt 0 ]]; then
        local joined
        joined=$(IFS=','; echo "${regular[*]}")
        run_eval "$joined" "$@"
    fi

    # Run pass@k tasks with --log_samples and aggregate.
    for task in "${pass_tasks[@]}"; do
        local k="${task##*_pass}"
        run_eval "$task" --log_samples "$@"

        local samples
        samples=$(ls -t "$OUTPUT_DIR"/*/samples_${task}_*.jsonl 2>/dev/null | head -1)
        if [[ -n "$samples" ]]; then
            uv run python eval/scripts/compute_pass_at_k.py "$samples" --k "$k"
        fi
    done
}
