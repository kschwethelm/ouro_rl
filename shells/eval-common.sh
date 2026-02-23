# Shared config for Ouro-Thinking evaluation scripts.
# Source this from eval-math.sh, eval-qa.sh, etc. — do not run directly.
#
# Override any variable via environment:
#   MODEL=path/to/checkpoint ./shells/eval-math.sh
#   TP=2 LIMIT=50 ./shells/eval-math.sh

cd "$(dirname "$0")/.."

MODEL="${MODEL:-ByteDance/Ouro-1.4B-Thinking}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16896}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-16384}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval}"
LIMIT="${LIMIT:-}"
LOG_SAMPLES="${LOG_SAMPLES:-false}"

# --- Patched tokenizer ---
TOKENIZER_DIR="models/tokenizer"
if [[ ! -d "$TOKENIZER_DIR" ]]; then
    echo "Setting up patched tokenizer..."
    uv run python scripts/setup_tokenizer.py --model "$MODEL" --output "$TOKENIZER_DIR"
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
        --include_path eval-tasks \
        --output_path "$OUTPUT_DIR" \
        --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0" \
        $LOG_SAMPLES_ARG \
        $LIMIT_ARG \
        "$@"
}
