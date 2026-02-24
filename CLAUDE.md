# Ouro RL - Exploration of Ouro LM

## Philosophy
Research code — simple, correct, and efficient:
- Simple, hackable implementations > frameworks
- Correctness is non-negotiable — write pytest tests for non-trivial functions
- Tests should cover behavior and edge cases, not implementation details — keep them maintainable so refactors don't require rewriting every test
- Compute is scarce (2×A100-SXM4-80GB) — always consider memory, FLOPs, and throughput implications

## Code Standards
- Before writing new functions, check existing modules for code that can be extended or reused
- Type hints on all signatures (modern syntax: `str | None`, `list[int]`)
- Run ruff after changes: `uv run ruff format . && uv run ruff check --fix .`

## Package Management (CRITICAL)
- ALWAYS: `uv add <package>`
- NEVER: manually edit pyproject.toml
- NEVER: `pip install` or `uv pip install`

## Running Code
Python scripts must be run within the uv environment:
- **Option 1**: `uv run python script.py` (recommended for one-off commands)
- **Option 2**: Activate environment first with `source .venv/bin/activate`, then run normally

## Key Dependencies
- `torch==2.9.0` — pinned, cu128
- `flash-attn` — for efficient attention

## Debugging
Check `.venv` source code directly for library implementation details

## Ouro-Thinking Model Quirks (Huggingface)
- **Wrong bos/eos upstream**: Both 1.4B and 2.6B Thinking models ship with bos/eos/pad all set to `<|endoftext|>` (id=0). Correct: bos=`<|im_start|>` (1), eos=`<|im_end|>` (2).
- **enable_thinking**: Ouro-Thinking won't emit `<think>` on its own — it must be prepended in the prompt. Upstream chat template lacks `enable_thinking` support. We use a local template at `templates/ouro_chat.j2` that appends `<think>\n` after `<|im_start|>assistant\n` when `enable_thinking=True`.

## Project Structure
- `ouro_rl/` — core library (training, model patches, data)
- `eval/` — lm-eval tasks, configs, and analysis scripts
- `scripts/` — standalone utilities
- `tests/` — pytest suite
- `knowledge/` — paper summaries and research notes
- `templates/` — chat templates (Jinja2)
- `dev/` — exploratory / throwaway scripts

## Research Stack
- Framework: PyTorch + HuggingFace Transformers + vLLM
- Testing: pytest for all non-trivial functions
