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

## Debugging
Check `.venv` source code directly for library implementation details

## Ouro Model (`ouro_rl/modeling/`)
We ship a local copy of the Ouro model with upstream bugs fixed:
- **`OuroForCausalLM`** — local model class with fixed `UniversalTransformerCache.get_mask_sizes` (upstream returns wrong kv_length during autoregressive steps, corrupting batched generation)
- **Token IDs** (`ouro_rl.modeling.constants`): Upstream ships bos/eos/pad all set to `<|endoftext|>` (id=0). Correct: `BOS_TOKEN_ID=1` (`<|im_start|>`), `EOS_TOKEN_ID=2` (`<|im_end|>`), `PAD_TOKEN_ID=0`.
- **Chat template**: `ouro_rl/modeling/chat_template.jinja` — supports `enable_thinking=True` to prepend `<think>\n` after `assistant\n`
