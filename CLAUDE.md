# Ouro RL - Exploration of Ouro LM

## Philosophy
Research code optimized for rapid iteration and debugging:
- Simple, hackable implementations > frameworks
- Missing error handling is GOOD (faster bug discovery)
- Understand every component > black-box abstractions

## Code Standards
- Type hints on all signatures (modern syntax: `str | None`, `list[int]`)
- Self-documenting names > comments
- Run ruff after changes: `uv run ruff format . && uv run ruff check --fix .`

## Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Package Management (CRITICAL)
- ✅ ALWAYS: `uv add <package>`
- ❌ NEVER: manually edit pyproject.toml
- ❌ NEVER: `pip install` or `uv pip install`

## Running Code
Python scripts must be run within the uv environment:
- **Option 1**: `uv run python script.py` (recommended for one-off commands)
- **Option 2**: Activate environment first with `source .venv/bin/activate`, then run normally

## Key Dependencies
- `transformers==4.56.0` — pinned, this is the version that works for loading the Ouro pretrained model
- `torch==2.9.0` — pinned, cu128
- `flash-attn` — for efficient attention

## Debugging
Check `.venv` source code directly for library implementation details

## Research Stack
- Framework: PyTorch + HuggingFace Transformers
- Testing: pytest for core components only (skip for exploratory code)
