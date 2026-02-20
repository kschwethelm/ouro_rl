"""MATH dataset loading and prompt formatting for GRPO training."""

from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

CHAT_TEMPLATE = (Path(__file__).resolve().parent.parent / "templates" / "ouro_chat.j2").read_text()

# Default system prompt for math reasoning.
MATH_SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the following math problem step by step. Put your final answer within \\boxed{}."
)


def load_math_train(dataset_name: str = "qwedsacf/competition_math") -> Dataset:
    """Load MATH training split.

    Columns: problem, level, type, solution.
    The solution contains the final answer in \\boxed{...}.
    """
    return load_dataset(dataset_name, split="train", trust_remote_code=True)


def format_prompt(
    problem: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    system_prompt: str = MATH_SYSTEM_PROMPT,
    enable_thinking: bool = True,
) -> str:
    """Format a MATH problem into a chat prompt string for generation."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=CHAT_TEMPLATE,
        enable_thinking=enable_thinking,
    )


def extract_boxed_answer(solution: str) -> str | None:
    r"""Extract the content of \boxed{...} from a MATH solution string."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None
    # Walk forward matching braces to handle nested \boxed{...{...}...}
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            if depth == 0:
                return solution[start:i]
            depth -= 1
    return None
