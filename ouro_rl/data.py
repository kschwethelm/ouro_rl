"""MATH dataset loading and prompt formatting for GRPO training."""

from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

CHAT_TEMPLATE = (Path(__file__).resolve().parent.parent / "templates" / "ouro_chat.j2").read_text()

# System prompt for math reasoning.
#
# For SFT'd thinking models (e.g. Ouro-Thinking) that already know <think> and
# \boxed{} format from training, no system prompt is needed — saves tokens
# across all rollouts. Set to None to omit entirely.
#
# For base/instruct models that haven't been trained on reasoning traces, use a
# format-only prompt (R1-Zero style — specify output format, NOT how to reason,
# so reasoning strategies emerge from RL):
#   "The assistant first thinks about the reasoning process in the mind and then "
#   "provides the user with the answer. The reasoning process is enclosed within "
#   "<think> </think> tags, and the final answer should be given within \\boxed{}."
MATH_SYSTEM_PROMPT: str | None = None

# Interruption phrase for truncated completions (ScaleRL, arXiv:2510.13786).
# When a completion hits the thinking budget without producing </think>,
# this phrase is appended to force the model to produce a final answer.
INTERRUPTION_PHRASE = "Okay, time is up. Let me stop thinking and formulate a final answer now.\n</think>\n"


def load_math_train(dataset_name: str = "qwedsacf/competition_math") -> Dataset:
    """Load MATH training split.

    Columns: problem, level, type, solution.
    The solution contains the final answer in \\boxed{...}.
    """
    return load_dataset(dataset_name, split="train")


def format_prompt(
    problem: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    system_prompt: str | None = MATH_SYSTEM_PROMPT,
    enable_thinking: bool = True,
) -> str:
    """Format a MATH problem into a chat prompt string for generation."""
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": problem})
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
