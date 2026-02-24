"""Custom process_results for math tasks using math-verify (symbolic comparison).

Replaces the fragile regex extraction + exact_match pipeline with proper
LaTeX-aware parsing and equivalence checking — same library used in the
GRPO reward function (ouro_rl/reward.py).

Note: lm-eval strips <think>...</think> before calling process_results
when enable_thinking=True is set in MODEL_ARGS.
"""

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

GOLD_EXTRACTION = [LatexExtractionConfig(), ExprExtractionConfig()]
PRED_EXTRACTION = [LatexExtractionConfig(), ExprExtractionConfig()]


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    """Score model output against gold answer using math-verify."""
    answer_text = results[0]

    # Gold: prefer 'answer' field (clean), fall back to full 'solution'.
    # Keys vary by dataset (e.g. MATH-500: 'answer', AIME24: 'Answer').
    gold_text = doc.get("answer") or doc.get("Answer") or doc.get("solution") or doc.get("Solution", "")
    if isinstance(gold_text, (int, float)):
        gold_text = str(gold_text)

    try:
        gold_parsed = parse(gold_text, extraction_config=GOLD_EXTRACTION)
        pred_parsed = parse(answer_text, extraction_config=PRED_EXTRACTION)
    except Exception:
        return {"exact_match": 0}

    if not gold_parsed or not pred_parsed:
        return {"exact_match": 0}

    try:
        return {"exact_match": 1 if verify(gold_parsed[0], pred_parsed[0]) else 0}
    except Exception:
        return {"exact_match": 0}
