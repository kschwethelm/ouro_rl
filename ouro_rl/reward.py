"""Binary exact-match reward for math problems using math-verify."""

import logging

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

logger = logging.getLogger(__name__)

# Extraction configs: try LaTeX first (boxed answers), then plain expressions.
GOLD_EXTRACTION = [LatexExtractionConfig(), ExprExtractionConfig()]
PRED_EXTRACTION = [LatexExtractionConfig(), ExprExtractionConfig()]


def score_answer(model_output: str, ground_truth: str) -> float:
    """Score a model output against ground truth. Returns 0.0 or 1.0.

    Args:
        model_output: Full model response (may include <think>...</think>).
        ground_truth: Gold solution string (containing \\boxed{...}).
    """
    # Strip thinking tokens if present.
    end_idx = model_output.find("</think>")
    answer_text = model_output[end_idx + 8 :] if end_idx != -1 else model_output

    try:
        gold_parsed = parse(ground_truth, extraction_config=GOLD_EXTRACTION)
        pred_parsed = parse(answer_text, extraction_config=PRED_EXTRACTION)
    except Exception:
        logger.debug("Parse failed for: %s", answer_text[:100])
        return 0.0

    if not gold_parsed or not pred_parsed:
        return 0.0

    try:
        # Compare first extracted expression from each.
        return 1.0 if verify(gold_parsed[0], pred_parsed[0]) else 0.0
    except Exception:
        logger.debug("Verify failed for gold=%s pred=%s", gold_parsed, pred_parsed)
        return 0.0
