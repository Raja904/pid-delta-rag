"""
eval/metrics.py
Delta P/R/F1 and chat groundedness metrics.
"""
from __future__ import annotations
from difflib import SequenceMatcher


def delta_precision_recall_f1(
    predicted: list[dict],
    expected:  list[dict],
    match_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """
    Soft P/R/F1 for delta detection.
    A predicted change matches an expected change if:
      - same change_type AND
      - text similarity (old+new combined) >= match_threshold
    """
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not expected:
        return 0.0, 0.0, 0.0

    def _text_sig(item: dict) -> str:
        return (item.get("old_value","") + " " + item.get("new_value","")).lower()

    matched_exp = set()
    true_positives = 0

    for pred in predicted:
        for j, exp in enumerate(expected):
            if j in matched_exp:
                continue
            if pred.get("change_type") != exp.get("change_type"):
                continue
            sim = SequenceMatcher(None, _text_sig(pred), _text_sig(exp)).ratio()
            if sim >= match_threshold:
                true_positives += 1
                matched_exp.add(j)
                break

    precision = true_positives / len(predicted) if predicted else 0.0
    recall    = true_positives / len(expected)  if expected  else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def groundedness_score_heuristic(
    answer: str,
    citations: list,
    expected_keywords: list[str],
) -> float:
    """
    Heuristic groundedness score (0.0 - 1.0).
    Rewards: having citations + containing expected keywords.
    For production use, replace with LLM-as-judge.
    """
    if not answer:
        return 0.0

    # Has citations in answer text (inline format [SOURCE·ref])
    has_inline_citation = "[" in answer and "]" in answer
    has_obj_citations   = len(citations) > 0

    citation_score = 0.5 * int(has_inline_citation) + 0.5 * int(has_obj_citations)

    # Keyword coverage
    lower_answer = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower_answer)
    kw_score = hits / max(len(expected_keywords), 1)

    return round(0.4 * citation_score + 0.6 * kw_score, 4)
