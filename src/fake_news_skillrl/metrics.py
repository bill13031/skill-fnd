from __future__ import annotations

from typing import Dict, Iterable, List


LABELS = ["fake", "real"]


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_classification_metrics(results: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = list(results)
    if not rows:
        return {
            "label_accuracy": 0.0,
            "macro_f1": 0.0,
            "fake_f1": 0.0,
            "real_f1": 0.0,
            "evidence_match_rate": 0.0,
            "invalid_action_rate": 0.0,
        }

    accuracy = sum(1 for row in rows if row.get("label_correct")) / len(rows)
    evidence_rate = sum(float(row.get("evidence_match_rate", 0.0)) for row in rows) / len(rows)
    invalid_rate = sum(int(row.get("invalid_action_count", 0)) for row in rows) / len(rows)

    f1_scores: Dict[str, float] = {}
    for label in LABELS:
        true_positive = sum(
            1
            for row in rows
            if row.get("predicted_label") == label and row.get("gold_label") == label
        )
        false_positive = sum(
            1
            for row in rows
            if row.get("predicted_label") == label and row.get("gold_label") != label
        )
        false_negative = sum(
            1
            for row in rows
            if row.get("predicted_label") != label and row.get("gold_label") == label
        )
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1_scores[label] = _safe_divide(2 * precision * recall, precision + recall)

    macro_f1 = sum(f1_scores.values()) / len(LABELS)
    return {
        "label_accuracy": accuracy,
        "macro_f1": macro_f1,
        "fake_f1": f1_scores["fake"],
        "real_f1": f1_scores["real"],
        "evidence_match_rate": evidence_rate,
        "invalid_action_rate": invalid_rate,
    }
