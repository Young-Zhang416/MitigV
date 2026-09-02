"""POPE evaluation helpers and the VCD paper's reference numbers.

The metric logic mirrors the official VCD implementation's ``eval_pope.py``
exactly: for a positive (``yes``) ground-truth, an answer containing ``yes`` is
a true positive; for a negative (``no``) ground-truth, an answer containing
``no`` is a true negative. All numbers are reported as percentages (0-100).
"""

from __future__ import annotations

import json
from typing import Any, Sequence

#: Reference numbers from the VCD paper (Leng et al., CVPR 2024), Table 1,
#: LLaVA-1.5-7B on the MSCOCO POPE benchmark.
REFERENCE: dict[str, dict[str, dict[str, float]]] = {
    "random": {
        "Regular": {"accuracy": 83.29, "precision": 92.13, "recall": 72.80, "f1": 81.33},
        "VCD": {"accuracy": 87.73, "precision": 91.42, "recall": 83.28, "f1": 87.16},
    },
    "popular": {
        "Regular": {"accuracy": 81.88, "precision": 88.93, "recall": 72.80, "f1": 80.06},
        "VCD": {"accuracy": 85.38, "precision": 86.92, "recall": 83.28, "f1": 85.06},
    },
    "adversarial": {
        "Regular": {"accuracy": 78.96, "precision": 83.06, "recall": 72.75, "f1": 77.57},
        "VCD": {"accuracy": 80.88, "precision": 79.45, "recall": 83.29, "f1": 81.33},
    },
}

METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def load_pope(path: str) -> list[dict[str, Any]]:
    """Load a POPE JSONL file (one JSON object per line) into a list."""
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def compute_metrics(gt_labels: Sequence[str], gen_texts: Sequence[str]) -> dict[str, float]:
    """Compute POPE accuracy / precision / recall / F1 (percentages)."""
    if len(gt_labels) != len(gen_texts):
        raise ValueError("ground-truth and generated lists must have equal length")

    true_pos = true_neg = false_pos = false_neg = 0
    yes_answers = 0
    for gt, gen in zip(gt_labels, gen_texts):
        gt = gt.strip().lower()
        gen = gen.strip().lower()
        if gt == "yes":
            if "yes" in gen:
                true_pos += 1
                yes_answers += 1
            else:
                false_neg += 1
        elif gt == "no":
            if "no" in gen:
                true_neg += 1
            else:
                false_pos += 1
                yes_answers += 1
        else:
            raise ValueError(f"unexpected ground-truth label: {gt!r}")

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = len(gt_labels)

    return {
        "accuracy": 100.0 * (true_pos + true_neg) / total,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
        "yes_proportion": 100.0 * yes_answers / total,
    }


def compare_to_reference(
    metrics: dict[str, float],
    split: str,
    method: str,
    tolerance: float = 2.0,
) -> dict[str, Any]:
    """Compare computed metrics to the paper reference and return a verdict.

    ``ignorable`` is true when every tracked metric differs from the reference
    by at most ``tolerance`` (percentage points).
    """
    reference = REFERENCE[split][method]
    diffs = {name: metrics[name] - reference[name] for name in METRIC_NAMES}
    ignorable = all(abs(diff) <= tolerance for diff in diffs.values())
    return {"reference": reference, "diffs": diffs, "ignorable": ignorable}
