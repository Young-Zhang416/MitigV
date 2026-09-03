"""POPE and AMBER discriminative evaluation.

Answers are parsed as the first standalone ``yes``/``no`` token, so variants
such as ``Yes,`` and ``No.`` are handled while unrelated substrings (e.g.
``yesterday``) are not.  The evaluator keeps one result row per question.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ANSWER_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_yes_no(answer: Any) -> str | None:
    """Parse the first standalone yes/no answer, including punctuation variants."""

    match = _ANSWER_RE.search(str(answer).strip())
    return match.group(1).lower() if match else None


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
            return parquet.read_table(path).to_pylist()
        except ImportError:
            try:
                import pandas as pd
            except ImportError as error:  # pragma: no cover - optional data reader
                raise ImportError("reading AMBER parquet requires pyarrow or pandas") from error
            return pd.read_parquet(path).to_dict(orient="records")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, Mapping):
            for key in ("items", "data", "annotations", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
        return data
    raise ValueError(f"unsupported evaluation file format: {path}")


def _prediction_answer(item: Mapping[str, Any]) -> Any:
    for key in ("answer", "generated_text", "caption", "text", "output", "response"):
        if key in item:
            return item[key]
    raise KeyError("prediction item lacks answer/generated_text/caption/text/output/response")


def _truth(item: Mapping[str, Any]) -> str:
    for key in ("label", "truth", "answer", "target", "gt"):
        if key in item:
            value = parse_yes_no(item[key])
            if value is None:
                raise ValueError(f"ground-truth value is not yes/no: {item[key]!r}")
            return value
    raise KeyError("question item lacks label/truth/target/gt")


def _question_id(item: Mapping[str, Any], index: int) -> Any:
    for key in ("id", "question_id", "qid", "image_id"):
        if key in item:
            return item[key]
    return index


def evaluate_discriminative(
    questions: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]] | Sequence[str],
) -> dict[str, Any]:
    """Evaluate aligned yes/no questions and return metrics plus per-question rows."""

    if not questions:
        raise ValueError("questions must not be empty")
    if len(questions) != len(predictions):
        raise ValueError("questions and predictions must have equal length")
    rows: list[dict[str, Any]] = []
    tp = tn = fp = fn = 0
    for index, (question, prediction) in enumerate(zip(questions, predictions)):
        answer = prediction if isinstance(prediction, str) else _prediction_answer(prediction)
        truth = _truth(question)
        parsed = parse_yes_no(answer)
        correct = parsed == truth
        if truth == "yes":
            if parsed == "yes":
                tp += 1
            else:
                fn += 1
        else:
            if parsed == "no":
                tn += 1
            else:
                fp += 1
        rows.append({
            "index": index,
            "id": _question_id(question, index),
            "image_id": question.get("image_id", question.get("image")),
            "question": question.get("question", question.get("query", question.get("text"))),
            "ground_truth": truth,
            "answer": str(answer),
            "parsed_answer": parsed,
            "correct": correct,
        })
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(questions),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "unparsed": sum(row["parsed_answer"] is None for row in rows)},
        "details": rows,
    }


def evaluate_pope_subsets(
    prediction_dir: str | Path,
    *,
    subsets: Sequence[str] = ("random", "popular", "adversarial"),
) -> dict[str, Any]:
    """Evaluate POPE random/popular/adversarial JSON files in a local directory."""

    root = Path(prediction_dir).expanduser()
    result: dict[str, Any] = {}
    for subset in subsets:
        gt_path = root / f"coco_pope_{subset}.json"
        pred_path = root / f"{subset}_predictions.json"
        if not pred_path.exists():
            pred_path = root / f"coco_pope_{subset}_predictions.json"
        result[subset] = evaluate_discriminative(_load_records(gt_path), _load_records(pred_path))
    return result


def evaluate_amber(
    questions_path: str | Path,
    predictions: Sequence[Mapping[str, Any]] | Sequence[str],
) -> dict[str, Any]:
    """Evaluate one local AMBER discriminative parquet subset."""

    return evaluate_discriminative(_load_records(questions_path), predictions)


def evaluate_suite(
    pope: Mapping[str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]] | Sequence[str]]] | None = None,
    amber: Mapping[str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]] | Sequence[str]]] | None = None,
) -> dict[str, Any]:
    """Evaluate a collection of POPE subsets and AMBER task subsets."""

    result: dict[str, Any] = {"pope": {}, "amber": {}}
    for name, (questions, predictions) in (pope or {}).items():
        result["pope"][name] = evaluate_discriminative(questions, predictions)
    for name, (questions, predictions) in (amber or {}).items():
        result["amber"][name] = evaluate_discriminative(questions, predictions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="POPE JSON/JSONL or AMBER parquet")
    parser.add_argument("--predictions", required=True, help="JSON/JSONL answers aligned with questions")
    parser.add_argument("--output-json", default="results/discriminative.json")
    args = parser.parse_args()
    result = evaluate_discriminative(_load_records(args.questions), _load_records(args.predictions))
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
