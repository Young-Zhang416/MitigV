"""Poisson length-control analysis for CHAIR image-level hallucinations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        if isinstance(value.get("details"), list):
            return value["details"]
        if isinstance(value.get("predictions"), list):
            return value["predictions"]
    raise ValueError("configuration must be a details list or an object containing details")


def _load(value: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, (str, Path)):
        return [dict(item) for item in value]
    data = json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
    return _records(data)


def _row(item: Mapping[str, Any]) -> tuple[float, float, float]:
    words = item.get("word_count", item.get("words"))
    if words is None:
        caption = str(item.get("caption", item.get("generated_text", "")))
        words = len(caption.split())
    hallucinated = item.get("hallucinated", item.get("hallucinated_objects", []))
    mentioned = item.get("mentioned_objects", item.get("generated_objects", []))
    return float(words), float(len(hallucinated) if isinstance(hallucinated, Sequence) else hallucinated), float(len(mentioned) if isinstance(mentioned, Sequence) else mentioned)


def _fit_poisson(words: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, bool]:
    centered = words - float(words.mean())
    x = np.column_stack([np.ones(len(words)), centered])
    # Stable Poisson negative log-likelihood with log-link.
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(x @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        value = float(np.sum(mu - counts * eta))
        gradient = x.T @ (mu - counts)
        return value, gradient

    result = minimize(lambda beta: objective(beta)[0], np.zeros(2), jac=lambda beta: objective(beta)[1], method="BFGS")
    if not result.success or not np.all(np.isfinite(result.x)):
        # A constant fallback still gives a well-defined, auditable analysis.
        return np.array([np.log(max(float(counts.mean()), 1e-12)), 0.0]), False
    return result.x, True


def analyze_length_control(configurations: Mapping[str, Any], baseline: str | None = None) -> dict[str, Any]:
    """Fit per-configuration Poisson models and report length-adjusted residual gain.

    For each configuration, ``hallucinated_count ~ Poisson(exp(a + b*(words -
    mean_words)))`` is fitted.  The adjusted CHAIRi is expected hallucinated
    mentions at the common mean description length divided by observed object
    mentions. ``residual_gain`` is the adjusted CHAIRi improvement over the
    selected baseline (zero for the baseline itself).
    """

    if not configurations:
        raise ValueError("configurations must not be empty")
    parsed = {name: [_row(item) for item in _load(value)] for name, value in configurations.items()}
    if any(not rows for rows in parsed.values()):
        raise ValueError("every configuration must contain at least one image")
    common_mean = float(np.mean([row[0] for rows in parsed.values() for row in rows]))
    fitted: dict[str, dict[str, Any]] = {}
    for name, rows in parsed.items():
        words = np.array([row[0] for row in rows])
        hallucinated = np.array([row[1] for row in rows])
        mentions = np.array([row[2] for row in rows])
        beta, converged = _fit_poisson(words, hallucinated)
        expected = float(np.exp(np.clip(beta[0] + beta[1] * (common_mean - words.mean()), -30, 30)))
        raw_h = float(hallucinated.sum())
        raw_m = float(mentions.sum())
        raw_chairi = raw_h / raw_m if raw_m else 0.0
        adjusted = expected / float(mentions.mean()) if mentions.mean() else 0.0
        fitted[name] = {
            "n_images": len(rows),
            "mean_words": float(words.mean()),
            "mean_hallucinated_count": float(hallucinated.mean()),
            "raw_chairi": raw_chairi,
            "length_adjusted_chairi": adjusted,
            "length_corrected_chairi": adjusted,
            "poisson_intercept": float(beta[0]),
            "poisson_length_coefficient": float(beta[1]),
            "poisson_converged": converged,
        }
    if baseline is None:
        baseline = next(iter(fitted))
    if baseline not in fitted:
        raise KeyError(f"baseline configuration not found: {baseline}")
    reference = fitted[baseline]["length_adjusted_chairi"]
    for item in fitted.values():
        item["residual_gain"] = reference - item["length_adjusted_chairi"]
        item["length_corrected_chairi_residual_gain"] = item["residual_gain"]
    scatter = [
        {"configuration": name, "mean_words": item["mean_words"], "chairi": item["raw_chairi"], "length_adjusted_chairi": item["length_adjusted_chairi"]}
        for name, item in fitted.items()
    ]
    return {"baseline": baseline, "common_mean_words": common_mean, "configurations": fitted, "length_chairi_scatter": scatter}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", nargs=2, metavar=("NAME", "DETAILS_JSON"), required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output-json", default="results/length_analysis.json")
    args = parser.parse_args()
    result = analyze_length_control({name: path for name, path in args.config}, baseline=args.baseline)
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
