"""Grid-search parameter tuning for MitigV pipelines.

The tuner deliberately keeps model loading outside the search loop.  A model
and processor are loaded once, while each candidate receives a fresh mitigator
with a validated configuration.  The tuning set is evaluated with the same
strict CHAIR implementation used by the regular pipeline evaluator.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mitigv.core.base import BaseMitigator, MitigatorConfig
from mitigv.core.registry import build_mitigator
from mitigv.evaluation.pipeline import evaluate_pipeline_json

__all__ = ["expand_parameter_grid", "tune_mitigator", "grid_search", "auto_tune"]

_MINIMIZE_METRICS = {
    "chairs",
    "chairi",
    "hallucination_rate",
    "hallucinated_count",
    "length_adjusted_chairi",
}


def _load_grid(value: Mapping[str, Sequence[Any]] | str | Path) -> dict[str, list[Any]]:
    """Normalize a parameter grid supplied as a mapping or JSON file."""

    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"parameter grid file not found: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError("param_grid must be a mapping or a JSON file path")
    normalized: dict[str, list[Any]] = {}
    for name, values in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("parameter names must be non-empty strings")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"grid values for {name!r} must be a sequence")
        candidates = list(values)
        if not candidates:
            raise ValueError(f"grid values for {name!r} must not be empty")
        normalized[name] = candidates
    return normalized


def expand_parameter_grid(
    param_grid: Mapping[str, Sequence[Any]] | str | Path,
) -> list[dict[str, Any]]:
    """Expand a mapping of parameter values into deterministic combinations."""

    grid = _load_grid(param_grid)
    if not grid:
        return [{}]
    names = list(grid)
    return [
        dict(zip(names, values))
        for values in itertools.product(*(grid[n] for n in names))
    ]


def _metric_value(evaluation: Mapping[str, Any], metric: str) -> float:
    """Extract a scalar from a pipeline evaluation, accepting common aliases."""

    summary = evaluation.get("chair", {}).get("summary", {})
    if metric in summary:
        value = summary[metric]
    else:
        wanted = metric.lower().replace("-", "_")
        value = None
        for name, item in summary.items():
            if name.lower().replace("-", "_") == wanted:
                value = item
                break
        if value is None:
            raise KeyError(
                f"metric {metric!r} is unavailable; choose one of {sorted(summary)}"
            )
    if isinstance(value, Mapping):
        value = value.get("value", value.get("score"))
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"metric {metric!r} must resolve to a numeric value") from exc


def _default_maximize(metric: str) -> bool:
    return metric.lower().replace("-", "_") not in _MINIMIZE_METRICS


def _candidate_pipeline(
    algorithm: str | type[BaseMitigator] | None,
    model: Any,
    processor: Any,
    params: Mapping[str, Any],
    base_config: MitigatorConfig | Mapping[str, Any] | None,
    algorithm_kwargs: Mapping[str, Any] | None,
    pipeline_factory: Callable[[Mapping[str, Any]], Callable[..., Any]] | None,
) -> Callable[..., Any]:
    if pipeline_factory is not None:
        return pipeline_factory(dict(params))
    if algorithm is None:
        raise ValueError("algorithm is required unless pipeline_factory is supplied")
    kwargs = dict(algorithm_kwargs or {})
    overlap = set(kwargs) & set(params)
    if overlap:
        raise ValueError(f"parameter grid overlaps algorithm_kwargs: {sorted(overlap)}")
    kwargs.update(params)
    if isinstance(algorithm, str):
        return build_mitigator(
            algorithm, model=model, processor=processor, config=base_config, **kwargs
        )
    if not isinstance(algorithm, type) or not issubclass(algorithm, BaseMitigator):
        raise TypeError("algorithm must be a registered name or BaseMitigator class")
    return algorithm(model=model, processor=processor, config=base_config, **kwargs)


def tune_mitigator(
    records_json: str | Path,
    *,
    param_grid: Mapping[str, Sequence[Any]] | str | Path,
    algorithm: str | type[BaseMitigator] | None = None,
    model: Any = None,
    processor: Any = None,
    pipeline_factory: Callable[[Mapping[str, Any]], Callable[..., Any]] | None = None,
    base_config: MitigatorConfig | Mapping[str, Any] | None = None,
    algorithm_kwargs: Mapping[str, Any] | None = None,
    image_root: str | Path = "~/dataset",
    prompt: str = "Describe the image in one sentence.",
    metric: str | Callable[[Mapping[str, Any]], float] = "CHAIRi",
    maximize: bool | None = None,
    output_json: str | Path | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Select the best parameter combination on a tuning record JSON file.

    ``pipeline_factory`` can be used for non-MitigV runtimes; it receives one
    parameter mapping and must return a callable accepted by
    :func:`evaluate_pipeline_json`.  Otherwise ``algorithm`` is built through
    the normal registry and the supplied model/processor are reused.
    """

    combinations = expand_parameter_grid(param_grid)
    if callable(metric):
        metric_name = getattr(metric, "__name__", "custom")
        scorer = metric
    else:
        metric_name = str(metric)

        def scorer(result: Mapping[str, Any]) -> float:
            return _metric_value(result, metric_name)
    if maximize is None:
        maximize = _default_maximize(metric_name)

    candidates: list[dict[str, Any]] = []
    for params in combinations:
        pipeline = _candidate_pipeline(
            algorithm, model, processor, params, base_config, algorithm_kwargs, pipeline_factory
        )
        evaluation = evaluate_pipeline_json(
            records_json,
            pipeline,
            image_root=image_root,
            prompt=prompt,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        score = float(scorer(evaluation))
        if not math.isfinite(score):
            raise ValueError(f"candidate score must be finite, got {score!r}")
        candidates.append({"params": dict(params), "score": score, "evaluation": evaluation})

    best = (
        max(candidates, key=lambda item: item["score"])
        if maximize
        else min(candidates, key=lambda item: item["score"])
    )
    result = {
        "records_json": str(Path(records_json).expanduser()),
        "image_root": str(Path(image_root).expanduser()),
        "prompt": prompt,
        "metric": metric_name,
        "maximize": bool(maximize),
        "num_candidates": len(candidates),
        "best_params": best["params"],
        "best_score": best["score"],
        "best_evaluation": best["evaluation"],
        "candidates": candidates,
    }
    if output_json is not None:
        destination = Path(output_json).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


# A concise alias for callers who think in terms of generic grid search.
grid_search = tune_mitigator
auto_tune = tune_mitigator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, dest="records_json")
    parser.add_argument("--param-grid", required=True, help="JSON object or path to a JSON object")
    parser.add_argument("--algorithm", default="vcd")
    parser.add_argument("--model-type", required=True, choices=("llava", "qwen2.5-vl"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--image-root", default="~/dataset")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prompt", default="Describe the image in one sentence.")
    parser.add_argument("--metric", default="CHAIRi")
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="maximize the metric instead of using its default direction",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="minimize the metric instead of using its default direction",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.maximize and args.minimize:
        parser.error("--maximize and --minimize are mutually exclusive")
    grid_value: Any = args.param_grid
    try:
        if grid_value.strip().startswith("{"):
            grid_value = json.loads(grid_value)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --param-grid JSON: {exc}")

    from mitigv import load_mitigator

    pipeline = load_mitigator(
        args.algorithm,
        model_type=args.model_type,
        model_id=str(Path(args.model_id).expanduser()),
        model_kwargs={"torch_dtype": "auto", "device_map": "auto"},
        max_new_tokens=args.max_new_tokens,
    )
    # Reuse the loaded objects; tuning creates a fresh algorithm instance for
    # every candidate while keeping the expensive model load out of the loop.
    tune_mitigator(
        args.records_json,
        param_grid=grid_value,
        algorithm=args.algorithm,
        model=pipeline.model,
        processor=pipeline.processor,
        base_config=pipeline.config,
        image_root=args.image_root,
        prompt=args.prompt,
        metric=args.metric,
        maximize=True if args.maximize else False if args.minimize else None,
        output_json=args.output_json,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
