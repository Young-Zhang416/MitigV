"""Compute and plot the recall/CHAIRi frontier from frozen evaluator outputs.

The input is a JSON or JSONL manifest.  Each manifest row identifies one
method/configuration result and points at a JSON result produced by
``evaluate_pipeline_json`` (or directly embeds that result)::

    {"name": "vcd-a1", "kind": "published", "family": "contrastive",
     "dataset": "coco_val500", "model": "llava", "result": "results/vcd.json"}

Baseline rows use ``kind: "baseline"`` (or ``trivial``); every other row is a
published method.  Rows are grouped by ``dataset`` and ``model`` so one
manifest can produce COCO/AMBER and LLaVA/Qwen figures in one invocation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "load_manifest",
    "pareto_frontier",
    "analyze_frontier",
    "write_report",
    "main",
]


def _read_json(path: str | Path) -> Any:
    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_manifest(value: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Load a manifest from JSON/JSONL or an already parsed sequence."""

    if isinstance(value, (str, Path)):
        data = _read_json(value)
    else:
        data = value
    if isinstance(data, Mapping):
        data = data.get("entries", data.get("data", data.get("results")))
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ValueError("manifest must be a list, JSONL file, or an entries wrapper")
    rows = [dict(row) for row in data if isinstance(row, Mapping)]
    if len(rows) != len(data) or not rows:
        raise ValueError("manifest must contain a non-empty list of objects")
    return rows


def _result_for_entry(entry: Mapping[str, Any], base_dir: Path) -> Mapping[str, Any]:
    value = entry.get("result", entry.get("result_json", entry.get("evaluation")))
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        loaded = _read_json(path)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"result file must contain an object: {path}")
        return loaded
    raise KeyError(f"manifest row {entry.get('name', '<unnamed>')!r} lacks result/result_json")


def _details(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = result.get("chair", result)
    if isinstance(value, Mapping) and isinstance(value.get("details"), list):
        rows = value["details"]
    elif isinstance(result.get("details"), list):
        rows = result["details"]
    else:
        raise ValueError("result must contain per-image CHAIR details")
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("result details must be a non-empty list of objects")
    return [dict(row) for row in rows]


def _objects(row: Mapping[str, Any], key: str, fallback: str | None = None) -> list[Any]:
    value = row.get(key, row.get(fallback, []) if fallback else [])
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value) if isinstance(value, Sequence) else []


def _row_metrics(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    recalled = _objects(row, "recalled")
    gt = _objects(row, "gt_objects", "ground_truth_objects")
    hallucinated = _objects(row, "hallucinated", "hallucinated_objects")
    mentioned = _objects(row, "mentioned_objects", "generated_objects")
    # ``mentioned_objects`` is the exact CHAIR denominator.  Older result
    # files may omit it; use the hallucinated + recalled union as a fallback.
    mentioned_count = len(mentioned) or len(hallucinated) + len(recalled)
    return float(len(recalled)), float(len(gt)), float(len(hallucinated)), float(mentioned_count)


def _metric_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    recalled = gt = hallucinated = mentioned = 0.0
    for row in rows:
        r, g, h, m = _row_metrics(row)
        recalled += r
        gt += g
        hallucinated += h
        mentioned += m
    return recalled / gt if gt else 0.0, hallucinated / mentioned if mentioned else 0.0


def _point(entry: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    rows = _details(result)
    recall, chairi = _metric_from_rows(rows)
    chair_result = result.get("chair", result)
    summary = chair_result.get("summary", {}) if isinstance(chair_result, Mapping) else {}
    def ci(name: str) -> list[float] | None:
        value = summary.get(name) if isinstance(summary, Mapping) else None
        if isinstance(value, Mapping) and isinstance(value.get("ci95"), Sequence):
            return [float(value["ci95"][0]), float(value["ci95"][1])]
        return None
    image_ids = [row.get("image_id", index) for index, row in enumerate(rows)]
    return {
        "name": str(entry.get("name", entry.get("method", "unnamed"))),
        "method": str(entry.get("method", entry.get("name", "unnamed"))),
        "family": str(entry.get("family", entry.get("method", "unknown"))),
        "kind": str(entry.get("kind", "published")).lower(),
        "dataset": str(entry.get("dataset", "default")),
        "model": str(entry.get("model", "default")),
        "object_recall": recall,
        "chairi": chairi,
        "object_recall_ci95": ci("object_recall"),
        "chairi_ci95": ci("CHAIRi"),
        "details": rows,
        "image_ids": image_ids,
    }


def pareto_frontier(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return non-dominated points (higher recall, lower CHAIRi is better)."""

    ordered = sorted(
        (dict(point) for point in points),
        key=lambda item: (float(item["object_recall"]), float(item["chairi"])),
    )
    frontier: list[dict[str, Any]] = []
    best_y = float("inf")
    for point in reversed(ordered):
        y = float(point["chairi"])
        if y < best_y:
            frontier.append(point)
            best_y = y
    return list(reversed(frontier))


def _interpolate(frontier: Sequence[Mapping[str, Any]], x: float) -> float | None:
    if not frontier:
        return None
    xs = np.array([float(point["object_recall"]) for point in frontier])
    ys = np.array([float(point["chairi"]) for point in frontier])
    if x < xs.min() or x > xs.max():
        return None
    return float(np.interp(x, xs, ys))


def _aligned_rows(point: Mapping[str, Any]) -> dict[Any, Mapping[str, Any]]:
    rows = point["details"]
    return {row.get("image_id", index): row for index, row in enumerate(rows)}


def _bootstrap_metrics(points: Sequence[Mapping[str, Any]], samples: int, seed: int) -> dict[str, np.ndarray]:
    if not points:
        return {}
    aligned = [_aligned_rows(point) for point in points]
    common = set(aligned[0])
    for rows in aligned[1:]:
        common &= set(rows)
    if not common:
        raise ValueError("all compared results must share image_id values")
    ids = sorted(common, key=str)
    rng = np.random.default_rng(seed)
    recalls = np.empty((samples, len(points)))
    chairis = np.empty((samples, len(points)))
    for sample in range(samples):
        selected = [ids[index] for index in rng.integers(0, len(ids), len(ids))]
        for column, rows in enumerate(aligned):
            chosen = [rows[image_id] for image_id in selected]
            recalls[sample, column], chairis[sample, column] = _metric_from_rows(chosen)
    return {"recall": recalls, "chairi": chairis, "image_ids": np.asarray(ids, dtype=object)}


def _frontier_band(baseline: Sequence[Mapping[str, Any]], bootstrap: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if not baseline:
        return {"x": [], "median": [], "ci95": []}
    x = np.linspace(
        min(float(p["object_recall"]) for p in baseline),
        max(float(p["object_recall"]) for p in baseline),
        101,
    )
    samples: list[np.ndarray] = []
    for recalls, chairis in zip(bootstrap["recall"].T, bootstrap["chairi"].T):
        current = pareto_frontier(
            [
                {"object_recall": recalls[i], "chairi": chairis[i]}
                for i in range(len(baseline))
            ]
        )
        values = [_interpolate(current, value) for value in x]
        samples.append(np.asarray([float("nan") if value is None else value for value in values]))
    matrix = np.asarray(samples)
    return {
        "x": x.tolist(),
        "median": np.nanmedian(matrix, axis=0).tolist(),
        "ci95": np.nanpercentile(matrix, [2.5, 97.5], axis=0).T.tolist(),
    }


def _ellipse(samples_x: np.ndarray, samples_y: np.ndarray) -> dict[str, Any]:
    center = [float(np.mean(samples_x)), float(np.mean(samples_y))]
    covariance = (
        np.cov(np.column_stack([samples_x, samples_y]), rowvar=False)
        if len(samples_x) > 1
        else np.zeros((2, 2))
    )
    covariance = np.atleast_2d(covariance)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        covariance = np.zeros((2, 2))
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    scale = math.sqrt(5.991)  # chi-square(2), 95% confidence region
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    return {"center": center, "width": float(2 * scale * math.sqrt(max(values[0], 0))), "height": float(2 * scale * math.sqrt(max(values[1], 0))), "angle": angle}


def _holm(p_values: Mapping[str, float], alpha: float) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[name] = running
    return {name: {"p_value": p_values[name], "holm_p_value": adjusted[name], "significant": adjusted[name] <= alpha} for name in p_values}


def analyze_frontier(
    manifest: str | Path | Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Analyze all manifest groups and return JSON-serializable statistics."""

    entries = load_manifest(manifest)
    base_dir = Path(manifest).expanduser().parent if isinstance(manifest, (str, Path)) else Path.cwd()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        result = _result_for_entry(entry, base_dir)
        point = _point(entry, result)
        groups.setdefault((point["dataset"], point["model"]), []).append(point)
    output: dict[str, Any] = {"alpha": alpha, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed, "groups": {}}
    for (dataset, model), points in groups.items():
        baselines = [point for point in points if point["kind"] in {"baseline", "trivial", "flat"}]
        published = [point for point in points if point not in baselines]
        if not baselines:
            raise ValueError(f"group {dataset}/{model} has no baseline points")
        all_points = [{key: value for key, value in point.items() if key not in {"details", "image_ids"}} for point in points]
        baseline_front = pareto_frontier(baselines)
        baseline_boot = _bootstrap_metrics(baselines, bootstrap_samples, seed)
        band = _frontier_band(baselines, baseline_boot)
        for index, point in enumerate(baselines):
            point["object_recall_ci95"] = np.quantile(
                baseline_boot["recall"][:, index], [0.025, 0.975]
            ).tolist()
            point["chairi_ci95"] = np.quantile(
                baseline_boot["chairi"][:, index], [0.025, 0.975]
            ).tolist()
        p_values: dict[str, float] = {}
        comparisons: dict[str, dict[str, Any]] = {}
        for index, point in enumerate(published):
            point_boot = _bootstrap_metrics([*baselines, point], bootstrap_samples, seed + index + 1)
            method_x = point_boot["recall"][:, -1]
            method_y = point_boot["chairi"][:, -1]
            point["object_recall_ci95"] = np.quantile(
                method_x, [0.025, 0.975]
            ).tolist()
            point["chairi_ci95"] = np.quantile(
                method_y, [0.025, 0.975]
            ).tolist()
            front_y = []
            for sample in range(bootstrap_samples):
                front = pareto_frontier(
                    [
                        {
                            "object_recall": point_boot["recall"][sample, j],
                            "chairi": point_boot["chairi"][sample, j],
                        }
                        for j in range(len(baselines))
                    ]
                )
                front_y.append(_interpolate(front, float(method_x[sample])))
            valid = np.asarray([value is not None for value in front_y])
            differences = np.asarray([float(front_y[i]) - method_y[i] for i in range(bootstrap_samples) if valid[i]])
            p_value = float((1 + np.sum(differences <= 0)) / (len(differences) + 1)) if len(differences) else 1.0
            p_values[point["name"]] = p_value
            median_diff = float(np.median(differences)) if len(differences) else float("nan")
            comparisons[point["name"]] = {
                "frontier_delta_chairi": median_diff,
                "comparable_bootstrap_fraction": float(np.mean(valid)),
                "p_value": p_value,
                "method_bootstrap": {
                    "recall_ci95": np.quantile(method_x, [0.025, 0.975]).tolist(),
                    "chairi_ci95": np.quantile(method_y, [0.025, 0.975]).tolist(),
                    "ellipse95": _ellipse(method_x, method_y),
                },
            }
        # Preserve the detailed comparison fields while adding Holm results.
        for name, correction in _holm(p_values, alpha).items():
            comparisons[name].update(correction)
            delta = comparisons[name]["frontier_delta_chairi"]
            if comparisons[name]["significant"] and delta > 0:
                comparisons[name]["verdict"] = "超越前沿"
            elif math.isfinite(delta) and delta <= 0:
                comparisons[name]["verdict"] = "劣于前沿"
            else:
                comparisons[name]["verdict"] = "前沿等价"
        output["groups"][f"{dataset}/{model}"] = {"dataset": dataset, "model": model, "points": all_points, "baseline_frontier": [{key: value for key, value in point.items() if key not in {"details", "image_ids"}} for point in baseline_front], "frontier_bootstrap": band, "comparisons": comparisons}
    return output


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _plot_group(group: Mapping[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError as exc:  # pragma: no cover - optional plotting dependency
        raise ImportError("plotting requires matplotlib; install it before running frontier.py") from exc
    fig, axis = plt.subplots(figsize=(8, 5.5))
    band = group["frontier_bootstrap"]
    x = np.asarray(band["x"], dtype=float)
    median = np.asarray(band["median"], dtype=float)
    ci = np.asarray(band["ci95"], dtype=float)
    if len(x):
        axis.fill_between(x, ci[:, 0], ci[:, 1], color="0.75", alpha=0.35, label="baseline frontier 95% CI")
        axis.step(x, median, where="mid", color="black", linewidth=2, label="trivial baseline frontier")
    colors: dict[str, Any] = {}
    for point in group["points"]:
        if point["kind"] in {"baseline", "trivial", "flat"}:
            axis.scatter(point["object_recall"], point["chairi"], color="0.45", marker=".", alpha=0.8)
            continue
        family = point["family"]
        colors.setdefault(family, plt.cm.tab10(len(colors) % 10))
        axis.scatter(point["object_recall"], point["chairi"], color=colors[family], s=60, label=family)
        comparison = group["comparisons"].get(point["name"], {})
        ellipse = comparison.get("method_bootstrap", {}).get("ellipse95")
        if ellipse and ellipse["width"] > 0 and ellipse["height"] > 0:
            axis.add_patch(Ellipse(ellipse["center"], ellipse["width"], ellipse["height"], angle=ellipse["angle"], facecolor=colors[family], alpha=0.12, edgecolor=colors[family]))
    axis.set_xlabel("Object recall")
    axis.set_ylabel("CHAIRi (lower is better)")
    axis.set_title(f"{group['dataset']} / {group['model']}")
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(analysis: Mapping[str, Any], output_dir: str | Path = "results") -> Path:
    """Write analysis JSON, group plots, and a concise Markdown report."""

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "frontier_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Experiment A frontier report", "", f"Bootstrap: {analysis['bootstrap_samples']} samples, seed {analysis['bootstrap_seed']}; Holm alpha={analysis['alpha']}.", ""]
    for key, group in analysis["groups"].items():
        image_name = f"frontier_{_safe_name(key)}.png"
        try:
            _plot_group(group, destination / image_name)
            lines.extend([f"## {key}", "", f"![{key}]({image_name})", ""])
        except ImportError:
            lines.extend([f"## {key}", "", "Plot skipped: install matplotlib to render the PNG.", ""])
        lines.extend(["| Method | Object recall (95% CI) | CHAIRi (95% CI) | Verdict | Holm p |", "|---|---:|---:|---|---:|"])
        for point in group["points"]:
            if point["kind"] in {"baseline", "trivial", "flat"}:
                verdict = "baseline"
                p_value = "-"
            else:
                comparison = group["comparisons"].get(point["name"], {})
                verdict = comparison.get("verdict", "前沿等价")
                p_value = f"{comparison.get('holm_p_value', 1.0):.4f}"
            recall_ci = point.get("object_recall_ci95") or [float("nan"), float("nan")]
            chairi_ci = point.get("chairi_ci95") or [float("nan"), float("nan")]
            lines.append(
                f"| {point['name']} | {point['object_recall']:.4f} "
                f"[{recall_ci[0]:.4f}, {recall_ci[1]:.4f}] | "
                f"{point['chairi']:.4f} [{chairi_ci[0]:.4f}, {chairi_ci[1]:.4f}] | "
                f"{verdict} | {p_value} |"
            )
        lines.append("")
    report = destination / "expA_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON/JSONL result manifest")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    analysis = analyze_frontier(args.manifest, bootstrap_samples=args.bootstrap_samples, seed=args.seed, alpha=args.alpha)
    report = write_report(analysis, args.output_dir)
    print(report)


if __name__ == "__main__":
    main()
