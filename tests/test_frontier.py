import json

from mitigv.evaluation.frontier import analyze_frontier, pareto_frontier, write_report


def _result(rows):
    return {"chair": {"details": rows}}


def _row(image_id, gt, recalled, hallucinated=()):
    mentioned = list(recalled) + list(hallucinated)
    return {
        "image_id": image_id,
        "gt_objects": list(gt),
        "recalled": list(recalled),
        "hallucinated": list(hallucinated),
        "mentioned_objects": mentioned,
    }


def test_pareto_frontier_prefers_more_recall_and_less_hallucination():
    points = [
        {"name": "a", "object_recall": 0.5, "chairi": 0.1},
        {"name": "b", "object_recall": 0.8, "chairi": 0.2},
        {"name": "c", "object_recall": 0.7, "chairi": 0.3},
    ]
    assert [point["name"] for point in pareto_frontier(points)] == ["a", "b"]


def test_frontier_analysis_and_report_apply_holm(tmp_path):
    rows = [_row(1, ["cat", "dog"], ["cat"]), _row(2, ["cat", "dog"], ["cat"])]
    better = [_row(1, ["cat", "dog"], ["cat"]), _row(2, ["cat", "dog"], ["cat"])]
    # Keep recall matched while removing the baseline's hallucinated mention.
    rows[0]["hallucinated"] = ["bird"]
    rows[1]["hallucinated"] = ["bird"]
    rows[0]["mentioned_objects"] = ["cat", "bird"]
    rows[1]["mentioned_objects"] = ["cat", "bird"]
    baseline_path = tmp_path / "baseline.json"
    method_path = tmp_path / "method.json"
    baseline_path.write_text(json.dumps(_result(rows)), encoding="utf-8")
    method_path.write_text(json.dumps(_result(better)), encoding="utf-8")
    manifest = [
        {"name": "temperature", "kind": "baseline", "dataset": "coco", "model": "llava", "result": baseline_path.name},
        {"name": "vcd", "kind": "published", "family": "contrastive", "dataset": "coco", "model": "llava", "result": method_path.name},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = analyze_frontier(manifest_path, bootstrap_samples=20)
    group = result["groups"]["coco/llava"]
    assert group["points"][1]["object_recall"] == 0.5
    assert group["comparisons"]["vcd"]["holm_p_value"] <= 0.05
    report = write_report(result, tmp_path / "out")
    assert report.exists()
    assert (tmp_path / "out" / "frontier_analysis.json").exists()
