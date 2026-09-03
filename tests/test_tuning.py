import json

from PIL import Image

from mitigv.evaluation.tuning import expand_parameter_grid, tune_mitigator


def _records(tmp_path):
    image = tmp_path / "one.jpg"
    Image.new("RGB", (4, 4), "white").save(image)
    records = tmp_path / "records.json"
    records.write_text(
        json.dumps([{"image_id": 1, "file_name": image.name, "gt_objects": ["cat"]}]),
        encoding="utf-8",
    )
    return records


def test_expand_parameter_grid_is_deterministic_and_supports_empty_grid():
    assert expand_parameter_grid({"alpha": [0.0, 1.0], "beta": [0.1, 0.2]}) == [
        {"alpha": 0.0, "beta": 0.1},
        {"alpha": 0.0, "beta": 0.2},
        {"alpha": 1.0, "beta": 0.1},
        {"alpha": 1.0, "beta": 0.2},
    ]
    assert expand_parameter_grid({}) == [{}]


def test_tuner_selects_lowest_chairi_and_keeps_candidate_results(tmp_path):
    records = _records(tmp_path)

    def factory(params):
        # alpha=1 mentions only GT; alpha=0 adds a hallucinated dog.
        caption = "A cat." if params["alpha"] == 1 else "A cat and dog."
        return lambda image, prompt: caption

    result = tune_mitigator(
        records,
        param_grid={"alpha": [0, 1]},
        pipeline_factory=factory,
        image_root=tmp_path,
        bootstrap_samples=5,
    )
    assert result["best_params"] == {"alpha": 1}
    assert result["metric"] == "CHAIRi"
    assert result["maximize"] is False
    assert len(result["candidates"]) == 2
    assert result["best_evaluation"]["chair"]["summary"]["CHAIRi"]["value"] == 0


def test_tuner_can_maximize_object_f1(tmp_path):
    records = _records(tmp_path)

    def factory(params):
        caption = "A cat." if params["include"] else "A dog."
        return lambda image, prompt: caption

    result = tune_mitigator(
        records,
        param_grid={"include": [False, True]},
        pipeline_factory=factory,
        image_root=tmp_path,
        metric="object_f1",
        bootstrap_samples=2,
    )
    assert result["best_params"] == {"include": True}
    assert result["maximize"] is True
