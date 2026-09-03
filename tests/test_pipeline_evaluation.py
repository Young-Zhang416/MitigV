import json

from PIL import Image

from mitigv.evaluation.pipeline import evaluate_pipeline_json, load_pipeline_records


def test_pipeline_evaluation_reads_records_calls_pipeline_and_writes_result(tmp_path):
    image_path = tmp_path / "COCO_val2014_000000000001.jpg"
    Image.new("RGB", (8, 8), "white").save(image_path)
    input_path = tmp_path / "records.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "file_name": image_path.name,
                    "gt_objects": ["cow", "person", "umbrella"],
                }
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def pipeline(image, prompt):
        calls.append((image.size, prompt))
        return "A person with an umbrella and a dog."

    output_path = tmp_path / "result.json"
    result = evaluate_pipeline_json(
        input_path,
        pipeline,
        image_root=tmp_path,
        output_json=output_path,
        bootstrap_samples=10,
    )
    assert calls == [((8, 8), "Describe the image in one sentence.")]
    assert result["num_images"] == 1
    assert result["generated"][0]["caption"].startswith("A person")
    assert result["chair"]["details"][0]["hallucinated"] == ["dog"]
    assert json.loads(output_path.read_text(encoding="utf-8"))["num_images"] == 1


def test_pipeline_records_accept_wrapped_object(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"data": [{"image_id": 1, "file_name": "x.jpg", "gt_objects": ["cat"]}]}), encoding="utf-8")
    assert load_pipeline_records(path)[0]["image_id"] == 1


def test_pipeline_records_accept_single_record_object(tmp_path):
    path = tmp_path / "record.json"
    path.write_text(
        json.dumps({"image_id": 1, "file_name": "x.jpg", "gt_objects": ["cat"]}),
        encoding="utf-8",
    )
    assert load_pipeline_records(path) == [
        {"image_id": 1, "file_name": "x.jpg", "gt_objects": ["cat"]}
    ]


def test_pipeline_evaluation_accepts_empty_gt_objects(tmp_path):
    image_path = tmp_path / "empty.jpg"
    Image.new("RGB", (8, 8), "white").save(image_path)
    input_path = tmp_path / "records.json"
    input_path.write_text(
        json.dumps(
            [{"image_id": 1, "file_name": image_path.name, "gt_objects": []}]
        ),
        encoding="utf-8",
    )

    result = evaluate_pipeline_json(
        input_path,
        lambda image, prompt: "A dog.",
        image_root=tmp_path,
        bootstrap_samples=10,
    )
    detail = result["chair"]["details"][0]
    assert detail["gt_objects"] == []
    assert detail["recalled"] == []
    assert detail["missed"] == []
    assert detail["hallucinated"] == ["dog"]
    assert result["chair"]["summary"]["CHAIRi"]["value"] == 1.0
