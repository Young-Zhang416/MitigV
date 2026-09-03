import json

from PIL import Image

from mitigv.evaluation.judge import DoubleJudgeEvaluator


class Extractor:
    def extract(self, caption):
        return [
            {"name": "red car", "attribute_free_head": "car"},
            {"name": "dog", "attribute_free_head": "dog"},
        ]


class Grounding:
    def verify(self, image, object_name):
        return {"confirmed": object_name == "car", "score": 0.8 if object_name == "car" else 0.1, "box": [0, 0, 1, 1], "label": object_name}


def test_double_judge_outputs_open_vocab_metrics_and_audit(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4), "white").save(image_path)
    output = DoubleJudgeEvaluator({1: {"car"}}, Extractor(), Grounding()).evaluate(
        [{"image_id": 1, "caption": "A car and a dog."}],
        image_paths={1: image_path},
        audit_path=tmp_path / "audit.jsonl",
        bootstrap_samples=20,
        seed=3,
    )
    assert output["summary"]["open_vocab_precision"]["value"] == 1.0
    assert len(output["summary"]["open_vocab_recall"]["ci95"]) == 2
    row = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert row["image_path"] == str(image_path)
    assert row["dino_judgments"][0]["confirmed"] is True
