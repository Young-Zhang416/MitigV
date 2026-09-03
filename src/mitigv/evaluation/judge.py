"""DeepSeek + GroundingDINO supplementary CHAIR judge.

This module is intentionally separate from strict CHAIR.  DeepSeek extracts
open-vocabulary noun phrases, then one resident GroundingDINO service verifies
each phrase on the original image.  The model and API clients are injectable,
so the evaluator is testable without network access or a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from mitigv.evaluation.chair import build_ground_truth, _load_json, _items

PROMPT_PATH = Path(__file__).with_name("prompts") / "extract_objects.txt"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-base"


class ObjectExtractor(Protocol):
    def extract(self, caption: str) -> list[dict[str, str]]: ...


class GroundingService(Protocol):
    def verify(self, image: Any, object_name: str) -> dict[str, Any]: ...


def _load_prompt(caption: str) -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace("{{caption}}", caption)


class DeepSeekObjectExtractor:
    """DeepSeek JSON extractor with caption-hash disk caching and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "deepseek-chat",
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        cache_dir: str | Path = ".chair_cache/deepseek",
        max_retries: int = 3,
        timeout: float = 60.0,
        backoff: float = 1.0,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required (pass api_key or set DEEPSEEK_API_KEY)")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max(0, int(max_retries))
        self.timeout = timeout
        self.backoff = backoff

    def _cache_path(self, caption: str) -> Path:
        digest = hashlib.sha256(caption.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    @staticmethod
    def _validate(payload: Any) -> list[dict[str, str]]:
        if not isinstance(payload, Mapping) or not isinstance(payload.get("objects"), list):
            raise ValueError("DeepSeek response must be {\"objects\": [...]}")
        objects: list[dict[str, str]] = []
        for item in payload["objects"]:
            if not isinstance(item, Mapping):
                raise ValueError("each extracted object must be an object")
            name = item.get("name")
            head = item.get("attribute_free_head")
            if not isinstance(name, str) or not name.strip() or not isinstance(head, str) or not head.strip():
                raise ValueError("each object needs non-empty name and attribute_free_head")
            objects.append({"name": name.strip(), "attribute_free_head": head.strip()})
        return objects

    def extract(self, caption: str) -> list[dict[str, str]]:
        path = self._cache_path(caption)
        if path.exists():
            try:
                return self._validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError):
                path.unlink(missing_ok=True)
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": _load_prompt(caption)}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    content = content.strip()
                    if content.startswith("```"):
                        content = content.split("\n", 1)[1] if "\n" in content else content
                        content = content.rsplit("```", 1)[0].strip()
                    parsed = json.loads(content)
                else:
                    parsed = content
                objects = self._validate(parsed)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"objects": objects}, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, path)
                return objects
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
        raise RuntimeError(f"DeepSeek extraction failed after {self.max_retries + 1} attempts") from last_error


class GroundingDINOService:
    """Single-device resident GroundingDINO service.

    The processor/model are loaded once in ``__init__`` and reused for every
    image/object query.  ``device`` can be ``cuda:0`` (or any torch device).
    """

    def __init__(self, model_id: str = DEFAULT_DINO_MODEL, device: str = "cuda:0", box_threshold: float = 0.0):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:  # pragma: no cover - optional runtime
            raise ImportError("GroundingDINOService requires torch and transformers") from error
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.box_threshold = box_threshold

    def verify(self, image: Any, object_name: str) -> dict[str, Any]:
        query = object_name.rstrip(".") + "."
        inputs = self.processor(images=image, text=[[query]], return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        size = getattr(image, "size", None)
        if size is None:
            raise TypeError("GroundingDINO image must expose a PIL-like .size=(width,height)")
        target_size = [size[1], size[0]]
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.box_threshold,
            target_sizes=[target_size],
        )[0]
        scores = processed.get("scores", [])
        if len(scores) == 0:
            return {"confirmed": False, "score": 0.0, "box": None, "label": object_name}
        best = int(scores.argmax().item()) if hasattr(scores, "argmax") else max(range(len(scores)), key=lambda i: float(scores[i]))
        score = float(scores[best])
        boxes = processed.get("boxes")
        box = boxes[best].detach().cpu().tolist() if boxes is not None else None
        labels = processed.get("text_labels") or processed.get("labels") or [object_name]
        label = str(labels[best]) if len(labels) > best else object_name
        return {"confirmed": score > 0.35, "score": score, "box": box, "label": label}


def _open_image(path: str | Path) -> Any:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise ImportError("opening judge images requires pillow") from error
    return Image.open(path).convert("RGB")


class DoubleJudgeEvaluator:
    """Evaluate open-vocabulary object claims with DeepSeek and DINO."""

    def __init__(self, ground_truth: Mapping[int, Iterable[str]], extractor: ObjectExtractor, grounding: GroundingService):
        self.ground_truth = {int(key): set(value) for key, value in ground_truth.items()}
        self.extractor = extractor
        self.grounding = grounding

    def evaluate(
        self,
        predictions: Sequence[Mapping[str, Any]],
        *,
        image_paths: Mapping[int, str | Path] | None = None,
        audit_path: str | Path | None = None,
        audit_size: int = 500,
        seed: int = 42,
        bootstrap_samples: int = 1000,
    ) -> dict[str, Any]:
        if not predictions:
            raise ValueError("predictions must not be empty")
        details: list[dict[str, Any]] = []
        for item in predictions:
            image_id = int(item["image_id"])
            caption = str(item.get("caption", item.get("generated_text", item.get("text", ""))))
            if image_id not in self.ground_truth:
                raise KeyError(f"prediction image_id={image_id} is not in ground truth")
            path = item.get("image_path", item.get("image"))
            if path is None and image_paths is not None:
                path = image_paths.get(image_id)
            if path is None:
                raise KeyError(f"no image path for image_id={image_id}")
            extracted = self.extractor.extract(caption)
            judgments: list[dict[str, Any]] = []
            with _open_image(path) as image:
                for obj in extracted:
                    verdict = dict(self.grounding.verify(image, obj["attribute_free_head"]))
                    verdict.update(obj)
                    judgments.append(verdict)
            verified = [obj for obj in judgments if bool(obj.get("confirmed"))]
            verified_names = {str(obj["attribute_free_head"]).lower() for obj in verified}
            gt = self.ground_truth[image_id]
            details.append({
                "image_id": image_id,
                "image_path": str(path),
                "caption": caption,
                "extracted_objects": extracted,
                "dino_judgments": judgments,
                "dino_verified_objects": sorted(verified_names),
                "gt_objects": sorted(gt),
            })
        metrics = self._metrics(details)
        rng = random.Random(seed)
        boot = [
            self._metrics([details[rng.randrange(len(details))] for _ in details])
            for _ in range(bootstrap_samples)
        ]
        for name in ("open_vocab_precision", "open_vocab_recall", "dino_verified_rate", "gt_object_recall"):
            values = sorted(item[name] for item in boot)
            metrics[name] = {
                "value": metrics[name],
                "ci95": [self._quantile(values, 0.025), self._quantile(values, 0.975)],
            }
        if audit_path is not None:
            self.write_audit_sample(details, audit_path, size=audit_size, seed=seed)
        return {"summary": metrics, "details": details, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed}

    @staticmethod
    def _quantile(values: Sequence[float], probability: float) -> float:
        if not values:
            return 0.0
        position = (len(values) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    def _metrics(self, details: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        extracted = 0
        verified = 0
        predicted_count = 0
        gt_total = 0
        gt_recalled = 0
        union_total = 0
        union_recalled = 0
        for item in details:
            predicted_names = [str(obj["attribute_free_head"]).lower() for obj in item["extracted_objects"]]
            supported = set(item["dino_verified_objects"])
            gt = set(item["gt_objects"])
            extracted += len(predicted_names)
            verified += sum(name in supported for name in predicted_names)
            predicted_count += len(supported)
            gt_total += len(gt)
            gt_recalled += len(supported & gt)
            union_total += len(supported | gt)
            union_recalled += len(supported & gt)
        # Reference = (LLM extraction AND DINO support) UNION COCO GT. Thus
        # recall measures coverage of the union and precision measures grounding
        # support among all extracted claims.
        return {
            "open_vocab_precision": gt_recalled / predicted_count if predicted_count else 0.0,
            "open_vocab_recall": union_recalled / union_total if union_total else 0.0,
            "dino_verified_rate": verified / extracted if extracted else 0.0,
            "gt_object_recall": gt_recalled / gt_total if gt_total else 0.0,
            "num_extracted_mentions": float(extracted),
            "num_dino_verified_mentions": float(verified),
            "num_union_objects": float(union_total),
        }

    @staticmethod
    def write_audit_sample(details: Sequence[Mapping[str, Any]], path: str | Path, *, size: int = 500, seed: int = 42) -> None:
        rng = random.Random(seed)
        selected = list(details)
        rng.shuffle(selected)
        selected = selected[: min(size, len(selected))]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for item in selected:
                handle.write(json.dumps({
                    "image_path": item["image_path"],
                    "caption": item["caption"],
                    "extracted_objects": item["extracted_objects"],
                    "dino_judgments": item["dino_judgments"],
                }, ensure_ascii=False) + "\n")


def evaluate_double_judge(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[int, Iterable[str]],
    extractor: ObjectExtractor,
    grounding: GroundingService,
    *,
    image_paths: Mapping[int, str | Path] | None = None,
    audit_path: str | Path | None = "results/judge_audit_sample.jsonl",
    audit_size: int = 500,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`DoubleJudgeEvaluator`."""

    return DoubleJudgeEvaluator(ground_truth, extractor, grounding).evaluate(
        predictions,
        image_paths=image_paths,
        audit_path=audit_path,
        audit_size=audit_size,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def load_image_paths(instances_json: str | Path, image_root: str | Path) -> dict[int, str]:
    """Map COCO image ids to local image paths."""

    data = _load_json(instances_json)
    root = Path(image_root).expanduser()
    return {int(item["id"]): str(root / item["file_name"]) for item in data.get("images", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-json", required=True)
    parser.add_argument("--instances-json", default="~/dataset/coco2017/annotations/instances_val2017.json")
    parser.add_argument("--captions-json", default="~/dataset/coco2017/annotations/captions_val2017.json")
    parser.add_argument("--image-root", default="~/dataset/coco2017/images/val2017")
    parser.add_argument("--output-json", default="results/judge.json")
    parser.add_argument("--audit-path", default="results/judge_audit_sample.jsonl")
    parser.add_argument("--deepseek-cache", default=".chair_cache/deepseek")
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generated = _items(_load_json(args.generated_json), "generated")
    instances = str(Path(args.instances_json).expanduser())
    gt = build_ground_truth(instances, str(Path(args.captions_json).expanduser()))
    image_paths = load_image_paths(instances, args.image_root)
    result = DoubleJudgeEvaluator(
        gt,
        DeepSeekObjectExtractor(cache_dir=args.deepseek_cache),
        GroundingDINOService(model_id=args.dino_model, device=args.device),
    ).evaluate(generated, image_paths=image_paths, audit_path=args.audit_path, seed=args.seed)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
