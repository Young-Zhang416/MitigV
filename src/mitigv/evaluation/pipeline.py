"""Run a captioning pipeline over image/GT records and evaluate strict CHAIR."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mitigv.evaluation.chair import ChairEvaluator, _caption, _items, load_chair_synonyms

__all__ = ["load_pipeline_records", "evaluate_pipeline_json"]


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    records = _items(value, "input records") if not isinstance(value, list) else value
    if not records or not all(isinstance(item, Mapping) for item in records):
        raise ValueError("input JSON must contain a non-empty list of objects")
    return [dict(item) for item in records]


load_pipeline_records = _load_records


def _resolve_image(file_name: str, image_root: str | Path) -> Path:
    name = Path(file_name).expanduser()
    if name.is_absolute() and name.is_file():
        return name
    root = Path(image_root).expanduser()
    direct = root / name
    if direct.is_file():
        return direct
    basename = Path(file_name).name
    candidate = root / basename
    if candidate.is_file():
        return candidate
    # COCO records often include a directory prefix while users point at the
    # dataset root. Search by basename only as a final, deterministic fallback.
    matches = sorted(path for path in root.rglob(basename) if path.is_file())
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"image {file_name!r} was not found below image_root={str(root)!r}"
    )


def _pipeline_caption(pipeline: Callable[..., Any], image: Any, prompt: str) -> str:
    """Call MitigV/Transformers-style pipelines and normalize their output."""

    try:
        result = pipeline(image, prompt)
    except TypeError as first_error:
        try:
            result = pipeline(image=image, text=prompt)
        except TypeError:
            try:
                result = pipeline({"image": image, "text": prompt})
            except TypeError:
                raise first_error
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        return _caption(result)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if len(result) != 1:
            raise ValueError("pipeline must return one caption per single image")
        item = result[0]
        if isinstance(item, str):
            return item
        if isinstance(item, Mapping):
            return _caption(item)
    raise TypeError(
        "pipeline output must be a caption string, mapping, or a one-item sequence"
    )


def evaluate_pipeline_json(
    records_json: str | Path,
    pipeline: Callable[..., Any],
    *,
    image_root: str | Path = "~/dataset",
    prompt: str = "Describe the image in one sentence.",
    output_json: str | Path | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate captions for records and evaluate strict CHAIR.

    Each record must contain ``image_id``, ``file_name`` and ``gt_objects``.
    ``pipeline`` may be a MitigV mitigator (``pipeline(image, prompt)``) or a
    compatible callable accepting ``image=``/``text=``. The returned object
    includes generated captions, resolved paths, per-image CHAIR details and
    bootstrap confidence intervals.
    """

    records = _load_records(records_json)
    ground_truth: dict[int, set[str]] = {}
    synonyms = load_chair_synonyms()
    predictions: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional runtime
        raise ImportError("pipeline evaluation requires Pillow") from error

    for record in records:
        for required in ("image_id", "file_name", "gt_objects"):
            if required not in record:
                raise KeyError(f"input record is missing required field {required!r}")
        image_id = int(record["image_id"])
        if image_id in ground_truth:
            raise ValueError(f"duplicate image_id={image_id}")
        objects = record["gt_objects"]
        if isinstance(objects, (str, bytes)) or not isinstance(objects, Sequence):
            raise TypeError(f"gt_objects for image_id={image_id} must be a list")
        ground_truth[image_id] = {
            synonyms.get(str(item).strip().lower(), str(item).strip().lower())
            for item in objects
            if str(item).strip()
        }
        if not ground_truth[image_id]:
            raise ValueError(f"gt_objects for image_id={image_id} must not be empty")
        image_path = _resolve_image(str(record["file_name"]), image_root)
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        caption = _pipeline_caption(pipeline, image, prompt)
        predictions.append({"image_id": image_id, "caption": caption})
        generated.append(
            {
                "image_id": image_id,
                "file_name": str(record["file_name"]),
                "image_path": str(image_path),
                "caption": caption,
            }
        )

    chair = ChairEvaluator(ground_truth).evaluate(
        predictions, bootstrap_samples=bootstrap_samples, seed=seed
    )
    result = {
        "input_json": str(Path(records_json).expanduser()),
        "image_root": str(Path(image_root).expanduser()),
        "prompt": prompt,
        "num_images": len(generated),
        "generated": generated,
        "chair": chair,
    }
    if output_json is not None:
        destination = Path(output_json).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--image-root", default="~/dataset")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prompt", default="Describe the image in one sentence.")
    parser.add_argument("--algorithm", default="vcd")
    parser.add_argument("--model-type", required=True, choices=("llava", "qwen2.5-vl"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from mitigv import load_mitigator

    model_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
    pipeline = load_mitigator(
        args.algorithm,
        model_type=args.model_type,
        model_id=os.path.expanduser(args.model_id),
        model_kwargs=model_kwargs,
        alpha=args.alpha,
        beta=args.beta,
        max_new_tokens=args.max_new_tokens,
    )
    if args.device is not None and hasattr(pipeline.model, "to"):
        pipeline.model.to(args.device)
    evaluate_pipeline_json(
        args.input_json,
        pipeline,
        image_root=args.image_root,
        prompt=args.prompt,
        output_json=args.output_json,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
