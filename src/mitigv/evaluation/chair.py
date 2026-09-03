"""Strict CHAIR evaluation for COCO image captions.

The lexical vocabulary is the original CHAIR synonym table.  Ground-truth
objects follow the official evaluator: COCO instance categories are unioned
with object mentions found in the COCO reference captions.  The evaluator is
deliberately deterministic and dependency-light; it does not use fuzzy
matching or an LLM.
"""

from __future__ import annotations

import argparse
import importlib.resources as resources
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def default_synonyms_path() -> Path:
    """Return the bundled copy of CHAIR's official synonym table."""

    return Path(resources.files("mitigv.evaluation").joinpath("data/chair_synonyms.txt"))


def load_chair_synonyms(path: str | Path | None = None) -> dict[str, str]:
    """Load the official CHAIR table and return surface -> COCO mappings."""

    source = Path(path) if path is not None else default_synonyms_path()
    groups: list[list[str]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            terms = [part.strip().lower() for part in line.split(",") if part.strip()]
            if terms:
                groups.append(terms)
    if len(groups) != 80:
        raise ValueError(f"CHAIR synonym table must contain 80 groups, got {len(groups)}")
    inverse: dict[str, str] = {}
    for group in groups:
        canonical = group[0]
        if canonical not in COCO_CLASSES:
            raise ValueError(f"unknown COCO canonical object in synonym table: {canonical}")
        for term in group:
            inverse[term] = canonical
    return inverse


def _singularize(word: str) -> str:
    irregular = {
        "people": "person", "children": "child", "men": "man", "women": "woman",
        "geese": "goose", "mice": "mouse", "teeth": "tooth", "feet": "foot",
        "oxen": "ox", "knives": "knife", "skis": "ski",
    }
    if word in irregular:
        return irregular[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ves"):
        return word[:-3] + "f"
    if len(word) > 3 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class ChairLexicon:
    """Exact token/phrase matcher backed by the official synonym table."""

    _SPECIAL = {
        "bow tie": "tie", "toilet seat": "toilet", "wine glas": "wine glass",
        "stove top oven": "oven",
        "home plate": "", "train track": "",
    }
    _DOUBLE_WORDS = {
        "motor bike", "motor cycle", "air plane", "traffic light", "street light",
        "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
        "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
        "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
        "hair drier", "potted plant", "bow tie", "laptop computer", "stove top oven",
    }

    def __init__(self, inverse: Mapping[str, str]):
        self.inverse = dict(inverse)
        self.terms = set(self.inverse)

    def caption_to_objects(self, caption: str) -> tuple[list[str], list[str], list[int], list[str]]:
        raw = [_singularize(x) for x in _TOKEN_RE.findall(str(caption).lower())]
        processed: list[str] = []
        indices: list[int] = []
        i = 0
        while i < len(raw):
            phrase3 = " ".join(raw[i : i + 3])
            phrase2 = " ".join(raw[i : i + 2])
            if phrase3 in self._SPECIAL or phrase3 in self._DOUBLE_WORDS:
                mapped = self._SPECIAL.get(phrase3, self.inverse.get(phrase3, phrase3))
                if mapped:
                    processed.append(mapped)
                    indices.append(i)
                i += 3
            elif phrase2 in self._SPECIAL or phrase2 in self._DOUBLE_WORDS:
                mapped = self._SPECIAL.get(phrase2, phrase2)
                if mapped:
                    processed.append(mapped)
                    indices.append(i)
                i += 2
            else:
                processed.append(raw[i])
                indices.append(i)
                i += 1
        if "toilet" in processed and "seat" in processed:
            processed = [word for word in processed if word != "seat"]
        surfaces: list[str] = []
        canonical: list[str] = []
        token_indices: list[int] = []
        for word, index in zip(processed, indices):
            if word in self.terms:
                surfaces.append(word)
                canonical.append(self.inverse[word])
                token_indices.append(index)
        return surfaces, canonical, token_indices, processed


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _items(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        for key in ("items", "data", "annotations", "results", "predictions"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError(f"{label} must be a list or a supported wrapper object")


def build_ground_truth(
    instances_json: str | Path,
    captions_json: str | Path | None = None,
    image_ids: Iterable[int] | None = None,
    synonyms_file: str | Path | None = None,
) -> dict[int, set[str]]:
    """Build official CHAIR GT object sets for each selected image."""

    inverse = load_chair_synonyms(synonyms_file)
    lexicon = ChairLexicon(inverse)
    instances = _load_json(instances_json)
    if not isinstance(instances, Mapping):
        raise ValueError("instances JSON must be a COCO object")
    categories = {int(item["id"]): str(item["name"]).lower() for item in instances["categories"]}
    selected = {int(x) for x in image_ids} if image_ids is not None else {
        int(image["id"]) for image in instances.get("images", [])
    }
    gt = {image_id: set() for image_id in selected}
    for annotation in instances.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id in gt:
            name = categories[int(annotation["category_id"])]
            gt[image_id].add(inverse.get(name, name))
    if captions_json is not None:
        captions = _load_json(captions_json)
        for annotation in _items(captions, "captions"):
            image_id = int(annotation["image_id"])
            if image_id in gt:
                _, objects, _, _ = lexicon.caption_to_objects(str(annotation.get("caption", "")))
                gt[image_id].update(objects)
    missing = [image_id for image_id, objects in gt.items() if not objects]
    if missing:
        raise ValueError(f"no GT objects found for image ids: {missing[:5]}")
    return gt


def _caption(item: Mapping[str, Any]) -> str:
    for key in ("caption", "generated_text", "text", "answer", "output"):
        if key in item:
            return str(item[key])
    raise KeyError("prediction item must contain caption, generated_text, text, answer, or output")


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class ImageResult:
    image_id: int
    caption: str
    mentioned_objects: tuple[str, ...]
    hallucinated: tuple[str, ...]
    recalled: tuple[str, ...]
    missed: tuple[str, ...]
    gt_objects: tuple[str, ...]
    word_count: int
    sentence_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "caption": self.caption,
            "mentioned_objects": list(self.mentioned_objects),
            "hallucinated": list(self.hallucinated),
            "recalled": list(self.recalled),
            "missed": list(self.missed),
            "gt_objects": list(self.gt_objects),
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
        }


class ChairEvaluator:
    """Strict CHAIR evaluator with image-level bootstrap confidence intervals."""

    def __init__(self, ground_truth: Mapping[int, Iterable[str]], synonyms_file: str | Path | None = None):
        self.ground_truth = {int(key): set(value) for key, value in ground_truth.items()}
        self.lexicon = ChairLexicon(load_chair_synonyms(synonyms_file))

    def evaluate(
        self,
        predictions: Sequence[Mapping[str, Any]],
        *,
        bootstrap_samples: int = 1000,
        seed: int = 42,
    ) -> dict[str, Any]:
        if not predictions:
            raise ValueError("predictions must not be empty")
        details: list[ImageResult] = []
        seen: set[int] = set()
        for item in predictions:
            image_id = int(item["image_id"])
            if image_id in seen:
                raise ValueError(f"duplicate prediction for image_id={image_id}")
            if image_id not in self.ground_truth:
                raise KeyError(f"prediction image_id={image_id} is not in ground truth")
            seen.add(image_id)
            caption = _caption(item)
            _, mentioned, _, _ = self.lexicon.caption_to_objects(caption)
            gt = self.ground_truth[image_id]
            hallucinated = tuple(obj for obj in mentioned if obj not in gt)
            recalled = tuple(sorted(set(mentioned) & gt))
            missed = tuple(sorted(gt - set(mentioned)))
            sentences = len(_SENTENCE_RE.findall(caption.strip())) if caption.strip() else 0
            details.append(ImageResult(
                image_id=image_id,
                caption=caption,
                mentioned_objects=tuple(mentioned),
                hallucinated=hallucinated,
                recalled=recalled,
                missed=missed,
                gt_objects=tuple(sorted(gt)),
                word_count=len(_TOKEN_RE.findall(caption)),
                sentence_count=sentences,
            ))
        point = self._metrics(details)
        rng = random.Random(seed)
        samples = [self._metrics([details[rng.randrange(len(details))] for _ in details]) for _ in range(bootstrap_samples)]
        summary = {
            name: {"value": point[name], "ci95": [_quantile([sample[name] for sample in samples], 0.025), _quantile([sample[name] for sample in samples], 0.975)]}
            for name in point
        }
        return {"summary": summary, "details": [item.to_dict() for item in details], "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed}

    @staticmethod
    def _metrics(details: Sequence[ImageResult]) -> dict[str, float]:
        n = len(details)
        object_mentions = sum(len(item.mentioned_objects) for item in details)
        hallucinated_mentions = sum(len(item.hallucinated) for item in details)
        gt_count = sum(len(item.gt_objects) for item in details)
        recalled_count = sum(len(item.recalled) for item in details)
        mentioned_unique = sum(len(set(item.mentioned_objects)) for item in details)
        precision = recalled_count / mentioned_unique if mentioned_unique else 0.0
        recall = recalled_count / gt_count if gt_count else 0.0
        return {
            "CHAIRs": sum(bool(item.hallucinated) for item in details) / n,
            "CHAIRi": hallucinated_mentions / object_mentions if object_mentions else 0.0,
            "object_recall": recall,
            "object_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "avg_words": sum(item.word_count for item in details) / n,
            "avg_sentences": sum(item.sentence_count for item in details) / n,
        }


def evaluate_chair(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[int, Iterable[str]],
    *,
    synonyms_file: str | Path | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate predictions against already-built GT sets."""

    return ChairEvaluator(ground_truth, synonyms_file=synonyms_file).evaluate(
        predictions, bootstrap_samples=bootstrap_samples, seed=seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-json", required=True)
    dataset_root = Path("~/dataset").expanduser()
    default_annotations = dataset_root / "coco2017" / "annotations"
    parser.add_argument(
        "--instances-json",
        default=str(default_annotations / "instances_val2017.json"),
        help="COCO instances annotations (default: ~/dataset/coco2017/annotations/instances_val2017.json)",
    )
    parser.add_argument(
        "--captions-json",
        default=str(default_annotations / "captions_val2017.json"),
        help="COCO reference captions (default: ~/dataset/coco2017/annotations/captions_val2017.json)",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--synonyms-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    for path, label in ((args.instances_json, "instances"), (args.captions_json, "captions"), (args.generated_json, "generated")):
        if path and not Path(path).expanduser().exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
    predictions = _items(_load_json(args.generated_json), "generated")
    gt = build_ground_truth(args.instances_json, args.captions_json, synonyms_file=args.synonyms_file)
    result = evaluate_chair(predictions, gt, synonyms_file=args.synonyms_file, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        destination = Path(args.output_json).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
