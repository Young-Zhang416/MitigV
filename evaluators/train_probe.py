"""Train a linear object-presence probe for :class:`~mitigv.algorithms.probe_steer.LinearProbeSteer`.

Fits a linear classifier ``presence = sigmoid(w . h + b)`` on top of the frozen
LVLM's intermediate hidden states (last token, at one decoder layer) for
``(image, object) -> present/absent`` pairs sampled from COCO. The probe's weight
vector ``w`` is the decision normal used as the steering direction at inference.

No model weight is updated — only the linear head is fit — so the LVLM stays
training-free. ~2000 samples take about half an hour on a single GPU.

Example:
    PYTHONPATH=src python -m evaluators.train_probe \
        --device cuda:2 --n-samples 2000 --layer 16 \
        --output ~/checkpoints/llava15_presence_probe.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from PIL import Image

#: LLaVA-1.5 ``llava_v1`` system prompt (same as inference).
SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def build_prompt(category: str) -> str:
    return (
        f"{SYSTEM} USER: <image>\n"
        f"Is there a {category} in this image? Please answer this question with one word. "
        "ASSISTANT:"
    )


def load_annotations(ann_file: str) -> tuple[dict[str, str], dict[int, set[int]]]:
    """Return ``(id -> category name, image_id -> set of category ids)``."""
    with open(ann_file) as f:
        data = json.load(f)
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    image_presence: dict[int, set[int]] = {}
    for ann in data["annotations"]:
        image_presence.setdefault(ann["image_id"], set()).add(ann["category_id"])
    return id2name, image_presence


def sample_pairs(
    image_presence: dict[int, set[int]], id2name: dict[str, str], n: int, seed: int
) -> list[tuple[int, str, int]]:
    """Return ``n`` balanced ``(image_id, category_name, label)`` triples."""
    rng = random.Random(seed)
    ids = list(image_presence)
    names = list(id2name.values())
    pairs: list[tuple[int, str, int]] = []
    for half in range(2):
        label = 1 - half  # present first, then absent
        while len([p for p in pairs if p[2] == label]) < n // 2:
            img_id = rng.choice(ids)
            present = image_presence[img_id]
            if label == 1 and not present:
                continue
            cat_id = rng.choice(sorted(present)) if label == 1 else rng.choice(
                [c for c in id2name if c not in present]
            )
            pairs.append((img_id, id2name[cat_id], label))
    return pairs


def extract_features(model, processor, image_dir, pairs, layer, device, dtype):
    """Forward each pair and collect the last-token hidden state at ``layer``."""
    features, labels = [], []
    for i, (img_id, category, label) in enumerate(pairs):
        path = os.path.join(image_dir, f"{img_id:012d}.jpg")
        image = Image.open(path).convert("RGB")
        inputs = processor(text=build_prompt(category), images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                inputs[k] = v.to(dtype)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        features.append(out.hidden_states[layer + 1][0, -1, :].to(torch.float32).cpu())
        labels.append(label)
        if (i + 1) % 200 == 0:
            print(f"  extracted {i + 1}/{len(pairs)}")
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32)


def train_probe(X: torch.Tensor, y: torch.Tensor, epochs: int = 200, lr: float = 1e-2):
    """Fit a logistic-regression probe and return ``(weight, bias, accuracy)``."""
    H = X.shape[1]
    w = torch.nn.Parameter(torch.zeros(H))
    b = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(X @ w + b, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (X @ w + b) > 0
        acc = (pred == (y > 0.5)).float().mean().item()
    return w.detach().clone(), b.detach().clone(), acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.path.expanduser("~/checkpoints/llava-1.5-7b-hf"))
    parser.add_argument("--image-dir", default=os.path.expanduser("~/dataset/coco2017/train2017"))
    parser.add_argument("--ann-file", default=os.path.expanduser("~/dataset/coco2017/annotations/instances_train2017.json"))
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--output", default=os.path.expanduser("~/checkpoints/llava15_presence_probe.pt"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    torch.manual_seed(args.seed)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    dtype = next(model.parameters()).dtype

    id2name, image_presence = load_annotations(args.ann_file)
    pairs = sample_pairs(image_presence, id2name, args.n_samples, args.seed)
    print(f"sampled {len(pairs)} (image, object, label) pairs")

    t0 = time.time()
    X, y = extract_features(model, processor, args.image_dir, pairs, args.layer, args.device, dtype)
    print(f"extracted {X.shape} features in {time.time() - t0:.1f}s")

    weight, bias, acc = train_probe(X, y)
    print(f"probe accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "weight": weight,          # (hidden_dim,) decision normal
            "bias": bias,              # scalar
            "layer": args.layer,
            "accuracy": acc,
            "n_samples": args.n_samples,
            "category_vocab": list(id2name.values()),
        },
        args.output,
    )
    print(f"saved probe to {args.output}")


if __name__ == "__main__":
    main()
