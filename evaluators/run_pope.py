"""End-to-end POPE validation of MitigV's VCD against the paper's reference.

Runs regular sampling (baseline) and VCD on the POPE benchmark using a
HuggingFace LLaVA-1.5-7B checkpoint, then reports the metrics alongside the VCD
paper's Table 1 numbers and judges whether the differences are ignorable.

Example:
    python -m evaluators.run_pope --device cuda:2 --limit 500
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from PIL import Image

from evaluators.pope import METRIC_NAMES, compare_to_reference, compute_metrics, load_pope
from mitigv.algorithms.vcd import VCD
from mitigv.backends.hf import HFMitigator

#: LLaVA-1.5 ``llava_v1`` conversation system prompt (from the official repo).
SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def build_prompt(question: str) -> str:
    """Build the exact LLaVA-1.5 prompt used by the official VCD POPE eval."""
    return (
        f"{SYSTEM} USER: <image>\n{question} Please answer this question with one word. ASSISTANT:"
    )


def load_model(model_path: str, device: str):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def run_split(mitigator, items, image_folder: str, batch_size: int) -> list[str]:
    """Run ``mitigator`` over POPE items and return the generated texts."""
    texts: list[str] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        images = [
            Image.open(os.path.join(image_folder, it["image"])).convert("RGB")
            for it in chunk
        ]
        prompts = [build_prompt(it["text"]) for it in chunk]
        out = mitigator(images, prompts)
        if isinstance(out, str):
            out = [out]
        texts.extend(out)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.path.expanduser("~/checkpoints/llava-1.5-7b-hf"))
    parser.add_argument("--image-folder", default=os.path.expanduser("~/dataset/coco/val2014"))
    parser.add_argument("--pope-dir", default=os.path.expanduser("~/dataset/POPE"))
    parser.add_argument("--splits", nargs="+", default=["random", "popular", "adversarial"])
    parser.add_argument("--limit", type=int, default=500, help="max samples per split")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--noise-step", type=int, default=999, help="T for POPE per paper")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, processor = load_model(args.model_path, args.device)

    base = HFMitigator(
        model, processor, do_sample=True, temperature=1.0, max_new_tokens=args.max_new_tokens
    )
    vcd = VCD(
        model,
        processor,
        alpha=args.alpha,
        beta=args.beta,
        distortion="diffusion_noise",
        distortion_kwargs={"noise_step": args.noise_step},
        do_sample=True,
        temperature=1.0,
        max_new_tokens=args.max_new_tokens,
    )

    header = f"{'Split':<12} {'Method':<8} " + " ".join(f"{m.capitalize():>8}" for m in METRIC_NAMES) + "  dAcc  Verdict"
    print(header)
    print("-" * len(header))

    overall_ignorable = True
    for split in args.splits:
        items = load_pope(os.path.join(args.pope_dir, f"coco_pope_{split}.json"))[: args.limit]
        gt = [it["label"] for it in items]
        for name, mitigator in (("Regular", base), ("VCD", vcd)):
            t0 = time.time()
            gen = run_split(mitigator, items, args.image_folder, args.batch_size)
            metrics = compute_metrics(gt, gen)
            cmp = compare_to_reference(metrics, split, name, args.tolerance)
            verdict = "ok" if cmp["ignorable"] else "DIFF"
            overall_ignorable &= cmp["ignorable"]
            cells = " ".join(f"{metrics[m]:>8.2f}" for m in METRIC_NAMES)
            print(
                f"{split:<12} {name:<8} {cells}  {cmp['diffs']['accuracy']:>+5.2f}  {verdict}"
                f"  ({time.time() - t0:.1f}s)"
            )

    print("-" * len(header))
    print(
        f"Verdict: differences {'within' if overall_ignorable else 'EXCEED'} "
        f"tolerance (+/-{args.tolerance}) -> "
        f"{'IGNORABLE' if overall_ignorable else 'NOT IGNORABLE'}"
    )


if __name__ == "__main__":
    main()
