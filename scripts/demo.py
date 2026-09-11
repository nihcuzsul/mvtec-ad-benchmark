#!/usr/bin/env python3
"""CLI demo for single-image anomaly inference and visualization."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.checkpoint import load_checkpoint, load_checkpoint_metadata
from benchmark.demo_utils import (
    preprocess_image,
    generate_heatmap,
    generate_overlay,
    generate_gt_visualization,
)
from benchmark.threshold import apply_threshold


MVTEC_AD_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image anomaly detection demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", required=True, type=Path,
        help="Path to input image (PNG or JPG)",
    )
    parser.add_argument(
        "--model", default=None, choices=["padim", "patchcore"],
        help="Model architecture (auto-detected from checkpoint if omitted)",
    )
    parser.add_argument(
        "--category", default=None,
        help="MVTec AD category (auto-detected from checkpoint if omitted)",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to fitted model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./demo_outputs"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--gt-mask", type=Path, default=None,
        help="Optional ground truth mask for visualization only",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for inference",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        return 1

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    from benchmark.config import resolve_device
    device = resolve_device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        metadata = load_checkpoint_metadata(args.checkpoint)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    model_name = args.model or metadata["model_name"]
    category = args.category or metadata["category"]

    if category not in MVTEC_AD_CATEGORIES:
        print(
            f"ERROR: Unknown category '{category}'. "
            f"Must be one of: {MVTEC_AD_CATEGORIES}",
            file=sys.stderr,
        )
        return 1

    try:
        adapter, image_threshold, pixel_threshold = load_checkpoint(
            args.checkpoint,
            device=device,
            expected_category=category,
            expected_model=model_name,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Preprocessing image: {args.image}")
    image_tensor = preprocess_image(args.image)

    print("Running inference...")
    start = time.time()
    anomaly_map, anomaly_score = adapter.predict(image_tensor.to(device))
    latency_ms = (time.time() - start) * 1000

    score = float(anomaly_score.cpu().item())
    prediction = apply_threshold(anomaly_score.cpu(), image_threshold)
    label = "ANOMALOUS" if prediction.item() == 1 else "NORMAL"

    print()
    print("=" * 50)
    print(f"Model:      {model_name}")
    print(f"Category:   {category}")
    print(f"Image:      {args.image}")
    print(f"Score:      {score:.4f}")
    print(f"Threshold:  {image_threshold:.4f}")
    print(f"Prediction: {label}")
    print(f"Latency:    {latency_ms:.2f} ms")
    print("=" * 50)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = output_dir / "heatmap.png"
    overlay_path = output_dir / "overlay.png"
    result_path = output_dir / "result.json"

    print(f"\nGenerating heatmap: {heatmap_path}")
    generate_heatmap(anomaly_map.cpu().squeeze(0), heatmap_path)

    print(f"Generating overlay: {overlay_path}")
    generate_overlay(image_tensor, anomaly_map.cpu().squeeze(0), overlay_path)

    if args.gt_mask is not None:
        if args.gt_mask.exists():
            import torch
            from PIL import Image
            gt_mask = torch.from_numpy(
                __import__("numpy").array(Image.open(args.gt_mask).convert("L"))
            ).bool()
            if gt_mask.ndim == 2:
                gt_mask = gt_mask.unsqueeze(0)
            gt_vis_path = output_dir / "gt_visualization.png"
            print(f"Generating GT visualization: {gt_vis_path}")
            generate_gt_visualization(
                image_tensor, anomaly_map.cpu().squeeze(0), gt_mask, gt_vis_path
            )
        else:
            print(f"WARNING: GT mask not found: {args.gt_mask}", file=sys.stderr)

    result = {
        "model": model_name,
        "category": category,
        "image": str(args.image),
        "anomaly_score": score,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "prediction": label,
        "latency_ms": round(latency_ms, 2),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result: {result_path}")

    print(f"\nAll outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
