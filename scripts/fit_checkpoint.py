#!/usr/bin/env python3
"""One-time script to fit a model and save a checkpoint for the demo."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.checkpoint import save_checkpoint, load_thresholds_from_result
from benchmark.config import DatasetConfig, ModelConfig
from benchmark.dataset import MVTecADWrapper
from benchmark.models import create_model_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a model on MVTec AD training data and save a checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, choices=["padim", "patchcore"],
        help="Model to fit",
    )
    parser.add_argument(
        "--category", required=True,
        help="MVTec AD category (e.g., bottle)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device to fit on",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output checkpoint path (e.g., checkpoints/patchcore_bottle.pt)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("./results"),
        help="Directory containing benchmark result JSONs",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("./datasets/MVTecAD"),
        help="Root directory for MVTec AD dataset",
    )
    parser.add_argument(
        "--backbone", type=str, default=None,
        help="Model backbone (default: resnet18 for padim, wide_resnet50_2 for patchcore)",
    )
    parser.add_argument(
        "--layers", nargs="+", default=None,
        help="Model layers (default: layer1 layer2 layer3 for padim, layer2 layer3 for patchcore)",
    )
    parser.add_argument(
        "--coreset-sampling-ratio", type=float, default=0.1,
        help="Coreset sampling ratio for PatchCore",
    )
    parser.add_argument(
        "--num-neighbors", type=int, default=9,
        help="Number of neighbors for PatchCore",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from benchmark.config import resolve_device
    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.model == "patchcore":
        backbone = args.backbone if args.backbone else "wide_resnet50_2"
        layers = tuple(args.layers) if args.layers else ("layer2", "layer3")
    else:
        backbone = args.backbone if args.backbone else "resnet18"
        layers = tuple(args.layers) if args.layers else ("layer1", "layer2", "layer3")

    model_config = ModelConfig(
        name=args.model,
        backbone=backbone,
        layers=layers,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
    )

    result_json = args.results_dir / f"{args.model}_{args.category}_results.json"
    if not result_json.exists():
        print(f"ERROR: Benchmark result not found: {result_json}")
        print("Run the benchmark first to produce calibrated thresholds.")
        return 1

    print(f"Loading thresholds from: {result_json}")
    image_threshold, pixel_threshold = load_thresholds_from_result(result_json)
    print(f"  image_threshold: {image_threshold:.6f}")
    print(f"  pixel_threshold: {pixel_threshold:.6f}")

    print(f"\nLoading dataset: {args.category}")
    ds_config = DatasetConfig(
        root=args.data_root,
        category=args.category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
        val_split=0.2,
        seed=42,
    )
    dataset = MVTecADWrapper(ds_config)
    dataset.prepare_data()

    print(f"Creating model: {args.model}")
    adapter = create_model_adapter(model_config, device)

    print("Fitting model on training data...")
    start = time.time()
    adapter.fit(dataset.train_loader)
    fit_time = time.time() - start
    print(f"  Fit time: {fit_time:.2f}s")

    save_checkpoint(adapter, model_config, args.category, image_threshold, pixel_threshold, args.output)

    print(f"\nVerifying checkpoint load...")
    from benchmark.checkpoint import load_checkpoint
    loaded_adapter, loaded_img_thresh, loaded_pix_thresh = load_checkpoint(
        args.output, device=device,
        expected_category=args.category,
        expected_model=args.model,
    )
    print(f"  Loaded image_threshold: {loaded_img_thresh:.6f}")
    print(f"  Loaded pixel_threshold: {loaded_pix_thresh:.6f}")
    assert loaded_img_thresh == image_threshold, "Image threshold mismatch after load"
    assert loaded_pix_thresh == pixel_threshold, "Pixel threshold mismatch after load"
    print("  Checkpoint verification passed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
