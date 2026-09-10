#!/usr/bin/env python3
"""CLI entry point for the benchmark pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.config import (
    BenchmarkConfig,
    DatasetConfig,
    ModelConfig,
    ThresholdConfig,
    EvalConfig,
)
from benchmark.runner import run_all_categories


MVTEC_AD_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MVTec AD Anomaly Detection Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle"],
        help="MVTec AD categories to benchmark (use 'all' for all 15 categories)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["padim"],
        choices=["padim", "patchcore"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./datasets/MVTecAD"),
        help="Root directory for MVTec AD dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (auto uses CUDA if available)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio from training data",
    )
    parser.add_argument(
        "--threshold-strategy",
        default="percentile",
        choices=["percentile", "max", "mean_std"],
        help="Threshold calibration strategy",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.0,
        help="Percentile for threshold calibration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training/evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=100,
        help="Number of runs for latency measurement",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Model backbone (default: resnet18 for padim, wide_resnet50_2 for patchcore)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Model layers (default: layer1 layer2 layer3 for padim, layer2 layer3 for patchcore)",
    )
    parser.add_argument(
        "--coreset-sampling-ratio",
        type=float,
        default=0.1,
        help="Coreset sampling ratio for PatchCore",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=9,
        help="Number of neighbors for PatchCore",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve 'all' categories
    if args.categories == ["all"]:
        args.categories = MVTEC_AD_CATEGORIES

    # Build configuration
    # Set model-specific defaults
    if args.models[0] == "patchcore":
        backbone = args.backbone if args.backbone else "wide_resnet50_2"
        layers = args.layers if args.layers else ("layer2", "layer3")
    else:
        backbone = args.backbone if args.backbone else "resnet18"
        layers = tuple(args.layers) if args.layers else ("layer1", "layer2", "layer3")

    config = BenchmarkConfig(
        dataset=DatasetConfig(
            root=args.data_root,
            category=args.categories[0],
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            val_split=args.val_split,
            seed=args.seed,
        ),
        model=ModelConfig(
            name=args.models[0],
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=args.coreset_sampling_ratio,
            num_neighbors=args.num_neighbors,
        ),
        threshold=ThresholdConfig(
            strategy=args.threshold_strategy,
            percentile=args.threshold_percentile,
        ),
        evaluation=EvalConfig(latency_runs=args.latency_runs),
        output_dir=args.output_dir,
        device=args.device,
    )

    print("Benchmark Configuration:")
    print(f"  Categories: {args.categories}")
    print(f"  Models: {args.models}")
    print(f"  Data root: {config.dataset.root}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Device: {config.device}")
    print(f"  Val split: {config.dataset.val_split}")
    print(f"  Threshold: {config.threshold.strategy} ({config.threshold.percentile}th percentile)")
    print(f"  Batch size: {config.dataset.train_batch_size}")
    print(f"  Seed: {config.dataset.seed}")

    # Run for each model
    all_results = []
    for model_name in args.models:
        config.model.name = model_name
        results = run_all_categories(config, args.categories)
        all_results.extend(results)

    # Print summary
    print("\n" + "="*125)
    print("BENCHMARK SUMMARY")
    print("="*125)
    print(f"{'Category':<12} {'Model':<8} {'ImgAUROC':>9} {'ImgAUPRC':>9} {'ImgF1':>7} {'ImgF1max':>8} "
          f"{'PixAUROC':>9} {'PixAUPRC':>9} {'PixAUPRO':>9} {'PixF1':>7} {'PixF1max':>8} {'Lat(ms)':>8}")
    print("-"*125)
    for r in all_results:
        print(f"{r.category:<12} {r.model:<8} {r.image_metrics.auroc:>9.4f} {r.image_metrics.auprc:>9.4f} "
              f"{r.image_metrics.f1:>7.4f} {r.image_metrics.f1_max:>8.4f} "
              f"{r.pixel_metrics.auroc:>9.4f} {r.pixel_metrics.auprc:>9.4f} "
              f"{r.pixel_metrics.aupro:>9.4f} {r.pixel_metrics.f1:>7.4f} {r.pixel_metrics.f1_max:>8.4f} "
              f"{r.image_metrics.latency_ms:>8.2f}")
    print("="*125)

    return 0


if __name__ == "__main__":
    sys.exit(main())