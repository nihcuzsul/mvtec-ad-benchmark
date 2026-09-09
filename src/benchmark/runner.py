"""Benchmark runner orchestrating the full pipeline."""

import random
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import csv
import time

import numpy as np
import torch

from .config import BenchmarkConfig
from .dataset import MVTecADWrapper
from .models import create_model_adapter
from .threshold import calibrate_threshold, calibrate_pixel_threshold
from .evaluation import compute_image_metrics, compute_pixel_metrics, ImageMetrics, PixelMetrics


def seed_everything(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class BenchmarkResult:
    """Result of a single category+model benchmark run."""
    category: str
    model: str
    image_metrics: ImageMetrics
    pixel_metrics: PixelMetrics
    config: dict

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "model": self.model,
            "image_metrics": self.image_metrics.to_dict(),
            "pixel_metrics": self.pixel_metrics.to_dict(),
            "config": self.config,
        }


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run complete benchmark for one category and model.

    Steps:
    1. Load dataset
    2. Create model adapter
    3. Fit model on training data
    4. Calibrate thresholds on validation data (normal only)
    5. Evaluate on test data (image-level and pixel-level)
    6. Measure latency
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {config.dataset.category} | Model: {config.model.name}")
    print(f"{'='*60}")

    # 0. Seed everything for reproducibility
    seed_everything(config.dataset.seed)

    # 1. Setup dataset
    print("Loading dataset...")
    dataset = MVTecADWrapper(config.dataset)
    dataset.prepare_data()

    # 2. Create model
    print("Creating model...")
    model = create_model_adapter(config.model, config.device)

    # 3. Fit model on training data
    print("Fitting model on training data...")
    start_fit = time.time()
    model.fit(dataset.train_loader)
    fit_time = time.time() - start_fit
    print(f"  Fit time: {fit_time:.2f}s")

    # 4. Calibrate thresholds on validation data (normal only)
    print("Calibrating thresholds on validation data...")
    val_scores, val_anomaly_maps = dataset.get_normal_val_scores(model)
    image_threshold = calibrate_threshold(val_scores, config.threshold)
    pixel_threshold = calibrate_pixel_threshold(val_anomaly_maps, config.threshold)
    print(f"  Image threshold ({config.threshold.strategy}): {image_threshold:.6f}")
    print(f"  Pixel threshold ({config.threshold.strategy}): {pixel_threshold:.6f}")
    print(f"  Val score stats: mean={val_scores.mean():.4f}, std={val_scores.std():.4f}, "
          f"min={val_scores.min():.4f}, max={val_scores.max():.4f}")

    # 5. Evaluate on test data
    print("Evaluating on test data...")
    test_images, test_labels, test_masks, test_paths = dataset.get_test_data()
    print(f"  Test samples: {len(test_images)} (normal={(test_labels==0).sum()}, anomalous={(test_labels==1).sum()})")
    print(f"  Test masks shape: {test_masks.shape}")

    test_scores = []
    test_maps = []
    model.model.eval()
    with torch.no_grad():
        for i in range(0, len(test_images), config.dataset.eval_batch_size):
            batch = test_images[i:i+config.dataset.eval_batch_size].to(model.device)
            anomaly_map, anomaly_score = model.predict(batch)
            test_scores.append(anomaly_score.cpu())
            test_maps.append(anomaly_map.cpu())

    test_scores = torch.cat(test_scores, dim=0)
    test_maps = torch.cat(test_maps, dim=0) if test_maps[0].numel() > 0 else torch.empty(0)

    # 6. Measure latency
    print("Measuring latency...")
    latency = model.get_latency(
        test_images[:1],
        n_warmup=config.evaluation.latency_warmup,
        n_runs=config.evaluation.latency_runs,
    )
    print(f"  Latency: {latency:.2f} ms/image")

    # 7. Compute image-level metrics
    image_metrics = compute_image_metrics(test_scores, test_labels, image_threshold, latency)
    print(f"\nImage-Level Results:")
    print(f"  AUROC:  {image_metrics.auroc:.4f}")
    print(f"  AUPRC:  {image_metrics.auprc:.4f}")
    print(f"  F1:     {image_metrics.f1:.4f} (threshold={image_metrics.threshold:.6f})")
    print(f"  Latency: {image_metrics.latency_ms:.2f} ms/image")

    # 8. Compute pixel-level metrics
    pixel_metrics = compute_pixel_metrics(test_maps, test_masks, pixel_threshold)
    print(f"\nPixel-Level Results:")
    print(f"  AUROC:  {pixel_metrics.auroc:.4f}")
    print(f"  AUPRC:  {pixel_metrics.auprc:.4f}")
    print(f"  F1:     {pixel_metrics.f1:.4f} (threshold={pixel_metrics.threshold:.6f})")

    result = BenchmarkResult(
        category=config.dataset.category,
        model=config.model.name,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        config=config.to_dict(),
    )

    return result


def save_results(result: BenchmarkResult, output_dir: Path) -> None:
    """Save results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON (detailed)
    json_path = output_dir / f"{result.model}_{result.category}_results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    # CSV (summary)
    csv_path = output_dir / "summary.csv"
    row = {
        "category": result.category,
        "model": result.model,
        "image_auroc": result.image_metrics.auroc,
        "image_auprc": result.image_metrics.auprc,
        "image_f1": result.image_metrics.f1,
        "image_threshold": result.image_metrics.threshold,
        "pixel_auroc": result.pixel_metrics.auroc,
        "pixel_auprc": result.pixel_metrics.auprc,
        "pixel_f1": result.pixel_metrics.f1,
        "pixel_threshold": result.pixel_metrics.threshold,
        "latency_ms": result.image_metrics.latency_ms,
    }

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Updated CSV: {csv_path}")


def run_all_categories(config: BenchmarkConfig, categories: list[str]) -> list[BenchmarkResult]:
    """Run benchmark for multiple categories."""
    results = []
    for category in categories:
        config.dataset.category = category
        result = run_benchmark(config)
        save_results(result, config.output_dir)
        results.append(result)
    return results