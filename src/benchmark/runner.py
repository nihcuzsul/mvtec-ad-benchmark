"""Benchmark runner orchestrating the full pipeline."""

import random
from dataclasses import dataclass
from pathlib import Path
import json
import csv
import time

import numpy as np
import torch

from .config import BenchmarkConfig, resolve_device
from .dataset import MVTecADWrapper
from .models import create_model_adapter
from .threshold import calibrate_threshold, calibrate_pixel_threshold, find_best_f1_threshold, find_best_f1_pixel
from .evaluation import compute_image_metrics, compute_pixel_metrics, compute_aupro, ImageMetrics, PixelMetrics


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
    peak_gpu_memory_mb: float = 0.0

    def to_dict(self) -> dict:
        result = {
            "category": self.category,
            "model": self.model,
            "image_metrics": self.image_metrics.to_dict(),
            "pixel_metrics": self.pixel_metrics.to_dict(),
            "config": self.config,
        }
        if self.peak_gpu_memory_mb > 0:
            result["peak_gpu_memory_mb"] = self.peak_gpu_memory_mb
        return result


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

    # 0b. Resolve device
    device = config.resolve_device()
    print(f"  Device: {device}")

    # 1. Setup dataset
    print("Loading dataset...")
    dataset = MVTecADWrapper(config.dataset)
    dataset.prepare_data()

    # 2. Create model
    print("Creating model...")
    model = create_model_adapter(config.model, device)

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
    latency_result = model.get_latency(
        test_images[:1],
        n_warmup=config.evaluation.latency_warmup,
        n_runs=config.evaluation.latency_runs,
    )
    latency_ms = latency_result["latency_ms"]
    peak_gpu_mb = latency_result["peak_gpu_memory_mb"]
    print(f"  Latency: {latency_ms:.2f} ms/image")
    if peak_gpu_mb > 0:
        print(f"  Peak GPU memory: {peak_gpu_mb:.1f} MB")

    # 7. Compute image-level metrics
    image_metrics = compute_image_metrics(test_scores, test_labels, image_threshold, latency_ms)

    # 7b. Compute image-level F1-max (best achievable F1 on test predictions)
    img_f1_max_thresh, img_f1_max = find_best_f1_threshold(test_scores, test_labels, n_thresholds=500)
    image_metrics.f1_max = img_f1_max
    image_metrics.f1_max_threshold = img_f1_max_thresh

    print(f"\nImage-Level Results:")
    print(f"  AUROC:  {image_metrics.auroc:.4f}")
    print(f"  AUPRC:  {image_metrics.auprc:.4f}")
    print(f"  F1:     {image_metrics.f1:.4f} (threshold={image_metrics.threshold:.6f})")
    print(f"  F1-max: {image_metrics.f1_max:.4f} (threshold={image_metrics.f1_max_threshold:.6f})")
    print(f"  Latency: {image_metrics.latency_ms:.2f} ms/image")

    # 8. Compute pixel-level metrics
    pixel_metrics = compute_pixel_metrics(test_maps, test_masks, pixel_threshold)

    # 8b. Compute pixel-level F1-max (exact, over all unique score boundaries)
    px_f1_max_thresh, px_f1_max = find_best_f1_pixel(test_maps, test_masks)
    pixel_metrics.f1_max = px_f1_max
    pixel_metrics.f1_max_threshold = px_f1_max_thresh

    # 8c. Compute AUPRO (Anomalib protocol, FPR limit = 0.3)
    pixel_metrics.aupro = compute_aupro(test_maps, test_masks, fpr_limit=0.3)

    print(f"\nPixel-Level Results:")
    print(f"  AUROC:  {pixel_metrics.auroc:.4f}")
    print(f"  AUPRC:  {pixel_metrics.auprc:.4f}")
    print(f"  AUPRO:  {pixel_metrics.aupro:.4f} (FPR limit=0.3)")
    print(f"  F1:     {pixel_metrics.f1:.4f} (threshold={pixel_metrics.threshold:.6f})")
    print(f"  F1-max: {pixel_metrics.f1_max:.4f} (threshold={pixel_metrics.f1_max_threshold:.6f})")

    result = BenchmarkResult(
        category=config.dataset.category,
        model=config.model.name,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        config=config.to_dict(),
        peak_gpu_memory_mb=peak_gpu_mb,
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
        "image_f1_max": result.image_metrics.f1_max,
        "image_threshold": result.image_metrics.threshold,
        "image_f1_max_threshold": result.image_metrics.f1_max_threshold,
        "pixel_auroc": result.pixel_metrics.auroc,
        "pixel_auprc": result.pixel_metrics.auprc,
        "pixel_aupro": result.pixel_metrics.aupro,
        "pixel_f1": result.pixel_metrics.f1,
        "pixel_f1_max": result.pixel_metrics.f1_max,
        "pixel_threshold": result.pixel_metrics.threshold,
        "pixel_f1_max_threshold": result.pixel_metrics.f1_max_threshold,
        "latency_ms": result.image_metrics.latency_ms,
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
    }

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Updated CSV: {csv_path}")


def run_all_categories(config: BenchmarkConfig, categories: list[str]) -> list[BenchmarkResult]:
    """Run benchmark for multiple categories with resume support.

    Skips categories that already have a completed result file in output_dir.
    After all categories, appends an unweighted mean row to summary.csv.
    """
    results = []
    skipped = []

    for category in categories:
        config.dataset.category = category

        # Check for existing result file (resume support)
        json_path = config.output_dir / f"{config.model.name}_{category}_results.json"
        if json_path.exists():
            print(f"\nSkipping {category} (already completed: {json_path.name})")
            skipped.append(category)
            # Load existing result for aggregate
            with open(json_path) as f:
                data = json.load(f)
            existing_result = BenchmarkResult(
                category=data["category"],
                model=data["model"],
                image_metrics=ImageMetrics(
                    auroc=data["image_metrics"]["image_auroc"],
                    auprc=data["image_metrics"]["image_auprc"],
                    f1=data["image_metrics"]["image_f1"],
                    threshold=data["image_metrics"]["image_threshold"],
                    latency_ms=data["image_metrics"]["latency_ms"],
                    f1_max=data["image_metrics"].get("image_f1_max", 0.0),
                    f1_max_threshold=data["image_metrics"].get("image_f1_max_threshold", 0.0),
                ),
                pixel_metrics=PixelMetrics(
                    auroc=data["pixel_metrics"]["pixel_auroc"],
                    auprc=data["pixel_metrics"]["pixel_auprc"],
                    f1=data["pixel_metrics"]["pixel_f1"],
                    threshold=data["pixel_metrics"]["pixel_threshold"],
                    f1_max=data["pixel_metrics"].get("pixel_f1_max", 0.0),
                    f1_max_threshold=data["pixel_metrics"].get("pixel_f1_max_threshold", 0.0),
                    aupro=data["pixel_metrics"].get("pixel_aupro", 0.0),
                ),
                config=data.get("config", {}),
                peak_gpu_memory_mb=data.get("peak_gpu_memory_mb", 0.0),
            )
            results.append(existing_result)
            continue

        result = run_benchmark(config)
        save_results(result, config.output_dir)
        results.append(result)

    if skipped:
        print(f"\nSkipped {len(skipped)} already-completed categories: {', '.join(skipped)}")

    # Append aggregate mean row
    if results:
        _append_aggregate_mean(results, config)

    return results


def _append_aggregate_mean(results: list[BenchmarkResult], config: BenchmarkConfig) -> None:
    """Append unweighted mean row to summary.csv."""
    csv_path = config.output_dir / "summary.csv"
    n = len(results)
    if n == 0:
        return

    mean_row = {
        "category": "MEAN",
        "model": results[0].model,
        "image_auroc": np.nanmean([r.image_metrics.auroc for r in results]),
        "image_auprc": np.nanmean([r.image_metrics.auprc for r in results]),
        "image_f1": np.nanmean([r.image_metrics.f1 for r in results]),
        "image_f1_max": np.nanmean([r.image_metrics.f1_max for r in results]),
        "image_threshold": np.nanmean([r.image_metrics.threshold for r in results]),
        "image_f1_max_threshold": np.nanmean([r.image_metrics.f1_max_threshold for r in results]),
        "pixel_auroc": np.nanmean([r.pixel_metrics.auroc for r in results]),
        "pixel_auprc": np.nanmean([r.pixel_metrics.auprc for r in results]),
        "pixel_aupro": np.nanmean([r.pixel_metrics.aupro for r in results]),
        "pixel_f1": np.nanmean([r.pixel_metrics.f1 for r in results]),
        "pixel_f1_max": np.nanmean([r.pixel_metrics.f1_max for r in results]),
        "pixel_threshold": np.nanmean([r.pixel_metrics.threshold for r in results]),
        "pixel_f1_max_threshold": np.nanmean([r.pixel_metrics.f1_max_threshold for r in results]),
        "latency_ms": np.nanmean([r.image_metrics.latency_ms for r in results]),
        "peak_gpu_memory_mb": np.nanmean([r.peak_gpu_memory_mb for r in results]),
    }

    if csv_path.exists():
        # Read existing CSV, remove old MEAN row if present, append new one
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = [row for row in reader if row.get("category") != "MEAN"]
    else:
        # Create new CSV from results
        fieldnames = list(mean_row.keys())
        rows = []
        for r in results:
            rows.append({
                "category": r.category,
                "model": r.model,
                "image_auroc": r.image_metrics.auroc,
                "image_auprc": r.image_metrics.auprc,
                "image_f1": r.image_metrics.f1,
                "image_f1_max": r.image_metrics.f1_max,
                "image_threshold": r.image_metrics.threshold,
                "image_f1_max_threshold": r.image_metrics.f1_max_threshold,
                "pixel_auroc": r.pixel_metrics.auroc,
                "pixel_auprc": r.pixel_metrics.auprc,
                "pixel_aupro": r.pixel_metrics.aupro,
                "pixel_f1": r.pixel_metrics.f1,
                "pixel_f1_max": r.pixel_metrics.f1_max,
                "pixel_threshold": r.pixel_metrics.threshold,
                "pixel_f1_max_threshold": r.pixel_metrics.f1_max_threshold,
                "latency_ms": r.image_metrics.latency_ms,
                "peak_gpu_memory_mb": r.peak_gpu_memory_mb,
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow(mean_row)

    print(f"Appended MEAN row to {csv_path} ({n} categories averaged)")