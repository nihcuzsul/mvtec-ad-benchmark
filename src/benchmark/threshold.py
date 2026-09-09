"""Threshold calibration using validation data only."""

import torch
import numpy as np

from .config import ThresholdConfig


def calibrate_threshold(
    val_scores: torch.Tensor,
    config: ThresholdConfig,
) -> float:
    """
    Calibrate decision threshold using ONLY normal validation data.

    Args:
        val_scores: Anomaly scores from normal validation images (N,)
        config: Threshold configuration

    Returns:
        Threshold value for binarization
    """
    scores = val_scores.cpu().numpy()
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        raise ValueError("No valid validation scores provided")

    if config.strategy == "percentile":
        threshold = float(np.percentile(scores, config.percentile))
    elif config.strategy == "max":
        threshold = float(np.max(scores))
    elif config.strategy == "mean_std":
        threshold = float(np.mean(scores) + config.k_std * np.std(scores))
    else:
        raise ValueError(f"Unknown threshold strategy: {config.strategy}")

    return threshold


def calibrate_pixel_threshold(
    val_anomaly_maps: torch.Tensor,
    config: ThresholdConfig,
) -> float:
    """
    Calibrate pixel-level threshold using ONLY normal validation anomaly maps.

    Args:
        val_anomaly_maps: Anomaly maps from normal validation images (N, H, W)
        config: Threshold configuration

    Returns:
        Pixel-level threshold value
    """
    maps = val_anomaly_maps.cpu().numpy()
    pixel_values = maps[np.isfinite(maps)]

    if len(pixel_values) == 0:
        raise ValueError("No valid validation pixel values provided")

    if config.strategy == "percentile":
        threshold = float(np.percentile(pixel_values, config.percentile))
    elif config.strategy == "max":
        threshold = float(np.max(pixel_values))
    elif config.strategy == "mean_std":
        threshold = float(np.mean(pixel_values) + config.k_std * np.std(pixel_values))
    else:
        raise ValueError(f"Unknown threshold strategy: {config.strategy}")

    return threshold


def apply_threshold(
    scores: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Apply threshold to get binary predictions."""
    return (scores > threshold).long()


def compute_f1_at_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> float:
    """Compute F1 score at a given threshold."""
    preds = apply_threshold(scores, threshold)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def find_best_f1_threshold(
    val_scores: torch.Tensor,
    val_labels: torch.Tensor,
    n_thresholds: int = 1000,
) -> tuple[float, float]:
    """
    Find threshold that maximizes F1 on validation data.
    NOTE: This uses validation labels which may not be available in practice.
    Provided for reference/analysis only.
    """
    scores = val_scores.cpu().numpy()
    labels = val_labels.cpu().numpy()

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_f1 = 0.0
    best_thresh = thresholds[0]

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return float(best_thresh), best_f1