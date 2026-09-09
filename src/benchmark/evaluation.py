"""Image-level evaluation metrics."""

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .threshold import apply_threshold, compute_f1_at_threshold


@dataclass
class ImageMetrics:
    """Container for image-level metrics."""
    auroc: float
    auprc: float
    f1: float
    threshold: float
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "image_auroc": self.auroc,
            "image_auprc": self.auprc,
            "image_f1": self.f1,
            "threshold": self.threshold,
            "latency_ms": self.latency_ms,
        }


def compute_image_metrics(
    anomaly_scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    latency_ms: float,
) -> ImageMetrics:
    """
    Compute image-level metrics.

    Args:
        anomaly_scores: Predicted anomaly scores (N,)
        labels: Ground truth labels (N,) - 0 for normal, 1 for anomalous
        threshold: Decision threshold for F1
        latency_ms: Inference latency per image in ms

    Returns:
        ImageMetrics with AUROC, AUPRC, F1, threshold, latency
    """
    scores = anomaly_scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # AUROC (threshold-independent)
    try:
        auroc = float(roc_auc_score(labels_np, scores))
    except ValueError:
        auroc = float("nan")
        warnings.warn(
            "AUROC is undefined: only one class present in labels. Returning NaN.",
            stacklevel=2,
        )
    if np.isnan(auroc):
        warnings.warn("AUROC is NaN.", stacklevel=2)

    # AUPRC (threshold-independent)
    try:
        auprc = float(average_precision_score(labels_np, scores))
    except ValueError:
        auprc = 0.0

    # F1 at calibrated threshold
    f1 = compute_f1_at_threshold(anomaly_scores, labels, threshold)

    return ImageMetrics(
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        threshold=threshold,
        latency_ms=latency_ms,
    )


def compute_metrics_at_thresholds(
    anomaly_scores: torch.Tensor,
    labels: torch.Tensor,
    thresholds: np.ndarray,
) -> list[dict]:
    """Compute precision, recall, F1 at multiple thresholds for analysis."""
    results = []
    scores = anomaly_scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        tp = ((preds == 1) & (labels_np == 1)).sum()
        fp = ((preds == 1) & (labels_np == 0)).sum()
        fn = ((preds == 0) & (labels_np == 1)).sum()
        tn = ((preds == 0) & (labels_np == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        results.append({
            "threshold": float(thresh),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    return results