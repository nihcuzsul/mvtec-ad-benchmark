"""Image-level and pixel-level evaluation metrics."""

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
    f1_max: float = 0.0
    f1_max_threshold: float = 0.0

    def to_dict(self) -> dict:
        return {
            "image_auroc": self.auroc,
            "image_auprc": self.auprc,
            "image_f1": self.f1,
            "image_threshold": self.threshold,
            "latency_ms": self.latency_ms,
            "image_f1_max": self.f1_max,
            "image_f1_max_threshold": self.f1_max_threshold,
        }


@dataclass
class PixelMetrics:
    """Container for pixel-level metrics."""
    auroc: float
    auprc: float
    f1: float
    threshold: float
    f1_max: float = 0.0
    f1_max_threshold: float = 0.0
    aupro: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pixel_auroc": self.auroc,
            "pixel_auprc": self.auprc,
            "pixel_f1": self.f1,
            "pixel_threshold": self.threshold,
            "pixel_f1_max": self.f1_max,
            "pixel_f1_max_threshold": self.f1_max_threshold,
            "pixel_aupro": self.aupro,
        }


def resize_anomaly_map(
    anomaly_map: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Resize an anomaly map to match target spatial dimensions.

    Args:
        anomaly_map: (H, W) or (1, H, W) or (B, H, W) tensor
        target_h: target height
        target_w: target width

    Returns:
        Resized tensor with same number of dimensions
    """
    if anomaly_map.ndim == 2:
        map_tensor = anomaly_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        resized = torch.nn.functional.interpolate(
            map_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return resized.squeeze(0).squeeze(0)  # (target_h, target_w)
    elif anomaly_map.ndim == 3:
        map_tensor = anomaly_map.unsqueeze(1)  # (B, 1, H, W)
        resized = torch.nn.functional.interpolate(
            map_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return resized.squeeze(1)  # (B, target_h, target_w)
    else:
        raise ValueError(f"Unexpected anomaly_map dimensions: {anomaly_map.ndim}")


def compute_pixel_metrics(
    anomaly_maps: torch.Tensor,
    gt_masks: torch.Tensor,
    threshold: float,
) -> PixelMetrics:
    """
    Compute pixel-level metrics.

    Args:
        anomaly_maps: Predicted anomaly maps (N, H_pred, W_pred)
        gt_masks: Ground truth masks (N, H_gt, W_gt), bool or float
        threshold: Pixel-level decision threshold

    Returns:
        PixelMetrics with AUROC, AUPRC, F1, threshold
    """
    maps_np = anomaly_maps.cpu().numpy()
    masks_np = gt_masks.cpu().numpy().astype(bool)

    # Resize anomaly maps to match GT mask dimensions
    n_images, h_gt, w_gt = masks_np.shape
    resized_maps = []
    for i in range(n_images):
        h_pred, w_pred = maps_np[i].shape
        if h_pred != h_gt or w_pred != w_gt:
            resized = resize_anomaly_map(
                torch.from_numpy(maps_np[i]), h_gt, w_gt
            ).numpy()
        else:
            resized = maps_np[i]
        resized_maps.append(resized)

    resized_maps = np.stack(resized_maps)  # (N, H_gt, W_gt)

    # Flatten to 1D arrays for sklearn
    pixel_scores = resized_maps.ravel()
    pixel_labels = masks_np.ravel().astype(int)

    # AUROC (threshold-independent)
    try:
        auroc = float(roc_auc_score(pixel_labels, pixel_scores))
    except ValueError:
        auroc = float("nan")
        warnings.warn(
            "Pixel AUROC is undefined: only one class present in pixel labels.",
            stacklevel=2,
        )
    if np.isnan(auroc):
        warnings.warn("Pixel AUROC is NaN.", stacklevel=2)

    # AUPRC (threshold-independent)
    try:
        auprc = float(average_precision_score(pixel_labels, pixel_scores))
    except ValueError:
        auprc = 0.0

    # F1 at calibrated pixel threshold
    preds = (pixel_scores > threshold).astype(int)
    tp = int(((preds == 1) & (pixel_labels == 1)).sum())
    fp = int(((preds == 1) & (pixel_labels == 0)).sum())
    fn = int(((preds == 0) & (pixel_labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return PixelMetrics(
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        threshold=threshold,
    )


def compute_aupro(
    anomaly_maps: torch.Tensor,
    gt_masks: torch.Tensor,
    fpr_limit: float = 0.3,
) -> float:
    """Compute AUPRO (Area Under Per-Region Overlap) using Anomalib's implementation.

    Uses connected component analysis on GT masks and computes per-region ROC
    curves, averaged and integrated up to fpr_limit.

    Args:
        anomaly_maps: Predicted anomaly maps (N, H_pred, W_pred)
        gt_masks: Ground truth masks (N, H_gt, W_gt), bool or float
        fpr_limit: FPR limit for AUPRO integration (default: 0.3 per MVTec protocol)

    Returns:
        AUPRO score (float)
    """
    from .evaluation import resize_anomaly_map
    from anomalib.metrics.aupro import _AUPRO

    maps_np = anomaly_maps.cpu().numpy()
    masks_np = gt_masks.cpu().numpy().astype(bool)

    # Resize anomaly maps to match GT mask dimensions
    n_images, h_gt, w_gt = masks_np.shape
    resized_maps = []
    for i in range(n_images):
        h_pred, w_pred = maps_np[i].shape
        if h_pred != h_gt or w_pred != w_gt:
            resized = resize_anomaly_map(
                torch.from_numpy(maps_np[i]), h_gt, w_gt
            ).numpy()
        else:
            resized = maps_np[i]
        resized_maps.append(resized)

    resized_maps = np.stack(resized_maps)

    # Convert to tensors for Anomalib's _AUPRO
    preds_tensor = torch.from_numpy(resized_maps).float()
    target_tensor = torch.from_numpy(masks_np.astype(np.float32))

    metric = _AUPRO(fpr_limit=fpr_limit)
    try:
        aupro_val = metric(preds_tensor, target_tensor)
        return float(aupro_val.item())
    except Exception:
        warnings.warn("AUPRO computation failed, returning 0.0", stacklevel=2)
        return 0.0


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