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


def find_best_f1_pixel(
    anomaly_maps: torch.Tensor,
    gt_masks: torch.Tensor,
) -> tuple[float, float]:
    """
    Find exact threshold that maximizes pixel-level F1 on test predictions.

    Uses sort+cumsum to evaluate F1 at every unique score boundary — O(N log N)
    time, O(N) memory. No approximation.

    This does NOT use test labels for calibration — it reports the best F1
    achievable on the test set given the predictions, for analysis only.

    Args:
        anomaly_maps: Predicted anomaly maps (N, H_pred, W_pred)
        gt_masks: Ground truth masks (N, H_gt, W_gt), bool or float

    Returns:
        (best_threshold, best_f1)
    """
    from .evaluation import resize_anomaly_map

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
    pixel_scores = resized_maps.ravel()
    pixel_labels = masks_np.ravel().astype(int)

    total_pos = int(pixel_labels.sum())
    if total_pos == 0 or pixel_scores.size == 0:
        return float(pixel_scores.min()) if pixel_scores.size > 0 else 0.0, 0.0

    # Sort descending by score
    order = np.argsort(pixel_scores)[::-1]
    sorted_labels = pixel_labels[order]
    sorted_scores = pixel_scores[order]

    # Cumulative TP at each position: number of positives among top-k scores
    cum_tp = np.cumsum(sorted_labels)

    # At position i (0-indexed), threshold = sorted_scores[i]:
    #   predicted positive = scores > sorted_scores[i], i.e. positions 0..i
    #   tp = cum_tp[i]
    #   fp = (i+1) - tp
    #   fn = total_pos - tp
    positions = np.arange(1, len(sorted_scores) + 1)
    tp = cum_tp
    fp = positions - tp
    fn = total_pos - tp

    denom = 2 * tp + fp + fn
    # Avoid division by zero — F1=0 when tp=0
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(denom > 0, 2.0 * tp / denom, 0.0)

    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    best_thresh = float(sorted_scores[best_idx])

    return best_thresh, best_f1