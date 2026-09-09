"""Tests for threshold calibration and evaluation metrics."""

import torch
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.threshold import (
    calibrate_threshold,
    apply_threshold,
    compute_f1_at_threshold,
    find_best_f1_threshold,
    ThresholdConfig,
)
from benchmark.evaluation import compute_image_metrics, ImageMetrics


class TestThresholdCalibration:
    """Tests for threshold calibration strategies."""

    def test_percentile_threshold(self):
        """Test percentile-based threshold."""
        scores = torch.arange(100, dtype=torch.float32)
        config = ThresholdConfig(strategy="percentile", percentile=90.0)
        threshold = calibrate_threshold(scores, config)
        assert abs(threshold - 89.0) < 1.0  # 90th percentile of 0-99 is ~89

    def test_max_threshold(self):
        """Test max-based threshold."""
        scores = torch.tensor([1.0, 2.0, 5.0, 3.0])
        config = ThresholdConfig(strategy="max")
        threshold = calibrate_threshold(scores, config)
        assert threshold == 5.0

    def test_mean_std_threshold(self):
        """Test mean + k*std threshold."""
        scores = torch.tensor([10.0] * 100, dtype=torch.float32)
        config = ThresholdConfig(strategy="mean_std", k_std=2.0)
        threshold = calibrate_threshold(scores, config)
        assert threshold == 10.0  # std is 0

    def test_empty_scores_raises(self):
        """Test that empty scores raises ValueError."""
        config = ThresholdConfig(strategy="percentile")
        with pytest.raises(ValueError):
            calibrate_threshold(torch.empty(0), config)

    def test_apply_threshold(self):
        """Test threshold application."""
        scores = torch.tensor([0.1, 0.5, 0.9])
        preds = apply_threshold(scores, 0.5)
        assert preds.tolist() == [0, 0, 1]


class TestF1Computation:
    """Tests for F1 score computation."""

    def test_perfect_f1(self):
        """Test F1 with perfect predictions."""
        scores = torch.tensor([0.9, 0.8, 0.1, 0.2])
        labels = torch.tensor([1, 1, 0, 0])
        f1 = compute_f1_at_threshold(scores, labels, 0.5)
        assert f1 == 1.0

    def test_zero_f1(self):
        """Test F1 with all wrong predictions."""
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 0])
        f1 = compute_f1_at_threshold(scores, labels, 0.5)
        assert f1 == 0.0

    def test_partial_f1(self):
        """Test F1 with partial predictions."""
        scores = torch.tensor([0.9, 0.8, 0.7, 0.1])
        labels = torch.tensor([1, 1, 0, 0])
        f1 = compute_f1_at_threshold(scores, labels, 0.5)
        # TP=2, FP=1, FN=0 -> precision=2/3, recall=1.0 -> F1=0.8
        assert abs(f1 - 0.8) < 1e-6


class TestImageMetrics:
    """Tests for image-level metrics computation."""

    def test_auroc_auprc_perfect(self):
        """Test AUROC/AUPRC with perfect separation."""
        scores = torch.tensor([0.9, 0.8, 0.1, 0.2])
        labels = torch.tensor([1, 1, 0, 0])
        metrics = compute_image_metrics(scores, labels, 0.5, 10.0)
        assert metrics.auroc == 1.0
        assert metrics.auprc == 1.0
        assert metrics.f1 == 1.0

    def test_auroc_random(self):
        """Test AUROC with random scores."""
        torch.manual_seed(42)
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,))
        metrics = compute_image_metrics(scores, labels, 0.5, 10.0)
        assert 0.0 <= metrics.auroc <= 1.0
        assert 0.0 <= metrics.auprc <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
        assert metrics.latency_ms == 10.0

    def test_single_class_auroc(self):
        """Test AUROC when only one class present returns NaN."""
        scores = torch.tensor([0.1, 0.2, 0.3])
        labels = torch.tensor([0, 0, 0])
        metrics = compute_image_metrics(scores, labels, 0.5, 10.0)
        assert np.isnan(metrics.auroc)  # Undefined for single class


class TestFindBestF1Threshold:
    """Tests for finding best F1 threshold."""

    def test_finds_reasonable_threshold(self):
        """Test that it finds a reasonable threshold."""
        scores = torch.cat([
            torch.randn(50) * 0.1 + 0.2,  # Normal scores around 0.2
            torch.randn(50) * 0.1 + 0.8,  # Anomalous scores around 0.8
        ])
        labels = torch.cat([torch.zeros(50), torch.ones(50)])

        thresh, f1 = find_best_f1_threshold(scores, labels)
        assert 0.3 < thresh < 0.7  # Should be between the two clusters
        assert f1 > 0.9  # Should achieve high F1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])