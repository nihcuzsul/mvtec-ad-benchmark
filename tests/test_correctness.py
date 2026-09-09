"""Tests for dataset validation filtering and evaluation correctness."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.evaluation import compute_image_metrics
from benchmark.threshold import apply_threshold, calibrate_threshold, ThresholdConfig


class TestThresholdDirection:
    """Verify score > threshold means anomalous."""

    def test_high_score_above_threshold(self):
        """A high score should be flagged as anomalous."""
        scores = torch.tensor([100.0, 200.0, 300.0])
        preds = apply_threshold(scores, 150.0)
        assert preds.tolist() == [0, 1, 1]

    def test_low_score_below_threshold(self):
        """A low score should be classified as normal."""
        scores = torch.tensor([10.0, 20.0, 30.0])
        preds = apply_threshold(scores, 50.0)
        assert preds.tolist() == [0, 0, 0]

    def test_equal_score_not_anomalous(self):
        """Score exactly equal to threshold is NOT anomalous (> not >=)."""
        scores = torch.tensor([50.0])
        preds = apply_threshold(scores, 50.0)
        assert preds.tolist() == [0]

    def test_threshold_on_boundary(self):
        """Scores on both sides of threshold are classified correctly."""
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = apply_threshold(scores, 3.0)
        assert preds.tolist() == [0, 0, 0, 1, 1]


class TestCalibrationExcludesAnomalous:
    """Verify threshold calibration uses only normal validation scores."""

    def test_percentile_excludes_high_scores(self):
        """High anomalous scores in validation should not inflate the threshold."""
        # Simulate: 80 normal scores around 10, 20 anomalous scores around 100
        normal_scores = torch.full((80,), 10.0)
        anomalous_scores = torch.full((20,), 100.0)
        mixed_scores = torch.cat([normal_scores, anomalous_scores])

        # Calibrate on mixed (bug scenario)
        config = ThresholdConfig(strategy="percentile", percentile=99.0)
        threshold_mixed = calibrate_threshold(mixed_scores, config)

        # Calibrate on normal-only (correct)
        threshold_normal = calibrate_threshold(normal_scores, config)

        # Normal-only threshold should be much lower
        assert threshold_normal < threshold_mixed
        # Normal-only threshold should be around 10 (the normal score value)
        assert threshold_normal == pytest.approx(10.0, abs=0.1)

    def test_empty_normal_scores_raises(self):
        """If no normal scores are provided, calibration should fail."""
        config = ThresholdConfig(strategy="percentile", percentile=99.0)
        with pytest.raises(ValueError):
            calibrate_threshold(torch.empty(0), config)

    def test_calibration_with_single_normal_score(self):
        """Calibration should work with a single normal score."""
        scores = torch.tensor([42.0])
        config = ThresholdConfig(strategy="percentile", percentile=99.0)
        threshold = calibrate_threshold(scores, config)
        assert threshold == 42.0

    def test_max_strategy_uses_only_normal(self):
        """Max strategy should use max of normal scores, not mixed."""
        normal_scores = torch.tensor([10.0, 20.0, 30.0])
        anomalous_scores = torch.tensor([100.0, 200.0])
        mixed_scores = torch.cat([normal_scores, anomalous_scores])

        config = ThresholdConfig(strategy="max")
        threshold_normal = calibrate_threshold(normal_scores, config)
        threshold_mixed = calibrate_threshold(mixed_scores, config)

        assert threshold_normal == 30.0
        assert threshold_mixed == 200.0
        assert threshold_normal < threshold_mixed


class TestAUROCUndefined:
    """Verify AUROC returns NaN when only one class is present."""

    def test_auroc_single_class_returns_nan(self):
        """AUROC should be NaN when labels have only one class."""
        scores = torch.tensor([0.1, 0.2, 0.3])
        labels = torch.tensor([0, 0, 0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = compute_image_metrics(scores, labels, 0.5, 10.0)
            assert np.isnan(metrics.auroc)
            warning_messages = [str(warning.message) for warning in w]
            assert any("AUROC" in msg for msg in warning_messages), (
                f"Expected an AUROC warning, got: {warning_messages}"
            )

    def test_auroc_two_classes_is_valid(self):
        """AUROC should be a valid number when both classes are present."""
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        labels = torch.tensor([0, 0, 1, 1])
        metrics = compute_image_metrics(scores, labels, 0.5, 10.0)
        assert not np.isnan(metrics.auroc)
        assert 0.0 <= metrics.auroc <= 1.0

    def test_auprc_single_class_returns_zero(self):
        """AUPRC should be 0.0 when no positive class is present."""
        scores = torch.tensor([0.1, 0.2, 0.3])
        labels = torch.tensor([0, 0, 0])
        metrics = compute_image_metrics(scores, labels, 0.5, 10.0)
        assert metrics.auprc == 0.0


class TestF1Consistency:
    """Verify F1 computation is consistent with threshold and labels."""

    def test_f1_perfect_separation(self):
        """Perfect score separation should yield F1=1.0."""
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        labels = torch.tensor([0, 0, 1, 1])
        metrics = compute_image_metrics(scores, labels, 0.5, 10.0)
        assert metrics.f1 == pytest.approx(1.0)

    def test_f1_all_wrong(self):
        """All scores on wrong side should yield F1=0.0."""
        scores = torch.tensor([0.8, 0.9])  # high scores
        labels = torch.tensor([0, 0])  # but all normal
        metrics = compute_image_metrics(scores, labels, 0.5, 10.0)
        assert metrics.f1 == 0.0

    def test_f1_matches_manual(self):
        """F1 should match manual computation."""
        scores = torch.tensor([0.3, 0.6, 0.7, 0.9])
        labels = torch.tensor([0, 0, 1, 1])
        threshold = 0.65

        metrics = compute_image_metrics(scores, labels, threshold, 10.0)

        # Manual: preds = [0, 0, 1, 1], TP=2, FP=0, FN=0
        assert metrics.f1 == pytest.approx(1.0)
        assert metrics.threshold == threshold


class TestValScoreFiltering:
    """Integration test: verify get_normal_val_scores excludes anomalous samples."""

    def test_only_normal_scores_returned(self):
        """Val scores must only come from normal validation images."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

        from benchmark.config import DatasetConfig, ModelConfig
        from benchmark.dataset import MVTecADWrapper
        from benchmark.models import create_model_adapter

        ds_config = DatasetConfig(root="datasets/MVTecAD", category="bottle")
        dataset = MVTecADWrapper(ds_config)
        dataset.prepare_data()
        dataset.setup("fit")

        model_config = ModelConfig(name="padim")
        model = create_model_adapter(model_config, "cpu")
        model.fit(dataset.train_loader)

        val_scores, _ = dataset.get_normal_val_scores(model)

        # The SYNTHETIC val split has ~21 normal + ~20 anomalous images
        # After filtering, we should have only normal images
        # Normal images produce low anomaly scores (typically < 60)
        # Anomalous images produce high scores (typically > 50)
        assert val_scores.numel() > 0, "Should have some normal val scores"
        assert val_scores.numel() <= 25, (
            f"Expected at most ~21 normal val scores, got {val_scores.numel()}. "
            "Filtering may not be working."
        )
        # All returned scores should be from normal images (low scores)
        assert val_scores.max().item() < 100.0, (
            f"Max val score {val_scores.max().item():.1f} is suspiciously high. "
            "Anomalous images may not be filtered out."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])