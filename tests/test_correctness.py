"""Tests for dataset validation filtering and evaluation correctness."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.evaluation import (
    compute_image_metrics,
    compute_pixel_metrics,
    resize_anomaly_map,
    ImageMetrics,
    PixelMetrics,
)
from benchmark.threshold import (
    apply_threshold,
    calibrate_threshold,
    calibrate_pixel_threshold,
    ThresholdConfig,
)


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

        val_scores, val_maps = dataset.get_normal_val_scores(model)

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
        # Verify anomaly maps are also returned
        assert val_maps.numel() > 0, "Should have val anomaly maps"
        assert val_maps.shape[0] == val_scores.shape[0], (
            f"Val maps count {val_maps.shape[0]} != val scores count {val_scores.shape[0]}"
        )


class TestResizeAnomalyMap:
    """Tests for anomaly map resizing."""

    def test_same_size_no_change(self):
        """When sizes match, map should be unchanged."""
        map_2d = torch.randn(256, 256)
        resized = resize_anomaly_map(map_2d, 256, 256)
        assert resized.shape == (256, 256)
        assert torch.allclose(resized, map_2d)

    def test_2d_resize(self):
        """Test resizing a 2D anomaly map."""
        map_2d = torch.randn(64, 64)
        resized = resize_anomaly_map(map_2d, 128, 128)
        assert resized.shape == (128, 128)

    def test_3d_resize(self):
        """Test resizing a batch of anomaly maps."""
        map_3d = torch.randn(4, 64, 64)
        resized = resize_anomaly_map(map_3d, 128, 128)
        assert resized.shape == (4, 128, 128)

    def test_upscale_interpolation(self):
        """Test that upsampling preserves general structure."""
        # Create a simple map with a bright spot in the center
        map_2d = torch.zeros(10, 10)
        map_2d[4:6, 4:6] = 1.0
        resized = resize_anomaly_map(map_2d, 20, 20)
        assert resized.shape == (20, 20)
        # Center should still be bright
        assert resized[9:11, 9:11].mean() > resized[0:2, 0:2].mean()


class TestPixelThresholdCalibration:
    """Tests for pixel-level threshold calibration."""

    def test_percentile_pixel_threshold(self):
        """Test percentile-based pixel threshold."""
        # Create a map with known pixel values
        maps = torch.full((5, 10, 10), 10.0)  # All pixels = 10
        config = ThresholdConfig(strategy="percentile", percentile=99.0)
        threshold = calibrate_pixel_threshold(maps, config)
        assert threshold == 10.0

    def test_max_pixel_threshold(self):
        """Test max-based pixel threshold."""
        maps = torch.zeros(2, 10, 10)
        maps[0, 5, 5] = 100.0  # One pixel is 100
        config = ThresholdConfig(strategy="max")
        threshold = calibrate_pixel_threshold(maps, config)
        assert threshold == 100.0

    def test_empty_maps_raises(self):
        """Empty maps should raise ValueError."""
        config = ThresholdConfig(strategy="percentile")
        with pytest.raises(ValueError):
            calibrate_pixel_threshold(torch.empty(0, 10, 10), config)


class TestPixelMetrics:
    """Tests for pixel-level metrics computation."""

    def test_perfect_pixel_auroc(self):
        """Perfect anomaly map separation should yield AUROC=1.0."""
        # Anomaly maps: anomalous images have high scores, normal have low
        anomaly_maps = torch.tensor([
            [[0.1, 0.1], [0.1, 0.1]],  # Normal image, low scores
            [[0.9, 0.9], [0.9, 0.9]],  # Anomalous image, high scores
        ])
        gt_masks = torch.tensor([
            [[False, False], [False, False]],  # Normal image, all zeros
            [[True, True], [True, True]],       # Anomalous image, all ones
        ])
        metrics = compute_pixel_metrics(anomaly_maps, gt_masks, 0.5)
        assert metrics.auroc == 1.0
        assert metrics.auprc == 1.0

    def test_pixel_f1_perfect(self):
        """Perfect pixel predictions should yield F1=1.0."""
        anomaly_maps = torch.tensor([
            [[0.1, 0.9], [0.1, 0.9]],
        ])
        gt_masks = torch.tensor([
            [[False, True], [False, True]],
        ])
        metrics = compute_pixel_metrics(anomaly_maps, gt_masks, 0.5)
        assert metrics.f1 == pytest.approx(1.0)

    def test_pixel_f1_zero(self):
        """All wrong pixel predictions should yield F1=0.0."""
        anomaly_maps = torch.tensor([
            [[0.9, 0.9], [0.9, 0.9]],  # All high scores
        ])
        gt_masks = torch.tensor([
            [[False, False], [False, False]],  # But all normal
        ])
        metrics = compute_pixel_metrics(anomaly_maps, gt_masks, 0.5)
        assert metrics.f1 == 0.0

    def test_resize_in_pixel_metrics(self):
        """Pixel metrics should handle different anomaly map/mask sizes."""
        # Anomaly map is 4x4, mask is 8x8
        anomaly_maps = torch.zeros(1, 4, 4)
        anomaly_maps[0, 1:3, 1:3] = 1.0  # Bright center
        gt_masks = torch.zeros(1, 8, 8).bool()
        gt_masks[0, 2:6, 2:6] = True  # Larger mask region
        metrics = compute_pixel_metrics(anomaly_maps, gt_masks, 0.5)
        assert not np.isnan(metrics.auroc)
        assert 0.0 <= metrics.f1 <= 1.0

    def test_pixel_metrics_single_class(self):
        """Pixel metrics with only normal pixels."""
        anomaly_maps = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        gt_masks = torch.tensor([[[False, False], [False, False]]]).bool()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            metrics = compute_pixel_metrics(anomaly_maps, gt_masks, 0.5)
            assert np.isnan(metrics.auroc)  # Only one class

    def test_pixel_metrics_matches_manual(self):
        """Pixel F1 should match manual computation."""
        anomaly_maps = torch.tensor([
            [[0.3, 0.7], [0.2, 0.8]],
        ])
        gt_masks = torch.tensor([
            [[False, True], [False, True]],
        ])
        threshold = 0.5
        metrics = compute_pixel_metrics(anomaly_maps, gt_masks, threshold)

        # Manual: pred=[0,1,0,1], label=[0,1,0,1] -> TP=2, FP=0, FN=0
        assert metrics.f1 == pytest.approx(1.0)
        assert metrics.threshold == threshold


class TestDeviceResolution:
    """Tests for device resolution logic."""

    def test_auto_cpu_fallback(self):
        """Auto should return cpu when cuda unavailable."""
        from benchmark.config import resolve_device
        result = resolve_device("auto")
        assert result in ("cpu", "cuda:0")

    def test_explicit_cpu(self):
        """Explicit cpu should always return cpu."""
        from benchmark.config import resolve_device
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        """Explicit cuda should return cuda:0 when available, cpu otherwise."""
        from benchmark.config import resolve_device
        import torch
        result = resolve_device("cuda")
        if torch.cuda.is_available():
            assert result == "cuda:0"
        else:
            assert result == "cpu"

    def test_auto_uses_cuda_when_available(self):
        """Auto should use cuda when available."""
        from benchmark.config import resolve_device
        import torch
        result = resolve_device("auto")
        if torch.cuda.is_available():
            assert result == "cuda:0"
        else:
            assert result == "cpu"

    def test_model_device_matches_config(self):
        """Model adapter should be on the correct device."""
        from benchmark.config import resolve_device
        from benchmark.models import create_model_adapter, ModelConfig
        device = resolve_device("auto")
        config = ModelConfig(name="padim")
        adapter = create_model_adapter(config, device)
        assert adapter.device == torch.device(device)

    def test_config_resolve_device_updates_config(self):
        """BenchmarkConfig.resolve_device should update config.device."""
        from benchmark.config import BenchmarkConfig
        config = BenchmarkConfig(device="auto")
        resolved = config.resolve_device()
        assert resolved in ("cpu", "cuda:0")
        assert config.device == resolved


class TestGPULatencyMeasurement:
    """Tests for GPU latency measurement."""

    def test_latency_returns_dict(self):
        """get_latency should return dict with latency_ms and peak_gpu_memory_mb."""
        from benchmark.models import create_model_adapter, ModelConfig
        from benchmark.config import DatasetConfig
        from benchmark.dataset import MVTecADWrapper

        # Use real dataset to create proper Batch objects for fitting
        ds_config = DatasetConfig(root="datasets/MVTecAD", category="bottle")
        dataset = MVTecADWrapper(ds_config)
        dataset.prepare_data()
        dataset.setup("fit")

        config = ModelConfig(name="padim")
        adapter = create_model_adapter(config, "cpu")
        adapter.fit(dataset.train_loader)

        # Test latency measurement
        test_images, _, _, _ = dataset.get_test_data()
        result = adapter.get_latency(test_images[:1], n_warmup=1, n_runs=2)
        assert isinstance(result, dict)
        assert "latency_ms" in result
        assert "peak_gpu_memory_mb" in result
        assert result["latency_ms"] > 0
        assert result["peak_gpu_memory_mb"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])