"""Focused tests for demo functionality: preprocessing, checkpoint, visualization."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.checkpoint import (
    load_checkpoint,
    load_thresholds_from_result,
    save_checkpoint,
    CHECKPOINT_CURRENT_VERSION,
)
from benchmark.config import ModelConfig
from benchmark.demo_utils import (
    generate_heatmap,
    generate_overlay,
    generate_gt_visualization,
    preprocess_image,
)
from benchmark.evaluation import resize_anomaly_map
from benchmark.threshold import apply_threshold


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocessingConsistency:
    """Verify demo preprocessing matches the benchmark dataloader output."""

    def test_preprocess_image_shape_dtype_range(self, tmp_path):
        """preprocess_image produces (1,3,H,W) float32 in [0,1]."""
        img = Image.new("RGB", (100, 80), color=(128, 64, 32))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        tensor = preprocess_image(img_path)
        assert tensor.shape == (1, 3, 80, 100)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_benchmark_vs_demo_preprocessing(self, tmp_path):
        """Demo preprocessing produces the same tensor as the benchmark dataloader."""
        from benchmark.config import DatasetConfig
        from benchmark.dataset import MVTecADWrapper

        ds_config = DatasetConfig(
            root="datasets/MVTecAD", category="bottle",
            train_batch_size=1, eval_batch_size=1, num_workers=0,
        )
        dataset = MVTecADWrapper(ds_config)
        dataset.prepare_data()
        dataset.setup("test")

        for batch in dataset.test_loader:
            benchmark_tensor = batch.image
            image_path = Path(batch.image_path[0])
            break

        demo_tensor = preprocess_image(image_path)

        assert demo_tensor.shape == benchmark_tensor.shape, (
            f"Shape mismatch: demo={demo_tensor.shape}, benchmark={benchmark_tensor.shape}"
        )
        assert demo_tensor.dtype == benchmark_tensor.dtype, (
            f"Dtype mismatch: demo={demo_tensor.dtype}, benchmark={benchmark_tensor.dtype}"
        )
        assert torch.allclose(demo_tensor, benchmark_tensor, atol=1e-6), (
            f"Max diff: {(demo_tensor - benchmark_tensor).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


def _make_mock_adapter(model_name="padim"):
    """Create a mock adapter with a minimal state_dict for testing."""
    adapter = MagicMock()
    adapter._fitted = True

    model_config = ModelConfig(name=model_name)
    mock_model = MagicMock()

    state_dict = {
        "idx": torch.tensor([0, 1, 2]),
        "feature_extractor.feature_extractor.conv1.weight": torch.randn(1, 1),
    }
    mock_model.model.state_dict.return_value = state_dict
    mock_model.model.load_state_dict = MagicMock(side_effect=lambda sd: sd)
    adapter.model = mock_model
    adapter.config = model_config

    return adapter, state_dict


class TestCheckpointSaveLoad:
    """Tests for checkpoint serialization roundtrip."""

    def test_save_load_roundtrip_padim(self, tmp_path):
        """Save and load a PaDiM checkpoint, verify state_dict matches."""
        adapter, original_sd = _make_mock_adapter("padim")
        ckpt_path = tmp_path / "padim_bottle.pt"

        save_checkpoint(
            adapter, adapter.config, "bottle",
            image_threshold=32.5, pixel_threshold=19.7,
            path=ckpt_path,
        )

        assert ckpt_path.exists()
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded["version"] == CHECKPOINT_CURRENT_VERSION
        assert loaded["model_name"] == "padim"
        assert loaded["category"] == "bottle"
        assert loaded["image_threshold"] == 32.5
        assert loaded["pixel_threshold"] == 19.7
        assert "model_state_dict" in loaded
        assert "model_config" in loaded

    def test_save_load_roundtrip_patchcore(self, tmp_path):
        """Save and load a PatchCore checkpoint."""
        adapter, _ = _make_mock_adapter("patchcore")
        ckpt_path = tmp_path / "patchcore_bottle.pt"

        save_checkpoint(
            adapter, adapter.config, "bottle",
            image_threshold=42.16, pixel_threshold=26.73,
            path=ckpt_path,
        )

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded["model_name"] == "patchcore"
        assert loaded["image_threshold"] == 42.16

    def test_load_checkpoint_restores_state(self, tmp_path):
        """Loading a checkpoint restores the model state_dict and thresholds."""
        adapter, original_sd = _make_mock_adapter("padim")
        ckpt_path = tmp_path / "test.pt"

        save_checkpoint(
            adapter, adapter.config, "bottle",
            image_threshold=10.0, pixel_threshold=5.0,
            path=ckpt_path,
        )

        with patch("benchmark.checkpoint.create_model_adapter") as mock_factory:
            mock_loaded_adapter = MagicMock()
            mock_loaded_adapter.model.model = MagicMock()
            mock_factory.return_value = mock_loaded_adapter

            loaded_adapter, img_thresh, pix_thresh = load_checkpoint(
                ckpt_path, device="cpu",
            )

            assert img_thresh == 10.0
            assert pix_thresh == 5.0
            assert loaded_adapter._fitted is True
            loaded_adapter.model.eval.assert_called_once()

    def test_prediction_consistency_before_after_load(self, tmp_path):
        """Same state_dict is saved and loaded during checkpoint roundtrip."""
        original_sd = {
            "idx": torch.tensor([0, 1, 2, 3, 4]),
            "feature_extractor.feature_extractor.conv1.weight": torch.randn(64, 3, 7, 7),
        }

        adapter, _ = _make_mock_adapter("padim")
        adapter.model.model.state_dict.return_value = original_sd

        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(
            adapter, adapter.config, "bottle",
            image_threshold=10.0, pixel_threshold=5.0,
            path=ckpt_path,
        )

        with patch("benchmark.checkpoint.create_model_adapter") as mock_factory:
            mock_loaded = MagicMock()
            mock_loaded.model.model = MagicMock()
            mock_factory.return_value = mock_loaded

            loaded_adapter, _, _ = load_checkpoint(ckpt_path, device="cpu")

            assert loaded_adapter.model.model.load_state_dict.call_count == 1
            call_args = loaded_adapter.model.model.load_state_dict.call_args
            loaded_sd = call_args[0][0]
            assert set(loaded_sd.keys()) == set(original_sd.keys())
            for key in original_sd:
                assert torch.equal(loaded_sd[key], original_sd[key])
            assert loaded_adapter._fitted is True


# ---------------------------------------------------------------------------
# Threshold loading tests
# ---------------------------------------------------------------------------


class TestThresholdLoading:
    """Tests for loading thresholds from benchmark result JSONs."""

    def test_load_thresholds_from_result(self, tmp_path):
        """Extract thresholds from a real benchmark result JSON."""
        result = {
            "category": "bottle",
            "model": "patchcore",
            "image_metrics": {
                "image_threshold": 42.16343307495117,
            },
            "pixel_metrics": {
                "pixel_threshold": 26.73492431640625,
            },
        }
        json_path = tmp_path / "patchcore_bottle_results.json"
        with open(json_path, "w") as f:
            json.dump(result, f)

        img_thresh, pix_thresh = load_thresholds_from_result(json_path)
        assert img_thresh == pytest.approx(42.16343307495117)
        assert pix_thresh == pytest.approx(26.73492431640625)

    def test_load_thresholds_missing_file(self, tmp_path):
        """Missing JSON raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_thresholds_from_result(tmp_path / "nonexistent.json")

    def test_load_thresholds_missing_keys(self, tmp_path):
        """JSON missing threshold keys raises KeyError."""
        json_path = tmp_path / "incomplete.json"
        with open(json_path, "w") as f:
            json.dump({"image_metrics": {}, "pixel_metrics": {}}, f)

        with pytest.raises(KeyError):
            load_thresholds_from_result(json_path)


# ---------------------------------------------------------------------------
# Prediction threshold logic tests (reuse existing apply_threshold)
# ---------------------------------------------------------------------------


class TestPredictionThresholdLogic:
    """Verify demo prediction logic matches the benchmark's apply_threshold."""

    def test_score_above_threshold_is_anomalous(self):
        """Score above threshold → ANOMALOUS (prediction=1)."""
        score = torch.tensor([50.0])
        threshold = 42.16
        prediction = apply_threshold(score, threshold)
        assert prediction.item() == 1

    def test_score_below_threshold_is_normal(self):
        """Score below threshold → NORMAL (prediction=0)."""
        score = torch.tensor([30.0])
        threshold = 42.16
        prediction = apply_threshold(score, threshold)
        assert prediction.item() == 0

    def test_score_equal_to_threshold_is_normal(self):
        """Score exactly equal to threshold → NORMAL (strict >)."""
        score = torch.tensor([42.16])
        threshold = 42.16
        prediction = apply_threshold(score, threshold)
        assert prediction.item() == 0

    def test_boundary_values(self):
        """Scores on both sides of threshold classified correctly."""
        scores = torch.tensor([1.0, 42.16, 42.17, 100.0])
        threshold = 42.16
        preds = apply_threshold(scores, threshold)
        assert preds.tolist() == [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# Anomaly map resizing tests
# ---------------------------------------------------------------------------


class TestAnomalyMapResizing:
    """Tests for anomaly map resizing for visualization."""

    def test_resize_preserves_peak_location(self):
        """Peak in center of small map stays in center when resized."""
        small_map = torch.zeros(16, 16)
        small_map[7:9, 7:9] = 10.0

        resized = resize_anomaly_map(small_map, 64, 64)
        assert resized.shape == (64, 64)
        peak_idx = resized.argmax()
        peak_r, peak_c = divmod(peak_idx.item(), 64)
        assert 28 <= peak_r <= 36
        assert 28 <= peak_c <= 36

    def test_resize_3d_batch(self):
        """Batch of anomaly maps resize correctly."""
        batch_map = torch.randn(4, 32, 32)
        resized = resize_anomaly_map(batch_map, 128, 128)
        assert resized.shape == (4, 128, 128)

    def test_resize_same_size_no_change(self):
        """When target equals source size, output matches input."""
        map_2d = torch.randn(64, 64)
        resized = resize_anomaly_map(map_2d, 64, 64)
        assert torch.allclose(resized, map_2d)


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestHeatmapGeneration:
    """Tests for heatmap generation."""

    def test_heatmap_creates_file(self, tmp_path):
        """generate_heatmap produces a PNG file."""
        anomaly_map = torch.randn(64, 64)
        out_path = tmp_path / "heatmap.png"
        generate_heatmap(anomaly_map, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_heatmap_headless(self, tmp_path):
        """Heatmap generation works without display (Agg backend)."""
        anomaly_map = torch.randn(32, 32)
        out_path = tmp_path / "heatmap.png"
        generate_heatmap(anomaly_map, out_path)
        img = Image.open(out_path)
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_heatmap_constant_map(self, tmp_path):
        """Heatmap of constant anomaly map does not crash."""
        anomaly_map = torch.full((32, 32), 5.0)
        out_path = tmp_path / "heatmap_const.png"
        generate_heatmap(anomaly_map, out_path)
        assert out_path.exists()


class TestOverlayGeneration:
    """Tests for overlay generation."""

    def test_overlay_creates_file(self, tmp_path):
        """generate_overlay produces a PNG file."""
        image = torch.rand(1, 3, 64, 64)
        anomaly_map = torch.randn(32, 32)
        out_path = tmp_path / "overlay.png"
        generate_overlay(image, anomaly_map, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_overlay_resizes_map_to_image(self, tmp_path):
        """Overlay resizes the anomaly map to match the original image dimensions."""
        image = torch.rand(1, 3, 200, 300)
        anomaly_map = torch.randn(16, 16)
        out_path = tmp_path / "overlay.png"
        generate_overlay(image, anomaly_map, out_path)
        assert out_path.exists()

    def test_overlay_4d_image_squeezed(self, tmp_path):
        """Overlay handles both (1,C,H,W) and (C,H,W) image tensors."""
        image_4d = torch.rand(1, 3, 32, 32)
        image_3d = image_4d.squeeze(0)
        anomaly_map = torch.randn(16, 16)
        out1 = tmp_path / "overlay_4d.png"
        out2 = tmp_path / "overlay_3d.png"
        generate_overlay(image_4d, anomaly_map, out1)
        generate_overlay(image_3d, anomaly_map, out2)
        assert out1.exists()
        assert out2.exists()


class TestGTVisualization:
    """Tests for ground truth visualization."""

    def test_gt_vis_creates_file(self, tmp_path):
        """generate_gt_visualization produces a PNG file."""
        image = torch.rand(1, 3, 64, 64)
        anomaly_map = torch.randn(32, 32)
        gt_mask = torch.zeros(1, 64, 64, dtype=torch.bool)
        gt_mask[0, 20:40, 20:40] = True
        out_path = tmp_path / "gt_vis.png"
        generate_gt_visualization(image, anomaly_map, gt_mask, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestCheckpointErrorHandling:
    """Tests for checkpoint loading error conditions."""

    def test_missing_checkpoint_raises(self, tmp_path):
        """Loading a nonexistent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_category_mismatch_raises(self, tmp_path):
        """Category mismatch between checkpoint and --category raises ValueError."""
        adapter, _ = _make_mock_adapter("padim")
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(
            adapter, adapter.config, "bottle",
            image_threshold=10.0, pixel_threshold=5.0,
            path=ckpt_path,
        )

        with patch("benchmark.checkpoint.create_model_adapter"):
            with pytest.raises(ValueError, match="Category mismatch"):
                load_checkpoint(ckpt_path, expected_category="cable")

    def test_model_mismatch_raises(self, tmp_path):
        """Model mismatch between checkpoint and --model raises ValueError."""
        adapter, _ = _make_mock_adapter("padim")
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(
            adapter, adapter.config, "bottle",
            image_threshold=10.0, pixel_threshold=5.0,
            path=ckpt_path,
        )

        with patch("benchmark.checkpoint.create_model_adapter"):
            with pytest.raises(ValueError, match="Model mismatch"):
                load_checkpoint(ckpt_path, expected_model="patchcore")

    def test_malformed_checkpoint_raises(self, tmp_path):
        """Malformed checkpoint (wrong keys) raises ValueError."""
        bad_path = tmp_path / "bad.pt"
        torch.save({"not_a_valid": "checkpoint"}, bad_path)

        with pytest.raises(ValueError, match="missing keys"):
            load_checkpoint(bad_path)

    def test_version_mismatch_raises(self, tmp_path):
        """Checkpoint with wrong version raises ValueError."""
        adapter, _ = _make_mock_adapter("padim")
        ckpt_path = tmp_path / "test.pt"

        ckpt_data = {
            "version": 9999,
            "model_name": "padim",
            "category": "bottle",
            "model_config": {"name": "padim", "backbone": "resnet18",
                             "layers": ["layer1"], "pre_trained": True,
                             "n_features": None, "coreset_sampling_ratio": 0.1,
                             "num_neighbors": 9},
            "model_state_dict": {},
            "image_threshold": 10.0,
            "pixel_threshold": 5.0,
        }
        torch.save(ckpt_data, ckpt_path)

        with pytest.raises(ValueError, match="version mismatch"):
            load_checkpoint(ckpt_path)


class TestImageErrorHandling:
    """Tests for image loading error conditions."""

    def test_missing_image_raises(self, tmp_path):
        """Loading a nonexistent image raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            preprocess_image(tmp_path / "nonexistent.png")

    def test_invalid_image_format(self, tmp_path):
        """Corrupt file raises an error when opening."""
        bad_path = tmp_path / "bad.png"
        bad_path.write_bytes(b"not an image")

        with pytest.raises(Exception):
            preprocess_image(bad_path)


class TestHeadlessExecution:
    """Verify visualization works without GUI display."""

    def test_matplotlib_agg_backend(self, tmp_path):
        """Visualization uses Agg backend (no display needed)."""
        import matplotlib
        import importlib
        import benchmark.demo_utils as du
        importlib.reload(du)
        assert matplotlib.get_backend().lower() == "agg"

    def test_all_visualizations_headless(self, tmp_path):
        """All visualization functions work headlessly."""
        image = torch.rand(1, 3, 32, 32)
        anomaly_map = torch.randn(16, 16)
        gt_mask = torch.zeros(1, 32, 32, dtype=torch.bool)

        generate_heatmap(anomaly_map, tmp_path / "heatmap.png")
        generate_overlay(image, anomaly_map, tmp_path / "overlay.png")
        generate_gt_visualization(image, anomaly_map, gt_mask, tmp_path / "gt.png")

        assert (tmp_path / "heatmap.png").exists()
        assert (tmp_path / "overlay.png").exists()
        assert (tmp_path / "gt.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
