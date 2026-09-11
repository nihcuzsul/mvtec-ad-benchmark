"""Tests for ensure_dataset() behavior."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts/ to path so we can import prepare_dataset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from prepare_dataset import ensure_dataset, is_dataset_prepared


class TestIsDatasetPrepared:
    """Tests for is_dataset_prepared()."""

    def test_returns_false_for_nonexistent_dir(self, tmp_path):
        assert is_dataset_prepared(tmp_path / "nonexistent") is False

    def test_returns_false_for_empty_dir(self, tmp_path):
        assert is_dataset_prepared(tmp_path) is False

    def test_returns_false_for_dir_without_expected_structure(self, tmp_path):
        (tmp_path / "bottle").mkdir()
        assert is_dataset_prepared(tmp_path) is False

    def test_returns_true_when_structure_exists(self, tmp_path):
        (tmp_path / "bottle" / "train" / "good").mkdir(parents=True)
        assert is_dataset_prepared(tmp_path) is True

    def test_returns_true_with_multiple_categories(self, tmp_path):
        (tmp_path / "bottle" / "train" / "good").mkdir(parents=True)
        (tmp_path / "cable" / "train" / "good").mkdir(parents=True)
        assert is_dataset_prepared(tmp_path) is True


class TestEnsureDataset:
    """Tests for ensure_dataset()."""

    def test_existing_dataset_returns_immediately(self, tmp_path):
        """If dataset already exists, ensure_dataset must not modify anything."""
        # Create the expected structure
        train_good = tmp_path / "bottle" / "train" / "good"
        train_good.mkdir(parents=True)

        hf_dir = tmp_path / "hf_source"
        hf_dir.mkdir()

        # Call ensure_dataset — should return immediately
        ensure_dataset(hf_dir=hf_dir, output_dir=tmp_path)

        # Verify nothing was written to hf_dir
        assert list(hf_dir.iterdir()) == []

    def test_existing_dataset_not_downloaded(self, tmp_path):
        """ensure_dataset must not download when dataset exists."""
        (tmp_path / "bottle" / "train" / "good").mkdir(parents=True)

        hf_dir = tmp_path / "hf_source"
        hf_dir.mkdir()

        with patch("huggingface_hub.snapshot_download") as mock_dl:
            ensure_dataset(hf_dir=hf_dir, output_dir=tmp_path)
            mock_dl.assert_not_called()

    def test_existing_dataset_not_built(self, tmp_path):
        """ensure_dataset must not rebuild when dataset exists."""
        (tmp_path / "bottle" / "train" / "good").mkdir(parents=True)

        hf_dir = tmp_path / "hf_source"
        hf_dir.mkdir()

        with patch("prepare_dataset.build_symlink_tree") as mock_build:
            ensure_dataset(hf_dir=hf_dir, output_dir=tmp_path)
            mock_build.assert_not_called()

    def test_missing_source_fails_clearly(self, tmp_path):
        """If Voxel51 source is missing, ensure_dataset must fail with error."""
        hf_dir = tmp_path / "nonexistent_hf"
        output_dir = tmp_path / "MVTecAD"

        # Patch snapshot_download to simulate download failure
        with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("network error")):
            with pytest.raises((FileNotFoundError, RuntimeError)):
                ensure_dataset(hf_dir=hf_dir, output_dir=output_dir)

    def test_force_rebuilds_even_when_prepared(self, tmp_path):
        """With force=True, ensure_dataset rebuilds even if structure exists."""
        (tmp_path / "bottle" / "train" / "good").mkdir(parents=True)

        hf_dir = tmp_path / "hf_source"
        hf_dir.mkdir()
        # Create dummy samples.json so ensure_dataset skips the download path
        (hf_dir / "samples.json").write_text("{}")

        with patch("prepare_dataset.load_samples_json") as mock_load:
            mock_load.return_value = []
            with patch("prepare_dataset.validate_source_files"):
                with patch("prepare_dataset.validate_counts"):
                    with patch("prepare_dataset.build_symlink_tree"):
                        with patch("prepare_dataset.verify_output"):
                            ensure_dataset(hf_dir=hf_dir, output_dir=tmp_path, force=True)
                            mock_load.assert_called_once()
