"""Smoke test for benchmark package imports and basic wiring."""

import sys

def test_imports():
    """Test that all benchmark modules can be imported."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from benchmark import config, dataset, models, threshold, evaluation, runner
    assert config is not None
    assert dataset is not None
    assert models is not None
    assert threshold is not None
    assert evaluation is not None
    assert runner is not None


def test_config_creation():
    """Test config dataclass creation and serialization."""
    from benchmark.config import BenchmarkConfig, DatasetConfig, ModelConfig

    config = BenchmarkConfig()
    assert config.dataset.category == "bottle"
    assert config.model.name == "padim"

    # Test serialization
    d = config.to_dict()
    assert "dataset" in d
    assert "model" in d

    # Test deserialization
    config2 = BenchmarkConfig.from_dict(d)
    assert config2.dataset.category == config.dataset.category


def test_model_factory():
    """Test model adapter factory."""
    from benchmark.models import create_model_adapter, ModelConfig

    config = ModelConfig(name="padim")
    adapter = create_model_adapter(config, "cpu")
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")
    assert hasattr(adapter, "get_latency")


def test_threshold_config():
    """Test threshold config."""
    from benchmark.threshold import ThresholdConfig

    cfg = ThresholdConfig(strategy="percentile", percentile=95.0)
    assert cfg.strategy == "percentile"
    assert cfg.percentile == 95.0


if __name__ == "__main__":
    test_imports()
    test_config_creation()
    test_model_factory()
    test_threshold_config()
    print("All smoke tests passed!")