"""Configuration dataclasses for the benchmark pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DatasetConfig:
    """Configuration for MVTec AD dataset."""
    root: Path = Path("./datasets/MVTecAD")
    category: str = "bottle"
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 0
    val_split: float = 0.2
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for anomaly detection model."""
    name: Literal["padim", "patchcore"] = "padim"
    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer1", "layer2", "layer3")
    pre_trained: bool = True
    n_features: int | None = None


@dataclass
class ThresholdConfig:
    """Configuration for threshold calibration."""
    strategy: Literal["percentile", "max", "mean_std"] = "percentile"
    percentile: float = 99.0
    k_std: float = 3.0


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    metrics: tuple[str, ...] = ("auroc", "auprc", "f1", "latency")
    latency_runs: int = 100
    latency_warmup: int = 10


@dataclass
class BenchmarkConfig:
    """Top-level benchmark configuration."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    output_dir: Path = Path("./results")
    device: str = "cpu"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "dataset": self.dataset.__dict__,
            "model": self.model.__dict__,
            "threshold": self.threshold.__dict__,
            "evaluation": self.evaluation.__dict__,
            "output_dir": str(self.output_dir),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkConfig":
        """Create from dictionary."""
        return cls(
            dataset=DatasetConfig(**data.get("dataset", {})),
            model=ModelConfig(**data.get("model", {})),
            threshold=ThresholdConfig(**data.get("threshold", {})),
            evaluation=EvalConfig(**data.get("evaluation", {})),
            output_dir=Path(data.get("output_dir", "./results")),
            device=data.get("device", "cpu"),
        )