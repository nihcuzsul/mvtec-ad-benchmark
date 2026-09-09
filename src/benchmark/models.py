"""Unified model adapter interface, PaDiM and PatchCore implementations."""

from abc import ABC, abstractmethod
from typing import Protocol

import torch
import torch.nn as nn
from anomalib.models import Padim, Patchcore
from anomalib.models.image.padim.torch_model import PadimModel

from .config import ModelConfig


class ModelAdapter(Protocol):
    """Unified interface for anomaly detection models."""

    def fit(self, train_loader) -> None:
        """Fit the model on normal training data."""
        ...

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomaly map and score.

        Args:
            images: Input images (B, C, H, W)

        Returns:
            Tuple of (anomaly_map, anomaly_score)
            - anomaly_map: (B, H, W) or (B, 1, H, W) pixel-level anomaly map
            - anomaly_score: (B,) image-level anomaly score
        """
        ...

    def get_latency(self, images: torch.Tensor, n_warmup: int = 10, n_runs: int = 100) -> dict:
        """Measure inference latency in ms/image and peak GPU memory.

        Returns:
            Dict with "latency_ms" and "peak_gpu_memory_mb" keys.
        """
        ...

    @property
    def device(self) -> torch.device:
        """Get model device."""
        ...


class PaDiMAdapter:
    """PaDiM adapter using anomalib's implementation."""

    def __init__(self, config: ModelConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = Padim(
            backbone=config.backbone,
            layers=list(config.layers),
            pre_trained=config.pre_trained,
            n_features=config.n_features,
            pre_processor=True,
            post_processor=True,
            evaluator=False,
            visualizer=False,
        ).to(self.device)
        self._fitted = False

    def fit(self, train_loader) -> None:
        """Fit PaDiM on normal training data (extract features and compute stats).

        The entire fitting process runs on CPU because torch.linalg.inv with
        MAGMA can fail on some CUDA configurations (e.g., NVIDIA T1200).
        After fitting, the model is moved to the configured device for inference.
        """
        self.model.train()
        from anomalib.engine import Engine

        # Fit on CPU to avoid CUDA MAGMA errors in covariance inversion
        engine = Engine(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
        )
        engine.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=None,
        )

        self._fitted = True
        self.model.eval()

        # Move to target device for inference
        if self.device.type == "cuda":
            self.model.to(self.device)

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict anomaly map and score."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        images = images.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(images)

        # InferenceBatch is a NamedTuple: (pred_score, pred_label, anomaly_map, pred_mask)
        anomaly_map = output.anomaly_map
        anomaly_score = output.pred_score

        # Ensure correct shapes
        if anomaly_map is not None and anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
            anomaly_map = anomaly_map.squeeze(1)  # (B, 1, H, W) -> (B, H, W)

        if anomaly_score is not None and anomaly_score.ndim > 1:
            anomaly_score = anomaly_score.squeeze()

        return anomaly_map, anomaly_score

    def get_latency(self, images: torch.Tensor, n_warmup: int = 10, n_runs: int = 100) -> dict:
        """Measure inference latency and peak GPU memory.

        Returns:
            Dict with "latency_ms" (mean ms/image) and "peak_gpu_memory_mb" (0 if CPU).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before latency measurement.")

        images = images.to(self.device)
        self.model.eval()

        # Reset peak memory stats
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(images[:1])

        # Timed runs
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        import time
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(images[:1])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time_ms = (end - start) * 1000
        latency_ms = total_time_ms / n_runs

        # Peak GPU memory
        peak_gpu_mb = 0.0
        if self.device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        return {"latency_ms": latency_ms, "peak_gpu_memory_mb": peak_gpu_mb}


class PatchCoreAdapter:
    """PatchCore adapter using anomalib's implementation."""

    def __init__(self, config: ModelConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = Patchcore(
            backbone=config.backbone,
            layers=list(config.layers),
            pre_trained=config.pre_trained,
            coreset_sampling_ratio=config.coreset_sampling_ratio,
            num_neighbors=config.num_neighbors,
            pre_processor=True,
            post_processor=True,
            evaluator=False,
            visualizer=False,
        ).to(self.device)
        self._fitted = False

    def fit(self, train_loader) -> None:
        """Fit PatchCore on normal training data (extract and subsample embeddings).

        After fitting, the model is moved to the configured device for inference.
        """
        self.model.train()
        from anomalib.engine import Engine

        # Use the configured device for fitting
        accelerator = "gpu" if self.device.type == "cuda" else "cpu"
        engine = Engine(
            max_epochs=1,
            accelerator=accelerator,
            devices=1,
        )
        engine.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=None,
        )

        self._fitted = True
        self.model.eval()

        # Move to target device for inference (Lightning teardown may move to CPU)
        self.model.to(self.device)

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict anomaly map and score."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        images = images.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(images)

        # InferenceBatch is a NamedTuple: (pred_score, pred_label, anomaly_map, pred_mask)
        # PatchCore returns InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
        anomaly_map = output.anomaly_map
        anomaly_score = output.pred_score

        # Ensure correct shapes
        if anomaly_map is not None and anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
            anomaly_map = anomaly_map.squeeze(1)  # (B, 1, H, W) -> (B, H, W)

        if anomaly_score is not None and anomaly_score.ndim > 1:
            anomaly_score = anomaly_score.squeeze()

        return anomaly_map, anomaly_score

    def get_latency(self, images: torch.Tensor, n_warmup: int = 10, n_runs: int = 100) -> dict:
        """Measure inference latency and peak GPU memory.

        Returns:
            Dict with "latency_ms" (mean ms/image) and "peak_gpu_memory_mb" (0 if CPU).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before latency measurement.")

        images = images.to(self.device)
        self.model.eval()

        # Reset peak memory stats
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(images[:1])

        # Timed runs
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        import time
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(images[:1])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time_ms = (end - start) * 1000
        latency_ms = total_time_ms / n_runs

        # Peak GPU memory
        peak_gpu_mb = 0.0
        if self.device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        return {"latency_ms": latency_ms, "peak_gpu_memory_mb": peak_gpu_mb}


def create_model_adapter(config: ModelConfig, device: str = "cpu") -> ModelAdapter:
    """Factory function to create model adapter."""
    if config.name == "padim":
        return PaDiMAdapter(config, device)
    elif config.name == "patchcore":
        return PatchCoreAdapter(config, device)
    else:
        raise ValueError(f"Unknown model: {config.name}")