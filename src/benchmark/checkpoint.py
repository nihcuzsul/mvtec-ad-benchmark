"""Checkpoint serialization, loading, and threshold metadata."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ModelConfig
from .models import create_model_adapter


CHECKPOINT_CURRENT_VERSION = 1


def save_checkpoint(
    adapter,
    model_config: ModelConfig,
    category: str,
    image_threshold: float,
    pixel_threshold: float,
    path: Path,
) -> None:
    """Save a fitted model checkpoint with calibrated thresholds.

    Args:
        adapter: Fitted PaDiMAdapter or PatchCoreAdapter.
        model_config: The ModelConfig used to create the adapter.
        category: MVTec AD category this model was trained on.
        image_threshold: Calibrated image-level threshold from benchmark.
        pixel_threshold: Calibrated pixel-level threshold from benchmark.
        path: Output file path for the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "version": CHECKPOINT_CURRENT_VERSION,
        "model_name": model_config.name,
        "category": category,
        "model_config": {
            "name": model_config.name,
            "backbone": model_config.backbone,
            "layers": list(model_config.layers),
            "pre_trained": model_config.pre_trained,
            "n_features": model_config.n_features,
            "coreset_sampling_ratio": model_config.coreset_sampling_ratio,
            "num_neighbors": model_config.num_neighbors,
        },
        "model_state_dict": adapter.model.model.state_dict(),
        "image_threshold": float(image_threshold),
        "pixel_threshold": float(pixel_threshold),
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Path,
    device: str = "cpu",
    expected_category: str | None = None,
    expected_model: str | None = None,
) -> tuple:
    """Load a fitted model checkpoint and return a ready-to-use adapter.

    Args:
        path: Path to the checkpoint file.
        device: Device to load the model onto.
        expected_category: If provided, the checkpoint's category must match.
        expected_model: If provided, the checkpoint's model name must match.

    Returns:
        Tuple of (adapter, image_threshold, pixel_threshold).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the checkpoint is malformed or has unexpected version.
        ValueError: If expected_category or expected_model do not match.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Malformed checkpoint: expected dict, got {type(checkpoint)}")

    required_keys = {"version", "model_name", "category", "model_config",
                     "model_state_dict", "image_threshold", "pixel_threshold"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Malformed checkpoint: missing keys {missing}")

    version = checkpoint.get("version", 0)
    if version != CHECKPOINT_CURRENT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch: expected v{CHECKPOINT_CURRENT_VERSION}, "
            f"got v{version}"
        )

    if expected_category is not None and checkpoint["category"] != expected_category:
        raise ValueError(
            f"Category mismatch: checkpoint was trained on '{checkpoint['category']}', "
            f"but --category specifies '{expected_category}'"
        )

    if expected_model is not None and checkpoint["model_name"] != expected_model:
        raise ValueError(
            f"Model mismatch: checkpoint is for '{checkpoint['model_name']}', "
            f"but --model specifies '{expected_model}'"
        )

    model_config = ModelConfig(**checkpoint["model_config"])
    adapter = create_model_adapter(model_config, device)
    adapter.model.model.load_state_dict(checkpoint["model_state_dict"])
    adapter._fitted = True
    adapter.model.eval()

    image_threshold = float(checkpoint["image_threshold"])
    pixel_threshold = float(checkpoint["pixel_threshold"])

    return adapter, image_threshold, pixel_threshold


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    """Load checkpoint metadata without creating a model adapter.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Dict with keys: model_name, category, model_config, image_threshold, pixel_threshold, version.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format: {checkpoint_path} "
            "(missing 'model_state_dict' key)"
        )

    version = checkpoint.get("version", 0)
    if version != CHECKPOINT_CURRENT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch: expected v{CHECKPOINT_CURRENT_VERSION}, "
            f"got v{version}"
        )

    return {
        "model_name": checkpoint["model_name"],
        "category": checkpoint["category"],
        "model_config": checkpoint["model_config"],
        "image_threshold": float(checkpoint["image_threshold"]),
        "pixel_threshold": float(checkpoint["pixel_threshold"]),
        "version": version,
    }


def load_thresholds_from_result(result_json_path: Path) -> tuple[float, float]:
    """Load calibrated thresholds from a benchmark result JSON file.

    Args:
        result_json_path: Path to the result JSON file.

    Returns:
        Tuple of (image_threshold, pixel_threshold).

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        KeyError: If required threshold keys are missing.
    """
    result_json_path = Path(result_json_path)
    if not result_json_path.exists():
        raise FileNotFoundError(f"Result JSON not found: {result_json_path}")

    with open(result_json_path) as f:
        data = json.load(f)

    image_threshold = float(data["image_metrics"]["image_threshold"])
    pixel_threshold = float(data["pixel_metrics"]["pixel_threshold"])

    return image_threshold, pixel_threshold
