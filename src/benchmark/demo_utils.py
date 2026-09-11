"""Single-image preprocessing and visualization utilities for the demo."""

from pathlib import Path

import torch
from torchvision.transforms.functional import to_tensor

from .evaluation import resize_anomaly_map


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Load an image and convert to the tensor format expected by the model.

    Produces a float32 tensor in [0, 1] range with shape (1, C, H, W).
    This matches the output format of the MVTecAD dataloader used by the
    benchmark. The model's internal PreProcessor then applies Resize(256,256)
    and ImageNet normalization.

    Args:
        image_path: Path to a PNG or JPG image.

    Returns:
        Float32 tensor of shape (1, 3, H, W) in [0, 1] range.

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If the image cannot be loaded.
    """
    from PIL import Image

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = to_tensor(image)  # (C, H, W), float32, [0, 1]
    return tensor.unsqueeze(0)  # (1, C, H, W)


def generate_heatmap(
    anomaly_map: torch.Tensor,
    output_path: Path,
) -> None:
    """Generate and save an anomaly heatmap using the inferno colormap.

    Normalization to [0, 255] is performed ONLY for visualization.
    The raw anomaly scores are never modified.

    Args:
        anomaly_map: 2D tensor (H, W) of raw anomaly scores.
        output_path: Path to save the heatmap PNG.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores = anomaly_map.cpu().numpy()
    vmin = float(scores.min())
    vmax = float(scores.max())
    if vmax - vmin < 1e-10:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(scores, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=150)
    plt.close(fig)


def generate_overlay(
    image_tensor: torch.Tensor,
    anomaly_map: torch.Tensor,
    output_path: Path,
    alpha: float = 0.4,
) -> None:
    """Generate and save an anomaly overlay on the original image.

    The anomaly map is resized to the original image dimensions before blending.
    Normalization to [0, 1] for the heatmap is performed ONLY for visualization.

    Args:
        image_tensor: Original image tensor (1, C, H, W) or (C, H, W) in [0, 1].
        anomaly_map: 2D anomaly map tensor (H_map, W_map).
        output_path: Path to save the overlay PNG.
        alpha: Blending factor for the heatmap (0.0 = original, 1.0 = heatmap).
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)

    orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    resized_map = resize_anomaly_map(anomaly_map, orig_h, orig_w)
    scores = resized_map.cpu().numpy()

    vmin = float(scores.min())
    vmax = float(scores.max())
    if vmax - vmin < 1e-10:
        vmin, vmax = 0.0, 1.0
    scores_norm = (scores - vmin) / (vmax - vmin)

    import matplotlib.cm as cm
    heatmap_rgba = cm.inferno(scores_norm)
    heatmap_rgb = heatmap_rgba[:, :, :3]

    blended = (1 - alpha) * image_np + alpha * heatmap_rgb
    blended = np.clip(blended, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(blended)
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=150)
    plt.close(fig)


def generate_gt_visualization(
    image_tensor: torch.Tensor,
    anomaly_map: torch.Tensor,
    gt_mask: torch.Tensor,
    output_path: Path,
) -> None:
    """Generate a side-by-side visualization: original | heatmap | GT mask.

    Args:
        image_tensor: Original image tensor (1, C, H, W) or (C, H, W) in [0, 1].
        anomaly_map: 2D anomaly map tensor (H_map, W_map).
        gt_mask: Ground truth mask tensor (H, W) or (1, H, W), bool or float.
        output_path: Path to save the visualization PNG.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze(0)

    orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    resized_map = resize_anomaly_map(anomaly_map, orig_h, orig_w)
    scores = resized_map.cpu().numpy()
    vmin = float(scores.min())
    vmax = float(scores.max())
    if vmax - vmin < 1e-10:
        vmin, vmax = 0.0, 1.0
    scores_norm = (scores - vmin) / (vmax - vmin)

    import matplotlib.cm as cm
    heatmap_rgba = cm.inferno(scores_norm)
    heatmap_rgb = heatmap_rgba[:, :, :3]

    gt_np = gt_mask.cpu().numpy().astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis("off")

    axes[2].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=150)
    plt.close(fig)
