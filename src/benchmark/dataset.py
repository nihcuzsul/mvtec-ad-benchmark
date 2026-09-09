"""MVTec AD dataset wrapper with validation split."""

from pathlib import Path
from typing import Optional

import torch
from anomalib.data import MVTecAD
from anomalib.data.datamodules.image.mvtecad import TestSplitMode, ValSplitMode

from .config import DatasetConfig


class MVTecADWrapper:
    """Wrapper around anomalib's MVTecAD datamodule with explicit validation split."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.datamodule: Optional[MVTecAD] = None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    def prepare_data(self) -> None:
        """Download and prepare the dataset."""
        self.datamodule = MVTecAD(
            root=str(self.config.root),
            category=self.config.category,
            train_batch_size=self.config.train_batch_size,
            eval_batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers,
            test_split_mode=TestSplitMode.FROM_DIR,
            test_split_ratio=0.2,
            val_split_mode=ValSplitMode.SYNTHETIC,
            val_split_ratio=self.config.val_split,
            seed=self.config.seed,
        )
        self.datamodule.prepare_data()

    def setup(self, stage: str = "fit") -> None:
        """Setup dataloaders for the given stage."""
        if self.datamodule is None:
            self.prepare_data()
        self.datamodule.setup(stage)

    @property
    def train_loader(self):
        """Get training dataloader (normal images only)."""
        if self._train_loader is None:
            self.setup("fit")
            self._train_loader = self.datamodule.train_dataloader()
        return self._train_loader

    @property
    def val_loader(self):
        """Get validation dataloader (normal images only, from training split)."""
        if self._val_loader is None:
            self.setup("fit")
            self._val_loader = self.datamodule.val_dataloader()
        return self._val_loader

    @property
    def test_loader(self):
        """Get test dataloader (normal + anomalous images with masks)."""
        if self._test_loader is None:
            self.setup("test")
            self._test_loader = self.datamodule.test_dataloader()
        return self._test_loader

    def get_normal_train_images(self) -> torch.Tensor:
        """Collect all normal training images for threshold calibration."""
        images = []
        for batch in self.train_loader:
            images.append(batch["image"])
        return torch.cat(images, dim=0) if images else torch.empty(0)

    def get_normal_val_scores(self, model) -> tuple[torch.Tensor, torch.Tensor]:
        """Get anomaly scores for normal validation images only.

        Filters out any synthetic anomalous samples present in the
        validation set (e.g. from ValSplitMode.SYNTHETIC) so that
        threshold calibration is performed exclusively on normal data.
        """
        scores = []
        maps = []
        for batch in self.val_loader:
            normal_mask = ~batch.gt_label
            if normal_mask.sum() == 0:
                continue
            images = batch.image[normal_mask].to(model.device)
            anomaly_map, anomaly_score = model.predict(images)
            scores.append(anomaly_score.cpu())
            maps.append(anomaly_map.cpu())
        return (
            torch.cat(scores, dim=0) if scores else torch.empty(0),
            torch.cat(maps, dim=0) if maps else torch.empty(0),
        )

    def get_test_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Get test images, labels, masks, and paths."""
        images = []
        labels = []
        masks = []
        paths = []
        for batch in self.test_loader:
            images.append(batch.image)
            labels.append(batch.gt_label)
            if batch.gt_mask is not None:
                masks.append(batch.gt_mask)
            else:
                masks.append(torch.zeros_like(batch.image[:, :1]))
            paths.extend(batch.image_path)
        return (
            torch.cat(images, dim=0) if images else torch.empty(0),
            torch.cat(labels, dim=0) if labels else torch.empty(0),
            torch.cat(masks, dim=0) if masks else torch.empty(0),
            paths,
        )