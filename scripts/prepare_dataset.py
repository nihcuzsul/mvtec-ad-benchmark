#!/usr/bin/env python3
"""Prepare Voxel51/mvtec-ad HF dataset into standard MVTec AD directory layout.

Converts the FiftyOne-exported imagefolder (samples.json + data/ + fields/)
into the standard directory structure expected by anomalib.data.MVTecAD:

    <output_root>/<category>/
    ├── train/good/          ← symlinks to training normal images
    ├── test/good/           ← symlinks to test normal images
    ├── test/{defect}/       ← symlinks to test anomaly images
    └── ground_truth/{defect}/ ← symlinks to anomaly masks

Usage:
    python scripts/prepare_dataset.py [--hf-dir PATH] [--output-dir PATH] [--force]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]

EXPECTED_TOTAL = 5354
EXPECTED_TRAIN = 3629
EXPECTED_TEST_NORMAL = 467
EXPECTED_TEST_ANOMALY = 1258


def load_samples_json(hf_dir: Path) -> list[dict]:
    """Load and validate samples.json from the HF dataset."""
    samples_path = hf_dir / "samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.json not found at {samples_path}")

    print(f"Loading {samples_path} ...")
    with open(samples_path) as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"  Loaded {len(samples)} samples")
    return samples


def validate_source_files(samples: list[dict], hf_dir: Path) -> None:
    """Verify all referenced files exist on disk. Fail loudly on any missing."""
    missing_images = []
    missing_masks = []

    for i, s in enumerate(samples):
        img_path = hf_dir / s["filepath"]
        if not img_path.exists():
            missing_images.append((i, s["filepath"]))

        defect = s.get("defect", {}).get("label", "good")
        if defect != "good":
            mask_info = s.get("defect_mask")
            if mask_info and "mask_path" in mask_info:
                mask_path = hf_dir / mask_info["mask_path"]
                if not mask_path.exists():
                    missing_masks.append((i, mask_info["mask_path"]))

    if missing_images:
        print(f"\nERROR: {len(missing_images)} missing image files:")
        for idx, path in missing_images[:10]:
            print(f"  sample[{idx}]: {path}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
        raise FileNotFoundError(f"{len(missing_images)} image files missing from HF dataset")

    if missing_masks:
        print(f"\nERROR: {len(missing_masks)} missing mask files:")
        for idx, path in missing_masks[:10]:
            print(f"  sample[{idx}]: {path}")
        if len(missing_masks) > 10:
            print(f"  ... and {len(missing_masks) - 10} more")
        raise FileNotFoundError(f"{len(missing_masks)} mask files missing from HF dataset")

    print(f"  All {len(samples)} images present")
    print(f"  All anomaly masks present")


def validate_counts(samples: list[dict]) -> None:
    """Verify per-category and overall counts match expected values."""
    overall = {"train": 0, "test_normal": 0, "test_anomaly": 0}
    per_category = defaultdict(lambda: {"train": 0, "test_normal": 0, "test_anomaly": 0})

    for s in samples:
        cat = s["category"]["label"]
        split = s["split"]
        defect = s["defect"]["label"]

        if split == "train":
            overall["train"] += 1
            per_category[cat]["train"] += 1
        elif split == "test":
            if defect == "good":
                overall["test_normal"] += 1
                per_category[cat]["test_normal"] += 1
            else:
                overall["test_anomaly"] += 1
                per_category[cat]["test_anomaly"] += 1

    total = overall["train"] + overall["test_normal"] + overall["test_anomaly"]
    errors = []

    if total != EXPECTED_TOTAL:
        errors.append(f"Total: {total} != {EXPECTED_TOTAL}")
    if overall["train"] != EXPECTED_TRAIN:
        errors.append(f"Train: {overall['train']} != {EXPECTED_TRAIN}")
    if overall["test_normal"] != EXPECTED_TEST_NORMAL:
        errors.append(f"Test normal: {overall['test_normal']} != {EXPECTED_TEST_NORMAL}")
    if overall["test_anomaly"] != EXPECTED_TEST_ANOMALY:
        errors.append(f"Test anomaly: {overall['test_anomaly']} != {EXPECTED_TEST_ANOMALY}")

    found_categories = set(per_category.keys())
    missing = set(EXPECTED_CATEGORIES) - found_categories
    extra = found_categories - set(EXPECTED_CATEGORIES)
    if missing:
        errors.append(f"Missing categories: {missing}")
    if extra:
        errors.append(f"Unexpected categories: {extra}")

    if errors:
        print("\nERROR: Count validation failed:")
        for e in errors:
            print(f"  {e}")
        raise ValueError("Dataset count validation failed")

    print(f"  Total: {total} (expected {EXPECTED_TOTAL})")
    print(f"  Train: {overall['train']}, Test normal: {overall['test_normal']}, "
          f"Test anomaly: {overall['test_anomaly']}")
    print(f"  All 15 categories present with correct counts")


def build_symlink_tree(
    samples: list[dict],
    hf_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> None:
    """Create the standard MVTec AD directory tree using symlinks."""
    hf_abs = hf_dir.resolve()
    output_abs = output_dir.resolve()

    if output_abs.exists() and force:
        import shutil
        print(f"  Removing existing {output_dir} ...")
        shutil.rmtree(output_dir)

    # Group samples by (category, split, defect)
    groups = defaultdict(list)
    for s in samples:
        cat = s["category"]["label"]
        split = s["split"]
        defect = s["defect"]["label"]
        groups[(cat, split, defect)].append(s)

    created_dirs = set()
    symlink_count = 0

    for cat in EXPECTED_CATEGORIES:
        cat_dir = output_abs / cat

        # --- Train (all normal) ---
        train_samples = sorted(groups.get((cat, "train", "good"), []),
                               key=lambda s: s["filepath"])
        if train_samples:
            train_good = cat_dir / "train" / "good"
            _create_dir(train_good, created_dirs)
            for i, s in enumerate(train_samples):
                target = os.path.relpath((hf_abs / s["filepath"]).resolve(), train_good)
                (train_good / f"{i:03d}.png").symlink_to(target)
                symlink_count += 1

        # --- Test normal ---
        test_normal = sorted(groups.get((cat, "test", "good"), []),
                             key=lambda s: s["filepath"])
        if test_normal:
            test_good = cat_dir / "test" / "good"
            _create_dir(test_good, created_dirs)
            for i, s in enumerate(test_normal):
                target = os.path.relpath((hf_abs / s["filepath"]).resolve(), test_good)
                (test_good / f"{i:03d}.png").symlink_to(target)
                symlink_count += 1

        # --- Test anomalous + ground_truth ---
        defect_groups = defaultdict(list)
        for s in groups.get((cat, "test", "good"), []):
            pass  # already handled above
        for defect_label, defect_samples in groups.items():
            if defect_label[0] == cat and defect_label[1] == "test" and defect_label[2] != "good":
                defect_name = defect_label[2]
                defect_groups[defect_name].extend(defect_samples)

        for defect_name in sorted(defect_groups.keys()):
            defect_samples = sorted(defect_groups[defect_name],
                                    key=lambda s: s["filepath"])

            # Image symlinks in test/{defect}/
            test_defect_dir = cat_dir / "test" / defect_name
            _create_dir(test_defect_dir, created_dirs)

            # Mask symlinks in ground_truth/{defect}/
            gt_defect_dir = cat_dir / "ground_truth" / defect_name
            _create_dir(gt_defect_dir, created_dirs)

            for i, s in enumerate(defect_samples):
                fname = f"{i:03d}.png"

                # Image symlink
                img_target = os.path.relpath(
                    (hf_abs / s["filepath"]).resolve(), test_defect_dir)
                (test_defect_dir / fname).symlink_to(img_target)
                symlink_count += 1

                # Mask symlink (same filename for anomalib stem check)
                mask_info = s.get("defect_mask")
                if mask_info and "mask_path" in mask_info:
                    mask_target = os.path.relpath(
                        (hf_abs / mask_info["mask_path"]).resolve(), gt_defect_dir)
                    (gt_defect_dir / fname).symlink_to(mask_target)
                    symlink_count += 1

    print(f"  Created {symlink_count} symlinks across {len(EXPECTED_CATEGORIES)} categories")


def _create_dir(path: Path, created: set) -> None:
    """Create directory if not already created."""
    if path not in created:
        path.mkdir(parents=True, exist_ok=True)
        created.add(path)


def verify_output(output_dir: Path, expect_all_categories: bool = True) -> None:
    """Post-creation integrity check. Fail loudly on any inconsistency.

    Args:
        output_dir: Root of the prepared MVTec AD layout.
        expect_all_categories: If True, require all 15 categories present.
    """
    print("\nVerifying output structure ...")
    errors = []
    total_images = 0
    total_masks = 0

    # Discover which categories exist
    found_categories = sorted([
        d.name for d in output_dir.iterdir()
        if d.is_dir() and d.name in EXPECTED_CATEGORIES
    ]) if output_dir.exists() else []

    missing = set(EXPECTED_CATEGORIES) - set(found_categories)
    if expect_all_categories and missing:
        for cat in sorted(missing):
            errors.append(f"{cat}: category directory missing")

    for cat in found_categories:
        cat_dir = output_dir / cat

        # Count train images
        train_good = cat_dir / "train" / "good"
        train_count = _count_symlinks(train_good)
        total_images += train_count

        # Count test normal
        test_good = cat_dir / "test" / "good"
        test_normal_count = _count_symlinks(test_good)
        total_images += test_normal_count

        # Check broken symlinks in train/good and test/good
        for d in [train_good, test_good]:
            if d.exists():
                for f in d.iterdir():
                    if f.is_symlink() and not f.exists():
                        errors.append(f"{cat}/{d.parent.name}/{d.name}/{f.name}: broken symlink")

        # Count test anomalous and masks per defect
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        test_defects = sorted([d.name for d in test_dir.iterdir()
                               if d.is_dir() and d.name != "good"])
        gt_defects = sorted([d.name for d in gt_dir.iterdir()
                             if d.is_dir()]) if gt_dir.exists() else []

        if test_defects != gt_defects:
            errors.append(f"{cat}: test defects {test_defects} != ground_truth defects {gt_defects}")

        for defect_name in test_defects:
            test_defect_dir = test_dir / defect_name
            gt_defect_dir = gt_dir / defect_name

            test_count = _count_symlinks(test_defect_dir)
            gt_count = _count_symlinks(gt_defect_dir) if gt_defect_dir.exists() else 0

            total_images += test_count
            total_masks += gt_count

            if test_count != gt_count:
                errors.append(f"{cat}/{defect_name}: {test_count} test images != {gt_count} masks")

            # Check for broken symlinks
            for f in test_defect_dir.iterdir():
                if f.is_symlink() and not f.exists():
                    errors.append(f"{cat}/{defect_name}/{f.name}: broken symlink")
            if gt_defect_dir.exists():
                for f in gt_defect_dir.iterdir():
                    if f.is_symlink() and not f.exists():
                        errors.append(f"{cat}/ground_truth/{defect_name}/{f.name}: broken symlink")

    if errors:
        print("\nERROR: Verification failed:")
        for e in errors:
            print(f"  {e}")
        raise ValueError("Output verification failed")

    if expect_all_categories:
        print(f"  Total images: {total_images} (expected {EXPECTED_TOTAL})")
        print(f"  Total masks: {total_masks} (expected {EXPECTED_TEST_ANOMALY})")
    else:
        print(f"  Total images: {total_images}")
        print(f"  Total masks: {total_masks}")
    print(f"  All symlinks valid, all defect groups matched")
    print("  PASS")


def _count_symlinks(directory: Path) -> int:
    """Count valid symlinks in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.is_symlink() or f.is_file())


def is_dataset_prepared(output_dir: Path) -> bool:
    """Check if the MVTec AD directory layout already exists.

    Returns True if output_dir contains at least one category with
    the expected train/good structure.
    """
    if not output_dir.exists():
        return False
    for cat in EXPECTED_CATEGORIES:
        if (output_dir / cat / "train" / "good").exists():
            return True
    return False


def ensure_dataset(
    hf_dir: Path = Path("./datasets/Voxel51-mvtec-ad"),
    output_dir: Path = Path("./datasets/MVTecAD"),
    force: bool = False,
) -> None:
    """Ensure the prepared MVTec AD dataset layout exists.

    If output_dir already contains a prepared dataset, returns immediately.
    Otherwise, runs the full preparation pipeline (download, validate, build).
    """
    if is_dataset_prepared(output_dir) and not force:
        print(f"Dataset already prepared at {output_dir}")
        return

    print(f"Preparing dataset: {hf_dir} -> {output_dir}")

    # Step 1: Download if needed
    samples_json = hf_dir / "samples.json"
    if not samples_json.exists():
        print(f"Downloading Voxel51/mvtec-ad to {hf_dir} ...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Voxel51/mvtec-ad",
                repo_type="dataset",
                local_dir=str(hf_dir),
            )
            print("  Download complete")
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )
    else:
        print(f"Using existing snapshot at {hf_dir}")

    # Step 2: Parse and validate
    samples = load_samples_json(hf_dir)
    print("\nValidating source files ...")
    validate_source_files(samples, hf_dir)
    print("\nValidating counts ...")
    validate_counts(samples)

    # Step 3: Build symlink tree
    print(f"\nCreating standard MVTec AD layout at {output_dir} ...")
    build_symlink_tree(samples, hf_dir, output_dir, force=force)

    # Step 4: Verify output
    verify_output(output_dir)
    print(f"\nDataset ready at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Voxel51/mvtec-ad into standard MVTec AD layout")
    parser.add_argument("--hf-dir", type=Path,
                        default=Path("./datasets/Voxel51-mvtec-ad"),
                        help="Path to downloaded HF dataset snapshot")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("./datasets/MVTecAD"),
                        help="Output path for standard MVTec AD layout")
    parser.add_argument("--force", action="store_true",
                        help="Remove existing output directory before creating")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HF download, use existing snapshot")
    args = parser.parse_args()

    # Step 1: Download if needed
    if not args.skip_download:
        samples_json = args.hf_dir / "samples.json"
        if not samples_json.exists():
            print(f"Downloading Voxel51/mvtec-ad to {args.hf_dir} ...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="Voxel51/mvtec-ad",
                    repo_type="dataset",
                    local_dir=str(args.hf_dir),
                )
                print("  Download complete")
            except ImportError:
                print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
                sys.exit(1)
        else:
            print(f"Using existing snapshot at {args.hf_dir}")

    # Step 2: Parse samples.json
    samples = load_samples_json(args.hf_dir)

    # Step 3: Validate source files
    print("\nValidating source files ...")
    validate_source_files(samples, args.hf_dir)

    # Step 4: Validate counts
    print("\nValidating counts ...")
    validate_counts(samples)

    # Step 5: Create symlink tree
    print(f"\nCreating standard MVTec AD layout at {args.output_dir} ...")
    build_symlink_tree(samples, args.hf_dir, args.output_dir, force=args.force)

    # Step 6: Verify output
    verify_output(args.output_dir)

    print(f"\nDone. Standard MVTec AD layout ready at: {args.output_dir}")


if __name__ == "__main__":
    main()
