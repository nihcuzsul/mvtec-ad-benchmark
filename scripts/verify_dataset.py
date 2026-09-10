#!/usr/bin/env python3
"""Verify MVTec AD dataset integrity for all 15 categories."""

import sys
from pathlib import Path

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def verify_category(root: Path, category: str) -> dict:
    """Verify a single category directory."""
    cat_dir = root / category
    result = {
        "category": category,
        "exists": cat_dir.is_dir(),
        "train_images": 0,
        "test_images": 0,
        "gt_masks": 0,
        "train_ok": False,
        "test_ok": False,
        "gt_ok": False,
        "zero_byte_files": [],
        "unreadable_files": [],
    }

    if not result["exists"]:
        return result

    # Check train/ directory
    train_dir = cat_dir / "train"
    if train_dir.is_dir():
        # MVTec train has subdirectories per defect type (usually just "good")
        train_files = list(train_dir.rglob("*.png"))
        result["train_images"] = len(train_files)
        result["train_ok"] = result["train_images"] > 0
        for f in train_files:
            if f.stat().st_size == 0:
                result["zero_byte_files"].append(str(f))
            try:
                f.read_bytes()
            except Exception:
                result["unreadable_files"].append(str(f))

    # Check test/ directory (images in subdirectories by defect type)
    test_dir = cat_dir / "test"
    if test_dir.is_dir():
        test_files = list(test_dir.rglob("*.png"))
        result["test_images"] = len(test_files)
        result["test_ok"] = result["test_images"] > 0
        for f in test_files:
            if f.stat().st_size == 0:
                result["zero_byte_files"].append(str(f))
            try:
                f.read_bytes()
            except Exception:
                result["unreadable_files"].append(str(f))

    # Check ground_truth/ directory
    gt_dir = cat_dir / "ground_truth"
    if gt_dir.is_dir():
        gt_files = list(gt_dir.rglob("*.png"))
        result["gt_masks"] = len(gt_files)
        result["gt_ok"] = result["gt_masks"] > 0
        for f in gt_files:
            if f.stat().st_size == 0:
                result["zero_byte_files"].append(str(f))
            try:
                f.read_bytes()
            except Exception:
                result["unreadable_files"].append(str(f))

    return result


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./datasets/MVTecAD")

    print(f"Verifying MVTec AD dataset at: {root}")
    print("=" * 75)

    all_ok = True
    results = []

    for cat in MVTEC_CATEGORIES:
        r = verify_category(root, cat)
        results.append(r)

        status = "OK" if (r["train_ok"] and r["test_ok"]) else "MISSING"
        if r["zero_byte_files"] or r["unreadable_files"]:
            status = "CORRUPTED"

        if status != "OK":
            all_ok = False

        gt_str = f"gt={r['gt_masks']:3d}" if r["gt_masks"] > 0 else "gt=  0"
        issues = ""
        if r["zero_byte_files"]:
            issues += f" [ZERO-BYTE: {len(r['zero_byte_files'])}]"
        if r["unreadable_files"]:
            issues += f" [UNREADABLE: {len(r['unreadable_files'])}]"

        print(f"  {cat:15s}  train={r['train_images']:3d}  test={r['test_images']:3d}  "
              f"{gt_str}  {status}{issues}")

    print("=" * 75)
    total_train = sum(r["train_images"] for r in results)
    total_test = sum(r["test_images"] for r in results)
    total_gt = sum(r["gt_masks"] for r in results)
    missing = [r["category"] for r in results if not r["train_ok"] or not r["test_ok"]]
    corrupted = [r["category"] for r in results if r["zero_byte_files"] or r["unreadable_files"]]

    print(f"  TOTAL: train={total_train}  test={total_test}  gt_masks={total_gt}")
    if missing:
        print(f"  MISSING: {', '.join(missing)}")
    if corrupted:
        print(f"  CORRUPTED: {', '.join(corrupted)}")
    if all_ok:
        print("  ALL CATEGORIES OK")
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
