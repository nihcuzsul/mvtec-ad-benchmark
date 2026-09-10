#!/usr/bin/env python3
"""Extract and repair MVTec AD dataset from downloaded archive.

Usage:
    python scripts/repair_dataset.py

Assumes datasets/MVTecAD/mvtec_anomaly_detection.tar.xz exists and is complete.
Extracts in-place, preserving existing complete files.
"""

import subprocess
import sys
from pathlib import Path

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def check_archive(archive_path: Path) -> bool:
    """Verify archive is complete and not truncated."""
    expected_size = 5_264_982_680  # 5.26 GB from Content-Length
    actual_size = archive_path.stat().st_size
    if actual_size < expected_size:
        print(f"ERROR: Archive truncated ({actual_size / 1e9:.2f} GB vs expected {expected_size / 1e9:.2f} GB)")
        return False
    print(f"Archive size OK: {actual_size / 1e9:.2f} GB")
    return True


def extract_archive(archive_path: Path, target_dir: Path) -> bool:
    """Extract tar.xz archive to target directory."""
    print(f"Extracting {archive_path.name} to {target_dir}...")
    try:
        result = subprocess.run(
            ["tar", "xJf", str(archive_path), "-C", str(target_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"tar extraction failed: {result.stderr}")
            return False
        print("Extraction complete.")
        return True
    except subprocess.TimeoutExpired:
        print("ERROR: Extraction timed out (600s)")
        return False


def main() -> int:
    root = Path("./datasets/MVTecAD")
    archive = root / "mvtec_anomaly_detection.tar.xz"

    if not archive.exists():
        print(f"ERROR: Archive not found: {archive}")
        return 1

    # Step 1: Verify archive
    if not check_archive(archive):
        return 1

    # Step 2: Extract
    if not extract_archive(archive, root):
        return 1

    # Step 3: Verify
    print("\nRunning dataset verification...")
    result = subprocess.run(
        [sys.executable, "scripts/verify_dataset.py", str(root)],
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
