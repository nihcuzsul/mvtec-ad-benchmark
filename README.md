# MVTec AD Anomaly Detection Benchmark

Automated benchmark pipeline for PaDiM and PatchCore anomaly detection on
the MVTec AD dataset. Both models share a unified `ModelAdapter` interface
with consistent `fit()`, `predict()`, and `get_latency()` methods.

## Pipeline

```
run_benchmark.py
  → auto-prepare dataset (Voxel51/mvtec-ad → MVTecAD)
  → fit model on training data
  → calibrate thresholds on validation data
  → evaluate on test data
  → save results to results/
```

## Project Structure

```
src/benchmark/
    config.py          Config dataclasses
    dataset.py         MVTecADWrapper (anomalib datamodule)
    models.py          PaDiMAdapter, PatchCoreAdapter
    threshold.py       Threshold calibration
    evaluation.py      Image/pixel metrics, AUPRO
    runner.py          Benchmark orchestration and aggregation
    checkpoint.py      Checkpoint save/load (demo only)
    demo_utils.py      Visualization utilities (demo only)

scripts/
    run_benchmark.py   Primary benchmark CLI
    prepare_dataset.py Voxel51 → MVTecAD layout adapter
    demo.py            Single-image inference demo (optional)
    fit_checkpoint.py  Save fitted checkpoint (optional)

tests/                 Test suite (6 files)
requirements.txt       Direct dependencies with CUDA 12.8 index
pyproject.toml         Project/package configuration
AGENTS.md              OpenCode development instructions
.opencode/skills/      OpenCode workflow skills
docs/review_report.md  OpenCode independent code review
```

## Environment Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verified with Python 3.12.3, PyTorch 2.8.0+cu128, Anomalib 2.6.1.

## Benchmark Usage

On first run, `run_benchmark.py` automatically downloads and prepares the
Voxel51/mvtec-ad dataset. Re-running skips completed categories.

```bash
# One model, one category
python scripts/run_benchmark.py --categories bottle --models padim

# Both models, all 15 categories
python scripts/run_benchmark.py --categories all --models padim patchcore
```

Run `python scripts/run_benchmark.py --help` for all options.

## Generated Outputs

Results are written to `results/` (not included in Git):

- Per-run: `results/{model}_{category}_results.json`
- Aggregate: `results/summary.csv`

## Single-Image Demo (Optional)

Requires a fitted checkpoint (needs prior benchmark results for thresholds):

```bash
# Fit checkpoint
python scripts/fit_checkpoint.py \
    --model patchcore --category bottle \
    --output checkpoints/patchcore_bottle.pt

# Run inference
python scripts/demo.py \
    --checkpoint checkpoints/patchcore_bottle.pt \
    --image path/to/image.png \
    --output-dir demo_output/
```

Model and category are auto-detected from the checkpoint.

## Testing

```bash
pytest
```

- **Unit tests**: Threshold calibration, F1 computation, preprocessing,
  visualization, device resolution.
- **Integration tests**: Dataset preparation, checkpoint roundtrip,
  resume logic, demo inference.
- **Correctness tests**: Dataset/model-dependent validation (slower,
  requires downloaded dataset).
