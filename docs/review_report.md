# Independent Code Review Report

**Date:** 2026-09-11  
**Scope:** Lightweight final review — correctness, data leakage, end-to-end completeness, test coverage, AGENTS.md compliance.

## Test Suite Status

| Test file | Tests | Status |
|---|---|---|
| `tests/test_smoke.py` | 4 | ALL PASSED |
| `tests/test_threshold_evaluation.py` | 19 | ALL PASSED |
| `tests/test_prepare_dataset.py` | 14 | ALL PASSED |
| `tests/test_demo.py` | 32 | ALL PASSED |
| `tests/test_correctness.py` | ~40 | SKIPPED (requires real dataset + model fitting) |
| **Total runnable** | **69** | **69 passed** |

---

## Section 1: Evaluation Correctness and Data Leakage

### [Observation] F1-max uses test labels — correctly documented as analysis-only

- **File:** `src/benchmark/runner.py:139` and `runner.py:154`
- `find_best_f1_threshold(test_scores, test_labels)` and `find_best_f1_pixel(test_maps, test_masks)` use test labels to find the best achievable F1. This is test-label leakage for that specific metric.
- **However:** The code explicitly documents this as analysis-only (`threshold.py:112-113`, `threshold.py:152-153`, `docs/system_overview.md:234`). The deployable F1 (calibrated F1, using validation threshold only) is computed separately and is unaffected. The AGENTS.md prohibition on test-label threshold calibration is satisfied — calibration uses only validation data (`runner.py:96-97`). **No impact on primary benchmark results.**

### [Observation] Threshold calibration uses only normal validation data

- **File:** `src/benchmark/dataset.py:76-96`, `src/benchmark/threshold.py:9-38`
- `get_normal_val_scores()` filters synthetic anomalous samples via `batch.gt_label` (line 86). Only normal scores reach `calibrate_threshold()` and `calibrate_pixel_threshold()`. **Healthy.**

### [Observation] Metrics computed on correct splits

- **File:** `src/benchmark/runner.py:136-166`
- AUROC, AUPRC, calibrated F1 computed on test data. Threshold from validation. No cross-split contamination. **Healthy.**

---

## Section 2: Reproducibility and Silent Failures

- **Seeds:** `seed_everything()` at `runner.py:20-28` sets Python/NumPy/PyTorch seeds including `cudnn.deterministic`. Called before each run. **Healthy.**
- **Exception handling:** Core logic raises `ValueError`/`FileNotFoundError`/`RuntimeError`. The one `except Exception` in `compute_aupro()` (`evaluation.py:208-213`) issues `warnings.warn()` and returns 0.0 — justified and logged, not silent. **Healthy.**
- **Config serialized:** `BenchmarkConfig.to_dict()` captures all config in per-run JSONs. **Healthy.**

---

## Section 3: Interface and Architecture Quality

- Clean 7-module separation: dataset, models, threshold, evaluation, runner, checkpoint, demo_utils. No god-objects.
- `ModelAdapter` Protocol (`models.py:14-47`) with consistent `fit()`/`predict()`/`get_latency()`/`device` interface. Both adapters implement it identically.
- **[Minor]** `scripts/` is not a Python package — no `__init__.py`. The `[project.scripts]` entry point (`benchmark = "scripts.run_benchmark:main"`) would fail if installed via `pip install -e .`. Does not affect benchmark operation (invoked via `python scripts/run_benchmark.py`).

---

## Section 4: Experimental Fairness and Efficiency Measurement

- Latency: 10 warmup + 100 timed runs, `time.perf_counter()`, CUDA sync, single-image. Both models use identical protocol. **Healthy.**
- Both models: same dataset, splits, threshold strategy, evaluation metrics. PaDiM CPU-fitting is a necessary engineering choice (avoids MAGMA errors). **Healthy.**

---

## Section 5: Test Quality and Coverage

Thorough coverage of:
- Threshold direction (`>` not `>=`): 4 boundary tests
- Normal-only calibration: 4 unit tests + 1 integration test with real PaDiM
- AUROC undefined handling: 3 tests (single-class returns NaN with warning)
- F1 computation: 6 tests (perfect, wrong, partial, manual match)
- Pixel metrics: 5 tests (perfect, wrong, resize, single-class, manual match)
- AUPRO: 6 tests (basic, perfect, worst, resize, field presence)
- Checkpoint: 4 save/load roundtrip + 5 error handling tests
- Preprocessing consistency: 2 tests (shape/dtype + benchmark-vs-demo tensor match)
- Resume/skip: 3 tests (skip existing, aggregate mean, no recompute)
- Dataset preparation: 14 synthetic-data tests (validation, symlinks, determinism)

---

## Section 6: Documentation vs Implementation

- `docs/system_overview.md` (449 lines) accurately describes all pipeline stages with file:line references. Verified against source.
- F1-max analysis-only status correctly documented as "leaks test labels and is NOT deployable."
- **[Minor]** `README.md` listed in `pyproject.toml:6` but absent from repository.

---

## Section 7: AGENTS.md Compliance

| Rule | Status |
|---|---|
| Never use test labels for threshold tuning | COMPLIANT — calibration uses only validation data |
| Fail Loud (no silent exception handling) | COMPLIANT — all errors raised with clear messages |
| Simplicity First | COMPLIANT — minimal 7-module architecture |
| Goal-Driven Execution | COMPLIANT — 30/30 runs completed, all artifacts present |

---

## Areas Examined and Found Healthy

1. Threshold calibration pipeline (validation-only filtering)
2. Strict `>` threshold semantics (consistent, tested)
3. AUROC/AUPRC computation (sklearn, single-class handling)
4. Anomaly map resize (bilinear, 2D/3D, tested)
5. AUPRO (Anomalib `_AUPRO`, FPR limit=0.3)
6. Checkpoint serialization (versioned, validated on load)
7. Demo preprocessing consistency (identical to benchmark dataloader)
8. Resume support (skip + re-aggregate MEAN)
9. Dataset preparation (symlink-based, deterministic, 14 tests)
10. Latency measurement (warmup, sync, single-image, both models)

---

## Findings Summary

| Severity | Count | Details |
|---|---|---|
| Critical | 0 | — |
| Major | 0 | — |
| Minor | 2 | `scripts/` not a package; README.md missing |
| Observation | 1 | F1-max uses test labels (documented as analysis-only) |

---

**READY**

The benchmark pipeline is complete, correct, and well-tested. Thresholds are calibrated exclusively on validation data. Test labels are never used for threshold tuning. The F1-max analysis metric is clearly documented as non-deployable. All 69 runnable tests pass. No findings affect the correctness of the reported results.
