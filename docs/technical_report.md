# MVTec AD Anomaly Detection Benchmark Pipeline

## 1. Objective and System Overview

**Objective:** Build an automated, reproducible benchmark pipeline for evaluating anomaly detection models on the MVTec AD dataset, with emphasis on correctness, low manual effort, and extensibility.

**Why MVTec AD:** MVTec AD is a standard industrial anomaly detection benchmark with 15 object/texture categories, providing both image-level and pixel-level ground truth. The assignment provides the dataset in Voxel51/mvtec-ad format (HuggingFace FiftyOne export), requiring conversion to the standard MVTec AD directory layout.

**Models:** PaDiM (deep feature distribution modeling) and PatchCore (coreset-subsampled patch embeddings) — two representative approaches with different accuracy-latency trade-offs. Both are implemented via Anomalib and wrapped behind a unified `ModelAdapter` protocol with consistent `fit()`, `predict()`, and `get_latency()` methods.

**Scale:** 15 categories × 2 models = 30 automated experiments, each producing image-level and pixel-level metrics.

**Pipeline:**

```
Voxel51/mvtec-ad
  → automatic dataset preparation (symlink-based)
  → ModelAdapter.fit() on normal training images
  → threshold calibration on normal validation data
  → test evaluation (image-level + pixel-level)
  → JSON + CSV outputs with resume/skip support
```

The single entry point `scripts/run_benchmark.py` handles dataset preparation, model fitting, threshold calibration, evaluation, and result aggregation. On first run, it automatically downloads and converts the Voxel51 dataset. Re-running skips completed categories.

[Figure 1: Automated benchmark pipeline]

## 2. AI-Assisted Development

OpenCode (an open-source AI coding assistant, powered by free open-weight models) served as an engineering agent across planning, implementation, refactoring, debugging, testing, review, and documentation. Routine engineering work was delegated to the AI agent; human involvement was reserved for defining the benchmark objective, making consequential decisions that could affect benchmark validity or project scope, and validating the final deliverables.

### Development Workflow

| Phase | AI Contribution | Human Role |
|-------|----------------|------------|
| **Architecture design** | Designed the modular benchmark pipeline, unified `ModelAdapter` interface, and supporting workflow | Defined benchmark objective and model scope |
| **Implementation & Refactoring** | Assisted with implementation and refactoring of pipeline modules, scripts, and model adapters | Validated key design choices |
| **Dataset Preparation** | Built automated dataset preparation, conversion, integrity checking, and skip logic | Confirmed final dataset integrity |
| **Testing & Debugging** | Generated tests, diagnosed failures, and assisted with fixes and regression verification | Checked final test status |
| **Code Review** | Performed a fresh structured repository review using the `independent-code-review` Skill | Reviewed the final QA summary |
| **Documentation** | Generated and refined README and technical-report content from repository evidence | Finalized submission content |

### Skills and Persistent Instructions

Three project-local OpenCode Skills specialized key workflows: `independent-code-review` for structured severity-ranked code review, `repository-readme` for evidence-based documentation, and `technical-report-writer` for traceable report generation.

`AGENTS.md` provided persistent project-level instructions across sessions, enforcing goal-driven execution, simplicity-first architecture, and fail-loud error handling. Routine implementation, debugging, testing, and documentation were delegated to the AI agent, while human approval was reserved for consequential decisions that could change benchmark validity or project scope. The combination of persistent instructions, reusable Skills, and automated testing reduced repeated prompting and manual intervention across development sessions.

[Figure 2: OpenCode AI-assisted development/review example]

## 3. Benchmark and Testing Protocol

### Evaluation Protocol

Each of the 30 experiments follows:

1. **Training**: Category-specific model fitted on normal training images only
2. **Threshold calibration**: Percentile-based thresholds computed on normal validation data (synthetic anomalous samples filtered out)
3. **Test evaluation**: Image-level and pixel-level metrics computed on held-out test data using validation-calibrated thresholds
4. **Latency measurement**: Single-image inference, 10 warmup runs followed by 100 timed runs with CUDA synchronization

**Metrics:** AUROC, AUPRC (image and pixel level), AU-PRO (pixel level, FPR limit=0.3), inference latency (ms/image).

**Data leakage prevention:** Test labels are never used for model fitting or deployable threshold calibration.

### Testing Levels

The test suite covers four levels of verification:

- **Unit tests**: Threshold direction (strict `>` semantics), calibration strategies, F1 computation, AUROC/AUPRC edge cases, pixel metrics, AUPRO, device resolution, and preprocessing consistency.
- **Component and regression tests**: Checkpoint save/load roundtrip, threshold loading from result files, resume/skip behavior, demo inference consistency, and headless execution.
- **Synthetic dataset-preparation integration tests**: Symlink tree creation, source file validation, count verification, output integrity, and `ensure_dataset()` skip/rebuild logic — all using synthetic temporary data.
- **Correctness tests with dataset and model**: Model output shapes, GPU latency measurement, validation-score filtering, and PatchCore-specific embedding verification. These fit real models and require the downloaded dataset.

### Independent Code Review

Final QA reported 111 passing tests with zero failures and no critical or major issues. The independent review also confirmed validation-only threshold calibration and no train/test leakage in deployable evaluation. See `docs/review_report.md` for the detailed review record.

## 4. Results and Analysis

All 30 experiments completed successfully. Unweighted macro averages across 15 categories:

| Metric | PaDiM | PatchCore |
|--------|-------|-----------|
| Image AUROC | 0.8801 | 0.9578 |
| Image AUPRC | 0.9398 | 0.9820 |
| Pixel AUROC | 0.9540 | 0.9753 |
| Pixel AUPRC | 0.3857 | 0.5438 |
| Pixel AU-PRO | 0.8738 | 0.9219 |
| Latency (ms/image) | 1.85 | 17.98 |

**Key observations:**

- **PatchCore** provides stronger aggregate detection and localization performance across all metrics, consistent with its patch-level feature comparison approach.
- **PaDiM** achieves substantially lower inference latency (1.85 ms vs 17.98 ms), making it more suitable for latency-sensitive applications.
- The benchmark exposes a clear accuracy-efficiency trade-off under a common evaluation protocol with identical dataset splits, threshold strategy, and metric computation.

## 5. Conclusion and Lessons Learned

**Deliverable:** An automated, reproducible benchmark pipeline for MVTec AD anomaly detection, with PaDiM and PatchCore evaluated across all 15 categories under a consistent protocol.

**Key lessons:**

- **Selective human oversight:** Routine engineering tasks were delegated to the AI agent, while human intervention focused on benchmark-defining decisions and final validation.
- **Automated verification reduces risk:** Automated tests were repeatedly used after implementation and fixes to detect regressions. The independent code review confirmed no critical issues.
- **Reproducibility requires deliberate design:** Seed management, config serialization, and resume support were all explicitly designed for reproducibility rather than added as afterthoughts.