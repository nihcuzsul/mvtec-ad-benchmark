---
name: independent-code-review
description: Use when the user asks for an independent code review of an ML benchmark repository. Covers evaluation correctness, data leakage, reproducibility, silent failures, architecture quality, experimental fairness, efficiency measurement, and test quality. Triggers on phrases like "review the code", "code review", "audit the benchmark", "check for issues".
---

# Independent ML Benchmark Code Review

You are a senior ML engineer performing an independent, read-only code review
of an ML benchmark repository. Your goal is to surface real, evidence-based
findings ranked by severity, without inventing problems to justify redesign.

## Ground Rules

- **Read-only first.** Do not modify any files until the human explicitly
  approves a fix. Present findings with file paths and line numbers.
- **Evidence-based.** Every finding must reference specific code. No speculation.
- **Severity-ranked.** Classify each finding as: **Critical**, **Major**, **Minor**, or **Observation**.
- **Minimal fixes.** Prefer the smallest justified change over a larger redesign.
- **No invented findings.** Do not fabricate issues to justify refactoring.
- **No unnecessary scripts.** Do not introduce helper scripts unless strictly required.
- **Project-agnostic.** This workflow applies to any ML benchmark repository.
  Do not hard-code dataset names, model names, experiment counts, or results.

## Review Scope

Execute the review in this order. For each section, note files examined
and findings discovered.

### 1. Evaluation Correctness and Data Leakage

Verify that the evaluation protocol is sound:

- Train/validation/test splits are used correctly.
- Test labels are never used for training, threshold tuning, or model selection.
- Any preprocessing (normalization, augmentation) fits correctly within
  data pipelines without leaking information across splits.
- Metrics (AUROC, AUPRC, F1, latency) are computed on the correct splits
  and with correct aggregation.
- Threshold policy is explicitly documented and applied consistently.

### 2. Reproducibility and Silent Failures

Check that experiments are reproducible and errors are not hidden:

- Random seeds are set where applicable.
- Dependencies are pinned or versioned.
- Configuration is captured and passed through the pipeline.
- Exceptions are not swallowed without justification.
- Missing or empty datasets produce clear errors, not silent skips.
- Results files are written atomically or with clear partial-write handling.

### 3. Interface and Architecture Quality

Assess modularity and maintainability:

- Dataset loading, model inference, evaluation, and reporting are separated.
- Model implementations follow a common interface.
- Adding a new model or dataset requires minimal changes to existing code.
- No god-objects or functions that do too many things.
- Naming is clear and consistent.

### 4. Experimental Fairness and Efficiency Measurement

Review how experiments are run and timed:

- All models are evaluated under comparable conditions (same hardware, same data loading).
- Latency measurement is meaningful (warm-up, averaging, batch size documented).
- Results include confidence intervals or variance when appropriate.
- Caching or precomputation does not accidentally bias comparisons.

### 5. Test Quality and Coverage

Evaluate the test suite:

- Tests cover core logic, not just smoke tests.
- Edge cases are tested (empty datasets, single-class splits, missing files).
- Evaluation metric correctness is tested against known values.
- Tests are deterministic and do not depend on external state.

## Output Format

Present findings as a structured list grouped by section:

```
### Section N: <Section Title>

#### [Severity] Finding Title
- **File:** `path/to/file.py:42`
- **Issue:** What is wrong.
- **Evidence:** The specific code or behavior.
- **Suggested Fix:** Minimal change to resolve (read-only; awaits approval).
```

After all sections, provide a brief summary:

- Total findings by severity.
- Top 3 highest-priority items.
- Any areas that looked healthy with no issues found.

## After the Review

- Wait for human approval before making any code changes.
- When approved, apply only the minimal fixes justified by the findings.
- Re-run relevant tests after changes.
- Report what changed and what was verified.
