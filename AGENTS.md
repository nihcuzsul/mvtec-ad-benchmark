# AGENTS.md

These instructions apply to all AI-assisted development in this project.

## Project Goal

Build an automated, reproducible, and extensible benchmark pipeline for
evaluating multiple anomaly detection models on the MVTec AD dataset.

The project should prioritize automation, low manual effort, ease of use,
correctness, and maintainability.

## 1. Think Before Coding

Before significant implementation:
- State important assumptions.
- Surface ambiguity instead of guessing.
- Prefer the simpler approach when multiple solutions satisfy the goal.
- Briefly explain major technical decisions before implementation.

## 2. Simplicity First

Implement the minimum architecture needed to satisfy the assignment.

- No speculative features.
- Avoid unnecessary abstractions.
- Avoid dependencies that do not provide clear value.
- Prefer readable and maintainable code over clever code.

## 3. Goal-Driven Execution

For each significant task:
1. Define the expected outcome.
2. Implement it.
3. Run the relevant command or test.
4. Fix failures when possible.
5. Verify the final result.

Do not claim completion without verification.

## 4. Read Before You Write

Before modifying existing code:
- Inspect the relevant module and interfaces.
- Check immediate dependencies and callers when applicable.
- Preserve established project conventions.

Do not modify unrelated code.

## 5. Benchmark Correctness

The benchmark must:
- support a unified model inference interface;
- keep dataset, model, evaluation, and reporting logic modular;
- produce reproducible experiments;
- support CPU execution with optional CUDA acceleration;
- export machine-readable results.

Primary evaluation outputs:
- image-level AUROC;
- AUPRC;
- F1 with an explicitly documented threshold policy;
- inference latency.

Never use test labels for model training or threshold tuning.

## 6. Testing

Core functionality requires tests.

Tests should cover meaningful behavior and important edge cases rather than
only checking that functions execute.

After changes affecting core functionality, run the relevant tests.

## 7. Checkpoint Significant Work

After each significant implementation step, briefly report:
- what changed;
- what was verified;
- any remaining issue or assumption.

Keep summaries concise.

## 8. Fail Loud

Do not silently:
- skip tests;
- ignore errors;
- catch exceptions without justification;
- change the evaluation protocol;
- substitute unavailable functionality.

Clearly report uncertainty or incomplete work.

## 9. Development Environment

Do not assume a Python version or install major dependencies without first
checking compatibility with the selected implementation.

Prefer the smallest dependency stack that satisfies the project requirements.

## 10. Human Decision Boundary

For consequential decisions such as:
- model selection;
- evaluation protocol;
- threshold strategy;
- major dependencies;
- architecture changes;

propose the approach and rationale first when the decision has not already
been approved.

Routine implementation, debugging, testing, and deterministic operations may
proceed autonomously once the approach is approved.