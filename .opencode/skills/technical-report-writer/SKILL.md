---
name: technical-report-writer
description: Use when the user asks to write a technical report about a completed engineering or ML benchmark project. Triggers on "write the report", "create the report", "technical report", "write up the project". Derives all claims from actual project evidence; never invents content.
---

# Technical Report Workflow

A lightweight, evidence-grounded workflow for writing a concise technical
report from a completed engineering project.

## Ground Rules

- **Inspect before writing.** Gather evidence from the repository, tests,
  results, documentation, review findings, and user-supplied context before
  drafting.
- **Never invent.** Do not fabricate implementation details, experiments,
  results, tests, review findings, or development history. If evidence is
  missing, say so.
- **Traceable claims.** Every quantitative claim must be traceable to an
  actual result artifact or other supplied evidence.
- **No result manipulation.** Never alter, improve, round misleadingly, or
  fabricate benchmark results.
- **Measured vs. interpreted.** Clearly distinguish measured results from
  your interpretation of those results.
- **Cross-check.** Verify important claims against source code, documentation,
  tests, result artifacts, or supplied evidence.
- **Surface contradictions.** If evidence sources disagree, report the
  contradiction rather than silently reconciling it.
- **Summarize, don't dump.** Summarize development evidence; do not paste
  raw prompts, logs, or transcripts.
- **No rework for the report.** Do not modify experiments or rerun expensive
  workloads to generate additional report material.
- **Respect page limits.** If a length is specified, stay within it.

## Evidence Hierarchy

Treat evidence in this order of reliability:

1. **Measured results** from result artifacts (JSON, CSV, logs).
2. **Test outputs** confirming correctness.
3. **Source code** showing actual implementation.
4. **Documentation** describing intended behavior.
5. **User-supplied context** about decisions and history.

When sources conflict, prefer the higher-reliability source and note the
discrepancy.

## Workflow

### Step 1: Gather Evidence

Collect available evidence from the project:

- Source code: architecture, entry points, key abstractions.
- Tests: what is covered, test results if available.
- Result artifacts: metrics, logs, benchmark outputs.
- Documentation: README, AGENTS.md, docs/ directory.
- Version control: commit history, branches, PRs.
- Code review findings if available.
- User-supplied development notes or AI usage history.

Record what exists and what is missing.

### Step 2: Establish Scope

Determine the report's scope and structure:

- What assignment or objective is the report addressing?
- What page limit or format constraints apply?
- What sections are required vs. optional?
- What evidence is available for each potential section?

Organize around the assignment requirements and available evidence.
Do not force sections that lack supporting evidence.

### Step 3: Organize the Narrative

For AI-assisted projects, organize development evidence as:

```
Task / Objective
  -> AI Contribution
  -> Human Decision or Intervention
  -> Verification
```

Do not claim AI performed a step unless evidence supports it.
Human decisions that shaped direction (model choice, threshold strategy,
architecture change, scope cuts) should be explicitly credited.

### Step 4: Draft the Report

Write the report using the evidence gathered. Suggested sections for an
ML benchmark or engineering project — use what fits, omit what does not:

| Section | Purpose |
|---------|---------|
| **Objective and System Design** | What was built and why. High-level architecture. |
| **Benchmark / Evaluation Protocol** | Datasets, splits, metrics, threshold policy, fairness controls. |
| **AI-Assisted Development and Testing** | How AI was used, what it contributed, what humans decided. |
| **Results and Trade-offs** | Measured outcomes, performance characteristics, design trade-offs. |
| **Independent Review / Verification** | Review findings, severity rankings, what was fixed. |
| **Limitations and Lessons Learned** | What did not work, what was cut, what would change. |

Rules for drafting:

- Use real file paths, real commands, real metric values.
- Every metric must cite its source artifact.
- If a section lacks evidence, state that rather than speculate.
- Keep prose concise; use tables and lists for dense information.
- Distinguish "implemented and verified" from "implemented but unverified"
  from "planned but not completed."

### Step 5: Cross-Check

Before finalizing, verify:

- All quantitative claims are backed by result artifacts or test output.
- All described features exist in the codebase.
- All commands mentioned actually work (or are marked unverified).
- No fabricated development history or AI contributions.
- No result values were altered or rounded misleadingly.
- Contradictions are noted, not hidden.

### Step 6: Present for Review

Present the draft to the user for review before finalizing. Flag:

- Any evidence gaps that could not be filled.
- Any sections that are thin due to missing evidence.
- Any claims that are interpretations rather than measurements.

## Output

A single markdown document suitable for the assignment's submission format.
Do not create the final file until the user approves the draft.
