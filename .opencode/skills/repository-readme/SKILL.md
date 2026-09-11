---
name: repository-readme
description: Use when the user asks to create, update, or improve a repository README. Triggers on "write the readme", "update the readme", "create readme", "improve the readme", "document the repo". Inspects the actual repository before writing; never invents content.
---

# Repository README Workflow

A lightweight, evidence-grounded workflow for creating or updating a
repository README based on what actually exists in the codebase.

## Ground Rules

- **Inspect before writing.** Read the repository structure, source code,
  configuration, existing docs, and tests before drafting anything.
- **Document reality.** Only describe features, commands, files, dependencies,
  and workflows that actually exist. Never invent them.
- **Verify when cheap.** Run `--help`, `python -m <module> --help`, or
  similar lightweight checks for important documented commands.
- **Mismatches are findings.** If existing docs contradict the implementation,
  report the mismatch rather than silently documenting nonexistent behavior.
- **Preserve useful content.** When updating an existing README, keep
  valuable sections and improve rather than rewrite from scratch.
- **No code changes for README convenience.** Do not modify source code
  merely to make the README easier to write.
- **No marketing language.** Write clearly and technically.

## Workflow

### Step 1: Discover the Repository

Inspect the repository to understand what it is and how it works:

- Read top-level files: `README.md`, `AGENTS.md`, `pyproject.toml`,
  `setup.py`, `setup.cfg`, `requirements*.txt`, `Makefile`, `Dockerfile`,
  `*.toml`, `*.cfg`.
- Explore the source tree structure.
- Identify entry points (CLI commands, main scripts, package `__main__.py`).
- Identify tests and how they are run.
- Identify configuration files and data directories.
- Check for existing documentation under `docs/` or similar.

### Step 2: Understand the Pipeline

Trace the primary execution flow:

- What does the main entry point do?
- What are the key stages (data loading, inference, evaluation, reporting)?
- What are the inputs (datasets, configs, checkpoints)?
- What are the outputs (results files, metrics, logs)?

Distinguish the **primary workflow** from optional or demo functionality.

### Step 3: Verify Key Commands

For any command that will be documented, verify it exists and check its
interface when lightweight verification is practical:

- `python -m <package> --help`
- `<script> --help`
- `pytest --co` (collect tests without running)
- Check `pyproject.toml` or `setup.cfg` for declared entry points

Note anything that could not be verified.

### Step 4: Draft the README

Write the README based on discovered information. Use the section
outline below as a guide — include sections that are relevant,
omit sections that are not applicable.

#### Section Outline

| Section | Purpose |
|---------|---------|
| **Overview** | What the project does, one or two paragraphs. No marketing. |
| **Pipeline / Architecture** | How the system is structured, key components and their relationships. |
| **Installation / Environment Setup** | How to install dependencies and prepare the environment. |
| **Data / Inputs** | What datasets or inputs are required, how to obtain and place them. |
| **Usage** | Primary commands and workflows with concrete examples. |
| **Outputs / Results** | What files or metrics are produced and where to find them. |
| **Testing** | How to run the test suite. |
| **Project Structure** | Key directories and their purpose. |
| **Optional / Demo** | Anything secondary — demos, notebooks, examples. Clearly labeled as optional. |

Rules for each section:

- Use real file paths, real commands, real dependency names.
- If a section would be empty, omit it rather than padding with filler.
- Use code blocks for commands and file listings.
- Keep it scannable: short paragraphs, bullets, tables where helpful.

### Step 5: Self-Check

Before presenting the final README, verify:

- Every command mentioned actually exists or is inferred from declared entry points.
- Every file path mentioned exists.
- Every dependency mentioned is declared in `pyproject.toml`, `requirements.txt`, or similar.
- Nothing contradicts the actual codebase.
- A reader could follow the instructions to install and run the project.

If something could not be verified, mark it explicitly:
`(unverified — confirm before publishing)`.

## Output

Produce a single `README.md` at the repository root, or present it
for the user to review before writing. Wait for explicit approval
before overwriting an existing README.
