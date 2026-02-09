# AGENTS.md

**Purpose:** Instructions for AI agents working on this repository. These instructions override default agent behaviors.

## Understanding This Project First

Before executing tasks, read **CONTEXT.md** to understand the project's objectives, scope, constraints, and current status.

## Code Quality Expectations

This is a **technical assessment demo** for a Full-Stack AI Engineer role. Code quality matters:

- Write **clean, industry-standard code** that demonstrates production thinking
- Use **proper abstractions** and maintainable patterns
- However, **avoid overengineering** - prefer demonstrating approach over perfect implementation
- Scope is constrained (approximately one working day) - focus on system thinking and clear technical decisions rather than exhaustive features

Balance professional quality with practical scope constraints.

## Execution policy for this repository

- Run commands outside the sandbox by default (use escalated permissions).
- This applies to all `make` commands and targets, including composed targets like `make all`.
- This also applies to `poetry` commands (`poetry install`, `poetry run ...`, `poetry lock`, tests, training, API runs).
- Always run Python and Poetry from the project venv: `.venv/bin/python` and `.venv/bin/poetry`.
- Prefer outside-sandbox execution for all project runtime commands that touch ML frameworks, local services, or IPC/shared-memory paths.

### Commonly sandbox-limited operations (run outside sandbox)

- `poetry` and Python runtime tasks (imports, training, evaluation, quantization, API startup).
- `pytest` runs that initialize TensorFlow/CatBoost/FastAPI test clients.
- Local service/network checks (`uvicorn`, `curl` to localhost, health/predict endpoint checks).
- Docker and compose workflows.
- dbt commands and DuckDB-backed data pipeline steps.
- Any command that writes outside repo-writable roots or relies on system resources/devices.

- NEVER run `python -c` or inline heredoc Python commands if the command exceeds 200 characters. ALWAYS write a script file instead.
- Script filename format: `adhoc_<descriptive_name>.py` where `<descriptive_name>` is 3-5 words separated by underscores.

## Artifact layout policy

- `reports/` is Markdown-only deliverables (`.md`).
- Non-Markdown analysis artifacts (`.csv`, `.json`, `.parquet`, etc.) must go under `diagnostics/` (or existing typed output dirs like `metrics/`, `figures/`, `data/processed/` as appropriate).
- For data quality and exploratory analysis outputs, prefer `diagnostics/data_investigation/`.

## Path Safety

- **NEVER use absolute paths** (especially `/home/`) in git-tracked files
- Use relative paths or `Path(__file__).parent` for portability
- Before committing, verify: `grep -r "/home/" src/ tests/ *.py *.toml *.yml *.md Makefile 2>/dev/null`
