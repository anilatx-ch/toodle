# Submission Checklist and Package Contents

This document defines what is included in `../full-stack-ai-solution.zip` and what is intentionally excluded.

## Deliverables Checklist

| Deliverable (from `0_OBJECTIVE.md`) | Status | Evidence |
|---|---|---|
| 1. Project code and working system | Met | `src/`, `scripts/`, `dbt_project/`, `Dockerfile`, `docker-compose.yml`, `Makefile`, `tests/` (`87 passed`) |
| 2. Architecture documentation | Met | `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/API_CONTRACT.md` |
| 3. Model documentation | Met | `docs/MODEL.md` (restored canonical long-form document) |
| 4. README with setup and usage | Met | `README.md` |

## Included in Submission Zip

The package target includes only these paths:

- `0_OBJECTIVE.md`
- `0_schema.json`
- `README.md`
- `pyproject.toml`
- `poetry.lock`
- `Makefile`
- `Dockerfile`
- `docker-compose.yml`
- `run_evaluation.py`
- `schema.py`
- `src/`
- `scripts/`
- `tests/`
- `dbt_project/`
- `docs/`
- `schemas/`
- `exploration/README.md`
- `exploration/subcategory_independence/`

## Excluded from Submission Zip

The package intentionally excludes local/dev/internal artifacts:

- Planning/internal process docs:
  - `CONTEXT.md`
  - `PLAN_PORTING.md`
  - `TASK_PORTING.md`
  - `AGENTS_PORTING.md`
  - `CLAUDE.md`
  - `archive/`
- Local debugging notes:
  - `BUGS.md`
  - `BUGS2.md`
  - `tools/`
- Runtime and generated artifacts:
  - `.git/`, `.venv/`, `.pyenv/`, `.pytest_cache/`, `.idea/`
  - `data/`, `models/`, `metrics/`, `figures/`, `reports/`, `diagnostics/`, `mlruns/`
  - `support_tickets.json`

## Packaging Command

```bash
make package
```

Output:

```text
../full-stack-ai-solution.zip
```
