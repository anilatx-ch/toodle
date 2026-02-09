# Porting Agent Instructions

## Task Context
You are helping port code from ../DOODLE to this TOODLE project. Each file is inspected
and cleaned during porting — this is not a copy operation.

## Quality Bar
This is a **technical assessment for a Full-Stack AI Engineer role**. The code must
demonstrate the developer's qualities: clean design, practical trade-offs, production
thinking. It is **not** a production system — it's a one-work-day demo.

## The Core Question
For every function, class, config block, and error handler, ask:
**"Could this be done cleaner or leaner in a way that improves readability, simplicity,
or maintainability?"**

## What To Cut
1. **Defensive code for impossible scenarios** — e.g., checking if a config dict has
   a key that's hardcoded three lines above. Trust internal invariants.
2. **Backward-compatibility shims** — there is no old version to be compatible with.
   Remove aliases, legacy branches, "kept for backward compatibility" comments.
3. **Comments that restate the code** — `# Load the model` above `model = load(path)`
   adds nothing. Only comment non-obvious *why*, never obvious *what*.
4. **Verbose error messages with usage instructions** — a `ValueError` with a clear
   message is enough. No need for multi-line f-strings explaining every valid option.
5. **Excessive logging** — one info log per major operation. Remove debug-level noise
   and per-item logging in loops.
6. **Dead code paths** — if a branch can't be reached given current config, remove it.
7. **Over-parameterized functions** — if a parameter is always called with the same
   value, inline it.

## What To Keep
1. **Type hints** — they're documentation that the compiler checks.
2. **Dataclasses for structured data** — clean, Pythonic.
3. **MLflow logging** — demonstrates MLOps awareness (jobspec requirement).
4. **Error handling at system boundaries** — API endpoints, file I/O, model loading.
5. **Deterministic seeds** — reproducibility matters for the assessment.
6. **The actual ML logic** — feature engineering choices, model architectures,
   evaluation metrics. This is what's being assessed.

## Style
- Imports: stdlib → third-party → local, alphabetical within groups.
- Line length: 100 chars soft limit (Black default).
- Naming: snake_case functions/variables, PascalCase classes, UPPER_CASE constants.
- No print() in library code — use logging. print() is fine in CLI entry points.
- Prefer returning values over mutating arguments.
- Prefer composition over inheritance.

## Scope Boundaries
- **Two traditional ML backends**: CatBoost + XGBoost (no LightGBM)
- **One deep learning backend**: DistilBERT
- **/predict returns**: category (real model), priority (placeholder), sentiment (placeholder)
- **Real sentiment** via dedicated /analyze-feedback endpoint
- **No subcategory code** — all subcategory configs, paths, model dirs removed
- **Clean training data**: ALL models train on ~110 deduplicated subject→category pairs (not noisy 100K)
- **Full 100K through dbt** — for RAG corpus and data engineering demonstration
- **SMOKE_TEST boolean** controls sample sizes; ENV controls path namespacing only

## Testing Protocol
- Each stage must pass `pytest` before commit
- Tests should be minimal but meaningful — test behavior, not implementation
- Mock heavy dependencies (TensorFlow, FAISS) in unit tests
- Integration tests can use smoke-sized data

## Git Protocol
- Agent suggests commit commands; user executes them
- One commit per stage, descriptive message summarizing changes

## Path Safety Check
- **CRITICAL**: Before each stage completion, check all git-tracked files for absolute paths (especially `/home/`)
- Use: `grep -r "/home/" --include="*.py" --include="*.toml" --include="*.yml" --include="*.md" --include="Makefile" . | grep -v ".venv" | grep -v "__pycache__" | grep -v ".pyenv"`
- Absolute paths break portability - use relative paths or Path(__file__).parent
- Build artifacts (.venv, .pyenv, etc.) are excluded from this check
