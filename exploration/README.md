# TOODLE Exploration & Analysis

This directory contains **reproducible investigations** that informed design decisions for the TOODLE project.

## Purpose

Per 0_OBJECTIVE.md deliverables:
> **Model Documentation** - Performance benchmarks, error analysis and failure cases
> **Architecture Documentation** - Technology choices with justifications

These investigations document **why** certain features were scoped in or out, providing evidence-based rationale for architectural decisions.

## Investigation Structure

Each subdirectory contains:
- **REPORT.md** - Consolidated findings and conclusions
- **investigate_*.py** - Self-contained reproducible analysis script
- **artifacts/** - Generated outputs (CSV/JSON, gitignored, regenerate locally)

## Current Investigations

### subcategory_independence/

**Question:** Why is subcategory prediction not included in Phase 1?

**Answer:** Statistical analysis across 7 independent tests proves subcategory is unpredictable from available features (accuracy ≤ 0.20 random baseline).

**Evidence:**
- Chi-squared uniformity: 4/5 categories perfectly uniform
- Conditional independence: No feature associations after Bonferroni correction
- ML upper bounds: GradientBoosting, joint-lookup, nested CV all ≤ 0.21
- Permutation tests: p = 1.0 (cannot beat shuffled labels)
- Power check: Method detects injected 0.28 accuracy with 100% power

**Impact:** Focus on category prediction (achieves >85% F1 with clean data). Subcategory scoped out as unpredictable, not due to model limitations.

## Running Investigations

Each investigation is self-contained and reproducible:

```bash
cd exploration/<investigation_name>
python investigate_*.py
```

Outputs are written to `artifacts/` (gitignored, regenerate locally).

## Design Philosophy

These are **research artifacts**, not production code. They document:
1. **What was tested** (methodology)
2. **What was found** (evidence)
3. **What was decided** (implications)

This ensures design decisions are traceable, evidence-based, and reproducible.
