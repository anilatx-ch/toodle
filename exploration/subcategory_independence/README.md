# Subcategory Independence Investigation

**Quick Reference Card**

## Purpose

Prove that subcategory cannot be predicted from available features with accuracy significantly above random guessing (0.20 for 5 uniform classes).

## Key Finding

**Subcategory is unpredictable** - 8 independent statistical tests converge to same conclusion.

## Files

- **`REPORT.md`** - Full investigation report with methodology, results, and conclusions
- **`investigate_subcategory.py`** - Reproducible analysis script (8 tests)
- **`encode_data.py`** - Data encoding utilities for statistical testing
- **`artifacts/`** - Generated outputs (CSV/JSON, gitignored, regenerate locally)

## Quick Run

```bash
# From TOODLE root directory
cd exploration/subcategory_independence

# Run investigation (requires data_int_encoded.pkl)
python investigate_subcategory.py --data <path_to_data> --output-dir artifacts

# View results
ls artifacts/
# Expected outputs:
#   - uniformity.csv
#   - conditional_independence.csv
#   - mutual_information.csv
#   - joint_lookup_bound.csv
#   - nested_cv.csv
#   - gradient_boosting.csv
#   - permutation_test.json
#   - power_check.json
#   - summary.json
```

## Test Battery (8 Independent Tests)

1. **Uniformity** - Chi-squared against uniform distribution → All pass (p > 0.05)
2. **Conditional Independence** - Fisher's method across features → All pass (p > 0.05)
3. **Mutual Information** - NMI < 0.01 threshold → All pass (max NMI = 0.0047)
4. **Joint-Lookup Upper Bound** - Cross-validated lookup table → 0.1997 accuracy (≈ 0.20)
5. **Nested CV** - Model selection with nested CV → Max 0.2037 accuracy (< 0.21)
6. **GradientBoosting** - Strong ensemble baseline → 0.1875 accuracy (< 0.20)
7. **Permutation Test** - Can ML beat shuffled labels? → No (p = 1.0)
8. **Power Check** - Can method detect injected signal? → Yes (100% power at 0.28)

## Conclusion

Subcategory contains **zero predictive signal**. Models should:
- Predict category only (5 classes)
- Not attempt subcategory prediction (proven unpredictable)

## Impact on TOODLE

- **Scoped out:** Subcategory prediction (data limitation, not model limitation)
- **Scoped in:** Category prediction (>85% F1 achievable on clean data)
- **Clean data approach:** Extract ~110 deterministic subject→category templates

## References

- **Full Report:** `REPORT.md` (detailed methodology and results)
- **Stage 1 Model Docs:** `../../docs/STAGE1_MODEL.md` (integration with pipeline)
- **Stage 1 Architecture:** `../../docs/STAGE1_ARCHITECTURE.md` (design decisions)
