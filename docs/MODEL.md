# Existing content

| Model      | F1 Score | Accuracy | Latency (p50) | Latency (p95) | Size (MB) |
|------------|----------|----------|---------------|---------------|-----------|
| CatBoost   | 1.0000 | 1.0000 | 0.58 | 0.64 | 2.93 |
| XGBoost    | 0.9395 | 0.9412 | 0.36 | 0.39 | 0.21 |
| DistilBERT | 1.0000 | 1.0000 | 166.66 | 172.22 | 761.00 |

### Recommendation

**Recommendation:**

Use **CatBoost** as the primary serving model. It achieves competitive F1 (1.0000) with significantly lower latency (~0.64ms p95) and simpler deployment (CPU-only, smaller model size).

**DistilBERT** can be used as an alternative for quality-focused scenarios where GPU inference is available and higher latency is acceptable. Consider using it for async batch processing or as a cross-validation model.

**Trade-offs:**
- **Traditional ML (CatBoost/XGBoost):** Fast inference (~1-5ms), small models (<10MB), CPU-friendly, deterministic. Limited ability to generalize to novel phrasing.
- **Deep Learning (BERT):** Better semantic understanding, handles paraphrasing, higher accuracy ceiling. Requires GPU for fast inference, larger model size (250-300MB), higher latency (50-200ms).