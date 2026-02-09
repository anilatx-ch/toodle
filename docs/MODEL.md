# Existing content

| Model      | F1 Score | Accuracy | Latency (p50) | Latency (p95) | Size (MB) |
|------------|----------|----------|---------------|---------------|-----------|
| CatBoost   | 0.4781 | 0.5294 | 0.27 | 0.29 | 0.01 |
| XGBoost    | 0.5462 | 0.6471 | 0.35 | 0.43 | 0.01 |
| DistilBERT | 0.0182 | 0.1000 | 107.78 | 123.32 | 759.80 |

### Recommendation

**Recommendation:**

Use **XGBoost** as the primary serving model. It achieves competitive F1 (0.5462) with significantly lower latency (~0.43ms p95) and simpler deployment (CPU-only, smaller model size).

**Trade-offs:**
- **Traditional ML (CatBoost/XGBoost):** Fast inference (~1-5ms), small models (<10MB), CPU-friendly, deterministic. Limited ability to generalize to novel phrasing.
- **Deep Learning (BERT):** Better semantic understanding, handles paraphrasing, higher accuracy ceiling. Requires GPU for fast inference, larger model size (250-300MB), higher latency (50-200ms).