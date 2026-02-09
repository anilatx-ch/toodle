# Existing content

| Model      | F1 Score | Accuracy | Latency (p50) | Latency (p95) | Size (MB) |
|-------|----------|----------|---------------|---------------|-----------|
| CatBoost   | 0.8800 | 0.8900 | 1.20 | 2.50 | 0.10 |
| XGBoost    | 0.8600 | 0.8700 | 1.50 | 3.00 | 0.05 |
| DistilBERT | 0.9200 | 0.9300 | 45.00 | 65.00 | 250.00 |

### Recommendation

**Recommendation:**

Use **DistilBERT** as the primary model due to superior F1 score (0.9200). The higher latency and model size are justified by the accuracy improvement. Ensure GPU inference is available for production deployment.

Traditional ML models (CatBoost/XGBoost) can serve as fast fallback options for latency-sensitive requests or CPU-only environments.

**Trade-offs:**
- **Traditional ML (CatBoost/XGBoost):** Fast inference (~1-5ms), small models (<10MB), CPU-friendly, deterministic. Limited ability to generalize to novel phrasing.
- **Deep Learning (BERT):** Better semantic understanding, handles paraphrasing, higher accuracy ceiling. Requires GPU for fast inference, larger model size (250-300MB), higher latency (50-200ms).