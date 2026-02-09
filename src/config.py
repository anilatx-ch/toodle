"""Centralized project configuration."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RETRIEVAL_DIR = DATA_DIR / "retrieval"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
METRICS_ROOT_DIR = BASE_DIR / "metrics"
FIGURES_ROOT_DIR = BASE_DIR / "figures"
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
SCHEMAS_DIR = BASE_DIR / "schemas"

# Environment configuration
ENV = os.getenv("ENV", "dev").strip().lower() or "dev"
if ENV not in {"dev", "test", "prod"}:
    raise ValueError(f"Invalid ENV={ENV!r}. Use 'dev', 'test', or 'prod'")

IS_INFRA_DEV = ENV == "dev"

# Smoke test configuration
_smoke_override = os.getenv("SMOKE_TEST", "").strip().lower()
if _smoke_override == "true":
    SMOKE_TEST = True
elif _smoke_override == "false":
    SMOKE_TEST = False
else:
    SMOKE_TEST = ENV in {"dev", "test"}

_optuna_override = os.getenv("TRADML_ENABLE_OPTUNA")
if _optuna_override is None or _optuna_override.strip() == "":
    TRADML_ENABLE_OPTUNA = False
else:
    TRADML_ENABLE_OPTUNA = _optuna_override.strip().lower() == "true"


def _env_name(stem: str, suffix: str) -> str:
    return f"{stem}_{ENV}{suffix}"


METRICS_DIR = METRICS_ROOT_DIR / ENV
FIGURES_DIR = FIGURES_ROOT_DIR / ENV

if IS_INFRA_DEV:
    TRAIN_SAMPLES = 20
    VAL_SAMPLES = 10
    OPTUNA_TIMEOUT_S = 2
else:
    TRAIN_SAMPLES = 500 if SMOKE_TEST else None
    VAL_SAMPLES = 100 if SMOKE_TEST else None
    OPTUNA_TIMEOUT_S = 120 if SMOKE_TEST else 1800

# Per-classifier early stopping thresholds
CLASSIFIER_F1_THRESHOLD = float(os.getenv("CLASSIFIER_F1_THRESHOLD", "0.80"))
CLASSIFIER_TIMEOUT_SECONDS = int(os.getenv("CLASSIFIER_TIMEOUT_SECONDS", "1800"))

if IS_INFRA_DEV:
    CLASSIFIER_F1_THRESHOLD = 0.0
    CLASSIFIER_TIMEOUT_SECONDS = 300
elif SMOKE_TEST:
    CLASSIFIER_TIMEOUT_SECONDS = 600

# Data file paths
_RAW_JSON_BASE_CANDIDATES = [
    RAW_DIR / "tickets.json",
    BASE_DIR / "support_tickets.json",
]

if ENV == "test":
    RAW_JSON_CANDIDATES = [BASE_DIR / "support_tickets_mini.json"] + _RAW_JSON_BASE_CANDIDATES
else:
    RAW_JSON_CANDIDATES = _RAW_JSON_BASE_CANDIDATES

DUCKDB_PATH = PROCESSED_DIR / _env_name("tickets", ".duckdb")
UNSPLIT_PARQUET_PATH = PROCESSED_DIR / _env_name("tickets_featured_unsplit", ".parquet")
SPLIT_PARQUET_PATH = PROCESSED_DIR / _env_name("tickets_featured", ".parquet")
CLEAN_TRAINING_PARQUET_PATH = PROCESSED_DIR / _env_name("clean_training", ".parquet")
EMBEDDINGS_PATH = EMBEDDINGS_DIR / _env_name("embeddings", ".npy")
EMBEDDINGS_TICKET_IDS_PATH = EMBEDDINGS_DIR / _env_name("ticket_ids", ".npy")
EMBEDDINGS_METADATA_PATH = EMBEDDINGS_DIR / _env_name("metadata", ".json")

# Model backends
TRADML_BACKENDS = ("catboost", "xgboost")
DEFAULT_TRADML_BACKEND = os.getenv("DEFAULT_TRADML_BACKEND", "xgboost").strip().lower() or "xgboost"
if DEFAULT_TRADML_BACKEND not in TRADML_BACKENDS:
    raise ValueError(
        f"Invalid DEFAULT_TRADML_BACKEND={DEFAULT_TRADML_BACKEND!r}. "
        f"Expected one of: {TRADML_BACKENDS}"
    )

SERVING_BACKEND = os.getenv("SERVING_BACKEND", "xgboost").strip().lower() or "xgboost"
if SERVING_BACKEND not in ("xgboost", "catboost", "bert"):
    raise ValueError(
        f"Invalid SERVING_BACKEND={SERVING_BACKEND!r}. "
        f"Expected one of: xgboost, catboost, bert"
    )

# API configuration
API_CONTRACT_VERSION = "1.0.0"

INFERENCE_BACKENDS = ("catboost", "xgboost", "bert")
MULTI_CLASSIFIER_ORDER = ("category", "priority", "sentiment")

# Model paths
FEATURE_PIPELINE_PATHS = {
    "category": MODELS_DIR / _env_name("feature_pipeline_category", ".pkl"),
    "priority": MODELS_DIR / _env_name("feature_pipeline_priority", ".pkl"),
    "sentiment": MODELS_DIR / _env_name("feature_pipeline_sentiment", ".pkl"),
}

CATBOOST_MODEL_PATHS = {
    "category": MODELS_DIR / _env_name("catboost_category", ".cbm"),
    "priority": MODELS_DIR / _env_name("catboost_priority", ".cbm"),
    "sentiment": MODELS_DIR / _env_name("catboost_sentiment", ".cbm"),
}

XGBOOST_MODEL_PATHS = {
    "category": MODELS_DIR / _env_name("xgboost_category", ".json"),
    "priority": MODELS_DIR / _env_name("xgboost_priority", ".json"),
    "sentiment": MODELS_DIR / _env_name("xgboost_sentiment", ".json"),
}

BERT_MODEL_DIRS = {
    "category": MODELS_DIR / _env_name("bert_category", ""),
    "priority": MODELS_DIR / _env_name("bert_priority", ""),
    "sentiment": MODELS_DIR / _env_name("bert_sentiment", ""),
}

TFIDF_VECTORIZER_PATH = MODELS_DIR / _env_name("tfidf_vectorizer", ".pkl")
QUANTIZED_BERT_PATH = MODELS_DIR / _env_name("m3_quantized", ".tflite")

# Shortcuts for category (default) models
FEATURE_PIPELINE_PATH = FEATURE_PIPELINE_PATHS["category"]
CATBOOST_MODEL_PATH = CATBOOST_MODEL_PATHS["category"]
XGBOOST_MODEL_PATH = XGBOOST_MODEL_PATHS["category"]
BERT_MODEL_DIR = BERT_MODEL_DIRS["category"]

# Metrics and reports paths
CATBOOST_OPTUNA_SUMMARY_PATH = METRICS_DIR / "catboost_optuna_summary.json"
XGBOOST_OPTUNA_SUMMARY_PATH = METRICS_DIR / "xgboost_optuna_summary.json"
EVALUATION_SUMMARY_PATH = METRICS_DIR / "evaluation_summary.json"
MDEEPL_TRAINING_SUMMARY_PATH = METRICS_DIR / "mdeepl_training_summary.json"
QUANTIZATION_RESULTS_PATH = METRICS_DIR / "quantization_results.json"
LATENCY_METRICS_PATH = METRICS_DIR / "latency.json"
PER_CLASS_MTRADML_PATH = METRICS_DIR / "per_class_metrics_mtradml.csv"
PER_CLASS_MDEEPL_PATH = METRICS_DIR / "per_class_metrics_mdeepl.csv"
ERROR_ANALYSIS_PATH = METRICS_DIR / "error_analysis.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / _env_name("model_comparison", ".md")
EDA_REPORT_PATH = DIAGNOSTICS_DIR / _env_name("eda_report", ".html")
COLUMNS_STATISTICS_PATH = DIAGNOSTICS_DIR / "columns_statistics.json"
DATA_INVESTIGATION_DIR = DIAGNOSTICS_DIR / "data_investigation"
API_CONTRACT_PATH = SCHEMAS_DIR / _env_name("api_contract", ".json")

# Sentiment model paths
SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment"
SENTIMENT_TRAINING_SUMMARY_PATH = METRICS_DIR / "sentiment_training_summary.json"
SENTIMENT_COMPARISON_PATH = METRICS_DIR / "sentiment_comparison.json"

# Anomaly detection configuration
ANOMALY_CONFIDENCE_ZSCORE_THRESHOLD = 2.0
ANOMALY_JSD_THRESHOLD = 0.10
ANOMALY_MIN_BASELINE_SAMPLES = 30
ANOMALY_BASELINE_PATH = DIAGNOSTICS_DIR / "anomaly" / _env_name("baseline", ".json")
ANOMALY_REPORT_PATH = DIAGNOSTICS_DIR / "anomaly" / _env_name("report", ".json")

# Retrieval / RAG configuration
FAISS_INDEX_PATH = RETRIEVAL_DIR / _env_name("faiss_index", ".bin")
RETRIEVAL_CORPUS_PATH = RETRIEVAL_DIR / _env_name("corpus", ".json")
RETRIEVAL_ENTITY_INDEX_PATH = RETRIEVAL_DIR / _env_name("entity_index", ".json")
RETRIEVAL_SMOKE_LIMIT = 500 if SMOKE_TEST else None

SPLIT_SEED = int(os.getenv("SPLIT_SEED", "42"))

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = f"ticket-classification-{ENV}"
MLFLOW_LOCAL_DIR = BASE_DIR / "mlruns"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_LOCAL_DIR.resolve().as_uri())
MLFLOW_FALLBACK_TRACKING_URI = os.getenv("MLFLOW_FALLBACK_TRACKING_URI", MLFLOW_LOCAL_DIR.resolve().as_uri())

# Feature parameters
if IS_INFRA_DEV:
    TFIDF_MAX_FEATURES = 2
    CHI2_SELECT_K = 2
    ERROR_CODE_HASH_BUCKETS = 2
else:
    TFIDF_MAX_FEATURES = 10000
    CHI2_SELECT_K = 5000
    ERROR_CODE_HASH_BUCKETS = 16

CONFIDENCE_THRESHOLD = 0.5

# CatBoost params
CATBOOST_BASELINE_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}

if IS_INFRA_DEV:
    CATBOOST_BASELINE_PARAMS = {
        **CATBOOST_BASELINE_PARAMS,
        "iterations": 2,
        "depth": 2,
        "learning_rate": 0.2,
    }

if IS_INFRA_DEV:
    CATBOOST_SEARCH_SPACE = {
        "iterations": (2, 2),
        "depth": (2, 2),
        "learning_rate": (0.1, 0.2),
        "l2_leaf_reg": (1.0, 2.0),
        "random_strength": (0.0, 1.0),
        "bagging_temperature": (0.0, 1.0),
        "border_count": (2, 16),
    }
else:
    CATBOOST_SEARCH_SPACE = {
        "iterations": (300, 2000),
        "depth": (4, 10),
        "learning_rate": (0.01, 0.2),
        "l2_leaf_reg": (1.0, 20.0),
        "random_strength": (0.0, 5.0),
        "bagging_temperature": (0.0, 5.0),
        "border_count": (64, 254),
    }

# XGBoost params
XGBOOST_BASELINE_PARAMS = {
    "objective": "multi:softprob",
    "random_state": 42,
    "verbosity": 0,
}

if IS_INFRA_DEV:
    XGBOOST_BASELINE_PARAMS = {
        **XGBOOST_BASELINE_PARAMS,
        "n_estimators": 2,
        "max_depth": 1,
        "learning_rate": 0.2,
    }

if IS_INFRA_DEV:
    XGBOOST_SEARCH_SPACE = {
        "n_estimators": (20, 40),
        "max_depth": (2, 4),
        "learning_rate": (0.05, 0.2),
        "subsample": (0.8, 1.0),
        "colsample_bytree": (0.8, 1.0),
        "gamma": (0.0, 1.0),
        "reg_alpha": (0.0, 2.0),
        "reg_lambda": (0.0, 2.0),
    }
else:
    XGBOOST_SEARCH_SPACE = {
        "n_estimators": (300, 2000),
        "max_depth": (4, 12),
        "learning_rate": (0.01, 0.2),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "gamma": (0.0, 5.0),
        "reg_alpha": (0.0, 10.0),
        "reg_lambda": (0.0, 10.0),
    }

# DistilBERT params (optimized for clean ~110-sample balanced dataset)
BERT_PRESET = os.getenv("BERT_PRESET", "distil_bert_base_en_uncased")

if IS_INFRA_DEV:
    BERT_MAX_LEN = 32
    BERT_BATCH_SIZE = 2
    BERT_TABULAR_DENSE = 2
    BERT_HIDDEN_1 = 2
    BERT_HIDDEN_2 = 2
    BERT_DROPOUT = 0.1
    BERT_EARLY_STOP_PATIENCE = 1
    BERT_EPOCHS = 2
elif SMOKE_TEST:
    BERT_MAX_LEN = 128
    BERT_BATCH_SIZE = 4
    BERT_TABULAR_DENSE = 16
    BERT_HIDDEN_1 = 32
    BERT_HIDDEN_2 = 16
    BERT_DROPOUT = 0.2
    BERT_EARLY_STOP_PATIENCE = 2
    BERT_EPOCHS = 5
else:
    # Optimized for clean 110-sample dataset (targets F1 > 0.9)
    BERT_MAX_LEN = 256
    BERT_BATCH_SIZE = 16
    BERT_TABULAR_DENSE = 32
    BERT_HIDDEN_1 = 128
    BERT_HIDDEN_2 = 64
    BERT_DROPOUT = 0.2
    BERT_EARLY_STOP_PATIENCE = 2
    BERT_EPOCHS = 4

BERT_LR = 2e-5
BERT_WEIGHT_DECAY = 0.01

# Label classes
CATEGORY_CLASSES = sorted([
    "Account Management",
    "Data Issue",
    "Feature Request",
    "Security",
    "Technical Issue",
])

PRIORITY_CLASSES = ["low", "medium", "high", "critical"]
SENTIMENT_CLASSES = ["angry", "confused", "frustrated", "grateful", "neutral", "satisfied"]

# Field definitions
CREATION_TIME_CATEGORICAL_FIELDS = [
    "product",
    "product_module",
    "channel",
    "customer_tier",
    "environment",
    "language",
    "region",
]

CREATION_TIME_ORDINAL_FIELDS = [
    "severity_ordinal",
]

CREATION_TIME_NUMERIC_FIELDS = [
    "account_age_days",
    "account_monthly_value",
    "previous_tickets",
    "product_version_age_days",
    "attachments_count",
    "ticket_text_length",
]

TEXT_COLUMNS = ["subject", "description", "error_logs", "stack_trace"]

# Classifier configuration
CLASSIFIER_TARGET_COLUMN = {
    "category": "category",
    "priority": "priority",
    "sentiment": "customer_sentiment",
}

CLASSIFIER_LABELS = {
    "category": CATEGORY_CLASSES,
    "priority": PRIORITY_CLASSES,
    "sentiment": SENTIMENT_CLASSES,
}

# Feature configurations
CLASSIFIER_FEATURE_CONFIGS = {
    "category": {
        "text_tfidf_fields": ["subject", "description"],
        "text_bert_fields": ["subject", "description"],
        "categorical_fields": [],
        "ordinal_fields": [],
        "scaled_numeric_fields": [],
        "passthrough_numeric_fields": [],
        "include_entities": False,
        "include_temporal": False,
        "requires_category_hint": False,
    },
    "priority": {
        "text_tfidf_fields": TEXT_COLUMNS,
        "text_bert_fields": TEXT_COLUMNS,
        "categorical_fields": [
            "product",
            "product_module",
            "channel",
            "customer_tier",
            "environment",
            "language",
            "region",
        ],
        "ordinal_fields": ["severity_ordinal"],
        "scaled_numeric_fields": [
            "account_age_days",
            "account_monthly_value",
            "product_version_age_days",
            "ticket_text_length",
        ],
        "passthrough_numeric_fields": ["previous_tickets", "attachments_count"],
        "include_entities": True,
        "include_temporal": True,
        "requires_category_hint": False,
    },
    "sentiment": {
        "text_tfidf_fields": ["feedback_text"],
        "text_bert_fields": ["feedback_text"],
        "categorical_fields": [],
        "ordinal_fields": [],
        "scaled_numeric_fields": [],
        "passthrough_numeric_fields": [],
        "include_entities": False,
        "include_temporal": False,
        "requires_category_hint": False,
    },
}


def ensure_directories() -> None:
    """Create runtime directories if missing."""
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        EMBEDDINGS_DIR,
        RETRIEVAL_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        DIAGNOSTICS_DIR,
        DATA_INVESTIGATION_DIR,
        SCHEMAS_DIR,
        MLFLOW_LOCAL_DIR,
        ANOMALY_BASELINE_PATH.parent,
        SENTIMENT_MODEL_DIR,
        *BERT_MODEL_DIRS.values(),
    ]:
        path.mkdir(parents=True, exist_ok=True)
