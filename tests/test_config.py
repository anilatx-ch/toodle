from src import config


def test_smoke_test_flag_type():
    assert isinstance(config.SMOKE_TEST, bool)


def test_env_validation():
    assert config.ENV in {"dev", "test", "prod"}


def test_tradml_backends():
    assert "catboost" in config.TRADML_BACKENDS
    assert "xgboost" in config.TRADML_BACKENDS
    assert "lightgbm" not in config.TRADML_BACKENDS


def test_inference_backends():
    assert "catboost" in config.INFERENCE_BACKENDS
    assert "xgboost" in config.INFERENCE_BACKENDS
    assert "bert" in config.INFERENCE_BACKENDS
    assert "lightgbm" not in config.INFERENCE_BACKENDS


def test_multi_classifier_order():
    assert config.MULTI_CLASSIFIER_ORDER == ("category", "priority", "sentiment")


def test_category_classes():
    assert len(config.CATEGORY_CLASSES) == 5
    assert "Technical Issue" in config.CATEGORY_CLASSES


def test_no_subcategory_classes():
    assert not hasattr(config, "SUBCATEGORY_CLASSES")


def test_clean_training_path_exists():
    assert hasattr(config, "CLEAN_TRAINING_PARQUET_PATH")
    assert "clean_training" in str(config.CLEAN_TRAINING_PARQUET_PATH)


def test_paths_exist_or_creatable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", tmp_path / "embeddings")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "MLFLOW_LOCAL_DIR", tmp_path / "mlruns")
    monkeypatch.setattr(config, "DATA_INVESTIGATION_DIR", tmp_path / "diagnostics/data_investigation")
    monkeypatch.setattr(config, "ANOMALY_BASELINE_PATH", tmp_path / "diagnostics/anomaly/baseline.json")
    monkeypatch.setattr(config, "SENTIMENT_MODEL_DIR", tmp_path / "models/sentiment")

    bert_model_dirs = {
        k: tmp_path / "models" / f"bert_{k}_test"
        for k in config.BERT_MODEL_DIRS.keys()
    }
    monkeypatch.setattr(config, "BERT_MODEL_DIRS", bert_model_dirs)

    config.ensure_directories()

    assert config.RAW_DIR.exists()
    assert config.PROCESSED_DIR.exists()
    assert config.EMBEDDINGS_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert config.FIGURES_DIR.exists()
    assert config.MLFLOW_LOCAL_DIR.exists()
    assert config.DATA_INVESTIGATION_DIR.exists()
    assert config.ANOMALY_BASELINE_PATH.parent.exists()
    assert config.SENTIMENT_MODEL_DIR.exists()
    for bert_dir in bert_model_dirs.values():
        assert bert_dir.exists()
