"""Tests for traditional ML training orchestrator."""

from src.training import run_training as orchestrator


def test_run_training_all_models(monkeypatch):
    calls: list[tuple[str, bool]] = []

    def fake_catboost(*, enable_optuna: bool):
        calls.append(("catboost", enable_optuna))
        return {"metrics": {"f1_weighted": 0.91}}

    def fake_xgboost(*, enable_optuna: bool):
        calls.append(("xgboost", enable_optuna))
        return {"metrics": {"f1_weighted": 0.89}}

    monkeypatch.setattr(orchestrator, "train_catboost_model", fake_catboost)
    monkeypatch.setattr(orchestrator, "train_xgboost_model", fake_xgboost)

    exit_code = orchestrator.main(["--model", "all", "--optuna"])

    assert exit_code == 0
    assert calls == [
        ("catboost", True),
        ("xgboost", True),
    ]


def test_run_training_stops_on_first_error(monkeypatch):
    calls: list[str] = []

    def fail_catboost(*, enable_optuna: bool):
        calls.append("catboost")
        raise RuntimeError("failed")

    def fake_xgboost(*, enable_optuna: bool):
        calls.append("xgboost")
        return {"metrics": {"f1_weighted": 0.88}}

    monkeypatch.setattr(orchestrator, "train_catboost_model", fail_catboost)
    monkeypatch.setattr(orchestrator, "train_xgboost_model", fake_xgboost)

    exit_code = orchestrator.main(["--model", "all"])

    assert exit_code == 1
    assert calls == ["catboost"]


def test_run_training_continue_on_error(monkeypatch):
    calls: list[str] = []

    def fail_catboost(*, enable_optuna: bool):
        calls.append("catboost")
        raise RuntimeError("failed")

    def fake_xgboost(*, enable_optuna: bool):
        calls.append("xgboost")
        return {"metrics": {"f1_weighted": 0.88}}

    monkeypatch.setattr(orchestrator, "train_catboost_model", fail_catboost)
    monkeypatch.setattr(orchestrator, "train_xgboost_model", fake_xgboost)

    exit_code = orchestrator.main(["--model", "all", "--continue-on-error"])

    assert exit_code == 1
    assert calls == ["catboost", "xgboost"]
