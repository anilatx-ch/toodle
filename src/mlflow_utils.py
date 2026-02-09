"""Best-effort MLflow helpers with local fallback tracking."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping

from src import config

try:  # pragma: no cover - runtime dependency
    import mlflow as _mlflow
except Exception:  # pragma: no cover - runtime dependency
    _mlflow = None


_enabled = _mlflow is not None
_configured_uri: str | None = None
_fallback_announced = False
_disabled_announced = False


def _tracking_uris() -> list[str]:
    """Get list of tracking URIs to try (primary and fallback)."""
    primary = config.MLFLOW_TRACKING_URI.strip()
    fallback = config.MLFLOW_FALLBACK_TRACKING_URI.strip()
    if fallback and fallback != primary:
        return [primary, fallback]
    return [primary]


def _disable(exc: Exception) -> None:
    """Disable MLflow logging after an error."""
    global _enabled, _disabled_announced
    _enabled = False
    if not _disabled_announced:
        print(f"MLflow logging disabled: {exc}")
        _disabled_announced = True


def _configure() -> bool:
    """Configure MLflow tracking URI and experiment."""
    global _configured_uri, _fallback_announced

    if not _enabled or _mlflow is None:
        return False
    if _configured_uri is not None:
        return True

    last_error = None
    uris = _tracking_uris()
    for idx, uri in enumerate(uris):
        try:
            _mlflow.set_tracking_uri(uri)
            _mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
            _configured_uri = uri
            if idx > 0 and not _fallback_announced:
                print(f"MLflow fallback tracking URI: {uri}")
                _fallback_announced = True
            return True
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        _disable(last_error)
    return False


def start_run(run_name: str):
    """Start an MLflow run (or return nullcontext if MLflow unavailable)."""
    if not _configure() or _mlflow is None:
        return nullcontext()
    try:
        active = _mlflow.active_run()
        return _mlflow.start_run(run_name=run_name, nested=active is not None)
    except Exception as exc:
        _disable(exc)
        return nullcontext()


def log_metric(key: str, value: float, step: int | None = None) -> None:
    """Log a single metric to MLflow."""
    if not _configure() or _mlflow is None:
        return
    if _mlflow.active_run() is None:
        return
    try:
        if step is None:
            _mlflow.log_metric(key, value)
        else:
            _mlflow.log_metric(key, value, step=step)
    except Exception as exc:
        _disable(exc)


def log_metrics(metrics: Mapping[str, float]) -> None:
    """Log multiple metrics to MLflow."""
    if not _configure() or _mlflow is None:
        return
    if _mlflow.active_run() is None:
        return
    try:
        _mlflow.log_metrics(dict(metrics))
    except Exception as exc:
        _disable(exc)


def log_params(params: Mapping[str, object]) -> None:
    """Log parameters to MLflow."""
    if not _configure() or _mlflow is None:
        return
    if _mlflow.active_run() is None:
        return
    try:
        _mlflow.log_params(dict(params))
    except Exception as exc:
        _disable(exc)


def log_tags(tags: Mapping[str, object]) -> None:
    """Log tags to MLflow."""
    if not _configure() or _mlflow is None:
        return
    if _mlflow.active_run() is None:
        return
    try:
        _mlflow.set_tags({k: str(v) for k, v in dict(tags).items()})
    except Exception as exc:
        _disable(exc)


def log_artifact(path: str) -> None:
    """Log an artifact file to MLflow."""
    if not _configure() or _mlflow is None:
        return
    if _mlflow.active_run() is None:
        return
    try:
        _mlflow.log_artifact(path)
    except Exception as exc:
        _disable(exc)
