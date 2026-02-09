"""FastAPI prediction service for multi-classifier support ticket system."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from scipy import sparse

from src import config
from src.anomaly.detector import AnomalyDetector
from src.api.schemas import (
    ErrorResponse,
    FeedbackSentimentRequest,
    FeedbackSentimentResponse,
    MultiClassifierPredictionResponse,
    TicketInputMultiClassifier,
)
from src.features.pipeline import FeaturePipeline
from src.features.preprocessing import detect_leakage_warning
from src.models.bert_model import BertClassifier
from src.models.catboost_model import CatBoostTicketClassifier
from src.models.xgboost_model import XGBoostTicketClassifier

logger = logging.getLogger("ticket-api")
logging.basicConfig(level=logging.INFO)


def _error_response(
    status_code: int, ticket_id: Optional[str], error: str, details: str
) -> JSONResponse:
    payload = ErrorResponse(
        ticket_id=ticket_id,
        error=error,
        fallback="manual_triage",
        details=details,
    ).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


def _distribution_from_proba(values: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    probs = np.asarray(values, dtype=np.float32)
    if probs.ndim == 2:
        probs = probs[0]
    probs = probs.reshape(-1)

    if len(labels) != probs.shape[0]:
        labels = [str(i) for i in range(probs.shape[0])]

    return {str(labels[i]): float(probs[i]) for i in range(probs.shape[0])}


def _normalize_distribution(distribution: Mapping[str, float], labels: Sequence[str]) -> dict[str, float]:
    return {str(label): float(distribution.get(str(label), 0.0)) for label in labels}


def _pick_label(
    distribution: Mapping[str, float], candidate_labels: Optional[Sequence[str]] = None
) -> tuple[str, float]:
    if candidate_labels:
        search_space = {str(label): float(distribution.get(str(label), 0.0)) for label in candidate_labels}
    else:
        search_space = {str(label): float(value) for label, value in distribution.items()}

    if not search_space:
        raise ValueError("Empty probability distribution")

    label = max(search_space, key=lambda name: (search_space[name], name))
    return label, float(search_space[label])


class ModelManager:
    def __init__(self):
        self.feature_pipelines: dict[str, FeaturePipeline] = {}
        self.catboost_models: dict[str, CatBoostTicketClassifier] = {}
        self.xgboost_models: dict[str, XGBoostTicketClassifier] = {}
        self.bert_models: dict[str, BertClassifier] = {}
        self.anomaly_detector: Optional[AnomalyDetector] = None

    def _attach_labels(self, classifier: str, model: object) -> None:
        pipeline = self.feature_pipelines.get(classifier)
        if pipeline is None:
            return
        if hasattr(model, "label_classes"):
            model.label_classes = pipeline.label_encoder.classes_.tolist()

    def load_models(self) -> None:
        self.feature_pipelines = {}
        self.catboost_models = {}
        self.xgboost_models = {}
        self.bert_models = {}

        for classifier, pipeline_path in config.FEATURE_PIPELINE_PATHS.items():
            try:
                self.feature_pipelines[classifier] = FeaturePipeline.load(pipeline_path)
                logger.info("Feature pipeline loaded classifier=%s path=%s", classifier, pipeline_path)
            except Exception as exc:
                logger.warning(
                    "Feature pipeline unavailable classifier=%s path=%s err=%s",
                    classifier,
                    pipeline_path,
                    exc,
                )

        for classifier, model_path in config.CATBOOST_MODEL_PATHS.items():
            try:
                model = CatBoostTicketClassifier.load(model_path)
                self._attach_labels(classifier, model)
                self.catboost_models[classifier] = model
                logger.info("CatBoost loaded classifier=%s path=%s", classifier, model_path)
            except Exception as exc:
                logger.warning(
                    "CatBoost unavailable classifier=%s path=%s err=%s", classifier, model_path, exc
                )

        for classifier, model_path in config.XGBOOST_MODEL_PATHS.items():
            try:
                model = XGBoostTicketClassifier.load(model_path)
                self._attach_labels(classifier, model)
                self.xgboost_models[classifier] = model
                logger.info("XGBoost loaded classifier=%s path=%s", classifier, model_path)
            except Exception as exc:
                logger.warning(
                    "XGBoost unavailable classifier=%s path=%s err=%s", classifier, model_path, exc
                )

        for classifier, model_dir in config.BERT_MODEL_DIRS.items():
            try:
                model = BertClassifier.load(model_dir)
                self._attach_labels(classifier, model)
                self.bert_models[classifier] = model
                logger.info("BERT loaded classifier=%s path=%s", classifier, model_dir)
            except Exception as exc:
                logger.warning("BERT unavailable classifier=%s path=%s err=%s", classifier, model_dir, exc)

        try:
            self.anomaly_detector = AnomalyDetector.from_path(config.ANOMALY_BASELINE_PATH)
            if self.anomaly_detector is not None:
                logger.info("Anomaly detector loaded path=%s", config.ANOMALY_BASELINE_PATH)
        except Exception as exc:
            logger.warning("Anomaly detector unavailable path=%s err=%s", config.ANOMALY_BASELINE_PATH, exc)

    def missing_components(self, backend: str) -> list[str]:
        """Check if required components for category prediction are available."""
        missing: list[str] = []

        if "category" not in self.feature_pipelines:
            missing.append("feature_pipeline:category")

        if backend == "catboost" and "category" not in self.catboost_models:
            missing.append("catboost_model:category")
        elif backend == "xgboost" and "category" not in self.xgboost_models:
            missing.append("xgboost_model:category")
        elif backend == "bert" and "category" not in self.bert_models:
            missing.append("bert_model:category")

        return missing

    def backend_ready(self, backend: str) -> bool:
        return len(self.missing_components(backend)) == 0


def _predict_distribution(
    manager: ModelManager,
    backend: str,
    classifier: str,
    ticket_df: pd.DataFrame,
) -> dict[str, float]:
    pipeline = manager.feature_pipelines.get(classifier)
    if pipeline is None:
        raise RuntimeError(f"Missing feature pipeline for classifier={classifier}")

    transformed = pipeline.transform(ticket_df)

    if backend in config.TRADML_BACKENDS:
        stacked = sparse.hstack(
            [transformed["tfidf"], sparse.csr_matrix(transformed["tabular"])],
            format="csr",
        )

    if backend == "catboost":
        model = manager.catboost_models.get(classifier)
        if model is None:
            raise RuntimeError(f"Missing CatBoost model for classifier={classifier}")
        labels = model.label_classes or pipeline.label_encoder.classes_.tolist()
        proba = model.predict_proba(stacked)
        return _distribution_from_proba(proba, labels)

    if backend == "xgboost":
        model = manager.xgboost_models.get(classifier)
        if model is None:
            raise RuntimeError(f"Missing XGBoost model for classifier={classifier}")
        labels = model.label_classes or pipeline.label_encoder.classes_.tolist()
        proba = model.predict_proba(stacked)
        return _distribution_from_proba(proba, labels)

    if backend == "bert":
        model = manager.bert_models.get(classifier)
        if model is None:
            raise RuntimeError(f"Missing BERT model for classifier={classifier}")
        labels = model.label_classes or pipeline.label_encoder.classes_.tolist()
        proba = model.predict_proba(transformed["text_bert"], transformed["tabular"])
        return _distribution_from_proba(proba, labels)

    raise ValueError(f"Unsupported backend={backend!r}")


model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    return model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for model loading."""
    model_manager.load_models()
    yield


app = FastAPI(
    title="Ticket Classifier API",
    version=config.API_CONTRACT_VERSION,
    lifespan=lifespan,
)

from src.api.search import router as search_router  # noqa: E402

app.include_router(search_router)


@app.get("/health")
def health(manager: ModelManager = Depends(get_model_manager)) -> dict:
    backend = config.SERVING_BACKEND

    from src.api.search import get_search_engine

    search_engine = get_search_engine()
    search_ready = search_engine.is_ready

    return {
        "status": "ok",
        "contract_version": config.API_CONTRACT_VERSION,
        "serving_backend": backend,
        "backend_ready": manager.backend_ready(backend),
        "missing_components": manager.missing_components(backend),
        "anomaly_detector_ready": manager.anomaly_detector is not None,
        "search_index_ready": search_ready,
    }


@app.post(
    "/predict",
    response_model=MultiClassifierPredictionResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict(
    ticket: TicketInputMultiClassifier,
    manager: ModelManager = Depends(get_model_manager),
):
    backend = config.SERVING_BACKEND

    if not manager.backend_ready(backend):
        details = ", ".join(manager.missing_components(backend))
        if details:
            details = f"{backend} backend not ready: {details}"
        else:
            details = f"{backend} backend not ready"
        return _error_response(503, ticket.ticket_id, "model_not_available", details)

    start = time.perf_counter()
    raw_ticket = ticket.model_dump()
    ticket_df = pd.DataFrame([raw_ticket])
    warning = detect_leakage_warning(raw_ticket)

    try:
        category_distribution = _predict_distribution(manager, backend, "category", ticket_df)
        category_distribution = _normalize_distribution(
            category_distribution, config.CLASSIFIER_LABELS["category"]
        )
        predicted_category, category_confidence = _pick_label(
            category_distribution, config.CLASSIFIER_LABELS["category"]
        )

        # Priority and sentiment are placeholders
        predicted_priority = "medium"
        priority_confidence = None
        priority_distribution = {
            label: (1.0 if label == predicted_priority else 0.0)
            for label in config.CLASSIFIER_LABELS["priority"]
        }

        predicted_sentiment = "neutral"
        sentiment_confidence = None
        sentiment_distribution = {
            label: (1.0 if label == predicted_sentiment else 0.0) for label in config.SENTIMENT_CLASSES
        }

    except Exception as exc:
        return _error_response(422, ticket.ticket_id, "preprocessing_failed", str(exc))

    warning_flags = []
    if warning:
        warning_flags.append(warning)
    if category_confidence < config.CONFIDENCE_THRESHOLD:
        warning_flags.append("low_confidence")

    if manager.anomaly_detector is not None:
        anomaly_result = manager.anomaly_detector.analyze_prediction(
            predicted_category=predicted_category,
            category_confidence=category_confidence,
        )
        if anomaly_result.is_confidence_anomaly:
            warning_flags.append("confidence_anomaly")
            logger.info(
                "Confidence anomaly detected ticket_id=%s category=%s confidence=%.3f zscore=%.2f",
                ticket.ticket_id,
                predicted_category,
                category_confidence,
                anomaly_result.confidence_zscore,
            )

    warning_flags.extend(["priority_placeholder", "sentiment_placeholder"])
    response_warning = ",".join(dict.fromkeys(warning_flags)) if warning_flags else None

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response = MultiClassifierPredictionResponse(
        ticket_id=ticket.ticket_id,
        predicted_category=predicted_category,
        predicted_priority=predicted_priority,
        predicted_sentiment=predicted_sentiment,
        category_confidence=category_confidence,
        priority_confidence=priority_confidence,
        sentiment_confidence=sentiment_confidence,
        category_probabilities=category_distribution,
        priority_probabilities=priority_distribution,
        sentiment_probabilities=sentiment_distribution,
        warning=response_warning,
        model_used=backend,
        inference_time_ms=elapsed_ms,
    )

    logger.info(
        "prediction ticket_id=%s model=%s category=%s warning=%s ms=%.2f",
        ticket.ticket_id,
        backend,
        predicted_category,
        response.warning,
        elapsed_ms,
    )
    return response


@app.post(
    "/analyze-feedback",
    response_model=FeedbackSentimentResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def analyze_feedback(
    request: FeedbackSentimentRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """Analyze sentiment of customer feedback text after ticket resolution."""
    backend = config.SERVING_BACKEND

    if not manager.backend_ready(backend):
        details = ", ".join(manager.missing_components(backend))
        if details:
            details = f"{backend} backend not ready: {details}"
        else:
            details = f"{backend} backend not ready"
        return _error_response(503, request.ticket_id, "model_not_available", details)

    start = time.perf_counter()

    if not request.feedback_text or request.feedback_text.strip() == "":
        return FeedbackSentimentResponse(
            ticket_id=request.ticket_id,
            predicted_sentiment="neutral",
            sentiment_confidence=0.0,
            sentiment_probabilities={cls: 0.0 for cls in config.SENTIMENT_CLASSES},
            model_used=f"{backend}_sentiment",
            inference_time_ms=0.0,
            warning="empty_feedback",
        )

    df = pd.DataFrame([{"feedback_text": request.feedback_text}])

    try:
        distribution = _predict_distribution(manager, backend, "sentiment", df)
        distribution = _normalize_distribution(distribution, config.SENTIMENT_CLASSES)
        predicted_sentiment, sentiment_confidence = _pick_label(distribution, config.SENTIMENT_CLASSES)
    except Exception as exc:
        return _error_response(422, request.ticket_id, "sentiment_analysis_failed", str(exc))

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    response = FeedbackSentimentResponse(
        ticket_id=request.ticket_id,
        predicted_sentiment=predicted_sentiment,
        sentiment_confidence=sentiment_confidence,
        sentiment_probabilities=distribution,
        model_used=f"{backend}_sentiment",
        inference_time_ms=elapsed_ms,
    )

    logger.info(
        "sentiment_analysis ticket_id=%s sentiment=%s confidence=%.3f ms=%.2f",
        request.ticket_id,
        predicted_sentiment,
        sentiment_confidence,
        elapsed_ms,
    )
    return response
