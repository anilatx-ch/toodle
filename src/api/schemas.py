"""API request and response schemas."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class TicketInput(BaseModel):
    """Legacy ticket input schema (for backward compatibility in preprocessing)."""

    ticket_id: str
    subject: str
    description: str
    error_logs: Optional[str] = None
    stack_trace: Optional[str] = None
    product: str
    product_module: str
    channel: str
    customer_tier: str
    environment: str
    language: str
    region: str
    priority: str
    severity: str
    account_age_days: int
    account_monthly_value: float
    previous_tickets: int
    product_version_age_days: int
    attachments_count: int
    created_at: str = Field(description="ISO-8601 timestamp")


class TicketInputMultiClassifier(BaseModel):
    ticket_id: str
    subject: str
    description: str
    error_logs: Optional[str] = None
    stack_trace: Optional[str] = None
    product: str
    product_module: str
    channel: str
    customer_tier: str
    environment: str
    language: str
    region: str
    severity: str
    account_age_days: int
    account_monthly_value: float
    previous_tickets: int
    product_version_age_days: int
    attachments_count: int
    created_at: str = Field(description="ISO-8601 timestamp")


class MultiClassifierPredictionResponse(BaseModel):
    ticket_id: str
    predicted_category: str
    predicted_priority: str
    predicted_sentiment: str
    category_confidence: float
    priority_confidence: Optional[float] = Field(
        None, description="Null for placeholder priority predictions"
    )
    sentiment_confidence: Optional[float] = Field(
        None, description="Null for placeholder sentiment predictions"
    )
    category_probabilities: Dict[str, float]
    priority_probabilities: Dict[str, float]
    sentiment_probabilities: Dict[str, float]
    warning: Optional[str] = None
    model_used: str
    inference_time_ms: float


class ErrorResponse(BaseModel):
    ticket_id: Optional[str] = None
    error: str
    fallback: str
    details: Optional[str] = None


class FeedbackSentimentRequest(BaseModel):
    ticket_id: str
    feedback_text: str


class FeedbackSentimentResponse(BaseModel):
    ticket_id: str
    predicted_sentiment: str
    sentiment_confidence: float
    sentiment_probabilities: Dict[str, float]
    model_used: str
    inference_time_ms: float
    warning: Optional[str] = None
