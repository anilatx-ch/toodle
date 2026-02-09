"""Resolution corpus loading and serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.data.loader import find_raw_json
from src.features.entities import ERROR_RE


@dataclass(frozen=True)
class ResolutionDocument:
    """Searchable resolution metadata from historical tickets."""

    ticket_id: str
    resolution_text: str
    resolution_code: str
    category: str
    product: str
    subject: str
    description: str
    error_codes: list[str]
    template_used: str | None

    def source_text(self) -> str:
        """Text used for semantic indexing."""
        return f"{self.subject} {self.description}".strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "ticket_id": self.ticket_id,
            "resolution": self.resolution_text,
            "resolution_code": self.resolution_code,
            "category": self.category,
            "product": self.product,
            "subject": self.subject,
            "description": self.description,
            "error_codes": self.error_codes,
            "template_used": self.template_used,
        }

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "ResolutionDocument":
        resolution_text = str(row.get("resolution") or row.get("resolution_text") or "").strip()
        description = str(row.get("description") or "")
        error_codes_value = row.get("error_codes")
        if isinstance(error_codes_value, list):
            error_codes = [str(code) for code in error_codes_value if str(code).strip()]
        else:
            error_logs = str(row.get("error_logs") or "")
            error_codes = ERROR_RE.findall(f"{description} {error_logs}")

        return cls(
            ticket_id=str(row.get("ticket_id") or ""),
            resolution_text=resolution_text,
            resolution_code=str(row.get("resolution_code") or ""),
            category=str(row.get("category") or ""),
            product=str(row.get("product") or ""),
            subject=str(row.get("subject") or ""),
            description=description,
            error_codes=error_codes,
            template_used=(
                str(row.get("template_used"))
                if row.get("template_used") is not None
                else (
                    str(row.get("resolution_template_used"))
                    if row.get("resolution_template_used") is not None
                    else None
                )
            ),
        )


def load_corpus(
    json_path: Path | None = None,
    limit: int | None = None,
    sample_rate: float | None = None,
) -> list[ResolutionDocument]:
    """Load documents with non-empty resolutions.

    Args:
        json_path: Path to JSON file with ticket data
        limit: Maximum number of documents to load (absolute cap)
        sample_rate: Fraction of total documents to load (0.0-1.0). Applied before limit.
    """
    path = json_path or find_raw_json()

    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)

    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list at {path}")

    # Calculate sample cap if sample_rate is provided
    sample_cap = None
    if sample_rate is not None and 0.0 < sample_rate < 1.0:
        sample_cap = max(1, int(len(rows) * sample_rate))

    # Use explicit limit, or sample_cap, or config default
    cap = limit if limit is not None else (sample_cap or config.RETRIEVAL_SMOKE_LIMIT)

    documents: list[ResolutionDocument] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc = ResolutionDocument.from_dict(row)
        if not doc.ticket_id or not doc.resolution_text:
            continue
        documents.append(doc)
        if cap is not None and len(documents) >= cap:
            break

    return documents


def save_corpus(documents: list[ResolutionDocument], path: Path) -> None:
    """Persist corpus documents as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [doc.to_dict() for doc in documents]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

