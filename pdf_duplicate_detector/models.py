"""Data model classes for the PDF Duplicate Detector system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from scipy.sparse import spmatrix


@dataclass
class DocumentRecord:
    """A stored entry representing an ingested PDF document.

    Attributes:
        doc_id: Unique identifier (UUID4) for the document.
        filename: Original PDF filename.
        text: Extracted plain text (stored for TF-IDF refitting).
        vector: TF-IDF sparse vector, or None before the first fit.
        ingested_at: UTC timestamp of ingestion.
    """

    doc_id: str
    filename: str
    text: str
    vector: spmatrix | None
    ingested_at: datetime


@dataclass
class ComparisonResult:
    """Result of comparing a query document against the vector store.

    Attributes:
        is_duplicate: True if the highest score meets or exceeds the 0.98 threshold.
        highest_score: Highest cosine similarity score (0.0–1.0).
        matching_doc_id: ID of the best match, or None if no duplicate found.
        store_empty: True if the vector store was empty during comparison.
    """

    is_duplicate: bool
    highest_score: float
    matching_doc_id: str | None
    store_empty: bool


@dataclass
class IngestionResult:
    """Result of a successful document ingestion.

    Attributes:
        doc_id: Assigned document identifier.
        message: Confirmation or status message.
    """

    doc_id: str
    message: str
