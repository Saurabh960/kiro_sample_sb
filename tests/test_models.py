"""Unit tests for the data model classes."""

import dataclasses
from datetime import datetime

import pytest
from scipy.sparse import csr_matrix

from pdf_duplicate_detector.models import (
    ComparisonResult,
    DocumentRecord,
    IngestionResult,
)


class TestDocumentRecord:
    """Tests for the DocumentRecord dataclass."""

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(DocumentRecord)

    def test_creation_with_all_fields(self):
        now = datetime(2024, 1, 15, 10, 30, 0)
        vec = csr_matrix([1.0, 0.0, 0.5])
        record = DocumentRecord(
            doc_id="abc-123",
            filename="report.pdf",
            text="some extracted text",
            vector=vec,
            ingested_at=now,
        )
        assert record.doc_id == "abc-123"
        assert record.filename == "report.pdf"
        assert record.text == "some extracted text"
        assert record.vector is vec
        assert record.ingested_at == now

    def test_vector_can_be_none(self):
        record = DocumentRecord(
            doc_id="id-1",
            filename="file.pdf",
            text="text",
            vector=None,
            ingested_at=datetime.utcnow(),
        )
        assert record.vector is None

    def test_has_expected_fields(self):
        field_names = {f.name for f in dataclasses.fields(DocumentRecord)}
        assert field_names == {"doc_id", "filename", "text", "vector", "ingested_at"}


class TestComparisonResult:
    """Tests for the ComparisonResult dataclass."""

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ComparisonResult)

    def test_duplicate_found(self):
        result = ComparisonResult(
            is_duplicate=True,
            highest_score=0.99,
            matching_doc_id="match-id",
            store_empty=False,
        )
        assert result.is_duplicate is True
        assert result.highest_score == 0.99
        assert result.matching_doc_id == "match-id"
        assert result.store_empty is False

    def test_no_duplicate(self):
        result = ComparisonResult(
            is_duplicate=False,
            highest_score=0.45,
            matching_doc_id=None,
            store_empty=False,
        )
        assert result.is_duplicate is False
        assert result.matching_doc_id is None

    def test_empty_store(self):
        result = ComparisonResult(
            is_duplicate=False,
            highest_score=0.0,
            matching_doc_id=None,
            store_empty=True,
        )
        assert result.store_empty is True

    def test_has_expected_fields(self):
        field_names = {f.name for f in dataclasses.fields(ComparisonResult)}
        assert field_names == {"is_duplicate", "highest_score", "matching_doc_id", "store_empty"}


class TestIngestionResult:
    """Tests for the IngestionResult dataclass."""

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(IngestionResult)

    def test_creation(self):
        result = IngestionResult(doc_id="new-id", message="Document ingested successfully")
        assert result.doc_id == "new-id"
        assert result.message == "Document ingested successfully"

    def test_has_expected_fields(self):
        field_names = {f.name for f in dataclasses.fields(IngestionResult)}
        assert field_names == {"doc_id", "message"}
