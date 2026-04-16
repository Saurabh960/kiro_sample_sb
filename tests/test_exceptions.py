"""Tests for custom exception classes."""

import pytest

from pdf_duplicate_detector.exceptions import (
    DuplicateIdentifierError,
    EmptyDocumentError,
    IngestionError,
    PDFExtractionError,
    VectorizerNotFittedError,
)


class TestPDFExtractionError:
    def test_can_be_raised_and_caught(self):
        with pytest.raises(PDFExtractionError, match="corrupted file"):
            raise PDFExtractionError("corrupted file")

    def test_is_exception_subclass(self):
        assert issubclass(PDFExtractionError, Exception)


class TestEmptyDocumentError:
    def test_can_be_raised_and_caught(self):
        with pytest.raises(EmptyDocumentError, match="no text content"):
            raise EmptyDocumentError("no text content")

    def test_is_exception_subclass(self):
        assert issubclass(EmptyDocumentError, Exception)


class TestVectorizerNotFittedError:
    def test_can_be_raised_and_caught(self):
        with pytest.raises(VectorizerNotFittedError, match="not fitted"):
            raise VectorizerNotFittedError("not fitted")

    def test_is_exception_subclass(self):
        assert issubclass(VectorizerNotFittedError, Exception)


class TestDuplicateIdentifierError:
    def test_can_be_raised_and_caught(self):
        with pytest.raises(DuplicateIdentifierError, match="already exists"):
            raise DuplicateIdentifierError("already exists")

    def test_is_exception_subclass(self):
        assert issubclass(DuplicateIdentifierError, Exception)


class TestIngestionError:
    def test_can_be_raised_and_caught(self):
        with pytest.raises(IngestionError, match="pipeline failed"):
            raise IngestionError("pipeline failed")

    def test_is_exception_subclass(self):
        assert issubclass(IngestionError, Exception)

    def test_supports_exception_chaining(self):
        original = ValueError("extraction failed")
        with pytest.raises(IngestionError) as exc_info:
            raise IngestionError("pipeline failed") from original
        assert exc_info.value.__cause__ is original
