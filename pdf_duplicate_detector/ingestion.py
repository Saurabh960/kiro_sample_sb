"""PDF ingestion and query orchestration service."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pdf_duplicate_detector.detector import DuplicateDetector
from pdf_duplicate_detector.exceptions import IngestionError
from pdf_duplicate_detector.extractor import PDFTextExtractor
from pdf_duplicate_detector.models import (
    ComparisonResult,
    DocumentRecord,
    IngestionResult,
)
from pdf_duplicate_detector.store import VectorStore
from pdf_duplicate_detector.vectorizer import TFIDFVectorizer


class PDFIngestionService:
    """Orchestrates the ingestion and query workflows.

    Wires together PDFTextExtractor, TFIDFVectorizer, VectorStore,
    and DuplicateDetector to provide high-level ingest and query
    operations on PDF documents.
    """

    def __init__(
        self,
        extractor: PDFTextExtractor,
        vectorizer: TFIDFVectorizer,
        store: VectorStore,
        detector: DuplicateDetector,
    ) -> None:
        self._extractor = extractor
        self._vectorizer = vectorizer
        self._store = store
        self._detector = detector

    def ingest(self, pdf_path: str, filename: str) -> IngestionResult:
        """Full ingestion pipeline: extract text → preprocess → store → refit → update vectors.

        Rolls back on failure to prevent partial data. If vectorization
        or storage fails after the document has been added to the store,
        the document is removed via ``remove_document``.

        Args:
            pdf_path: Path to the PDF file.
            filename: Original filename for the record.

        Returns:
            IngestionResult with assigned doc_id and confirmation message.

        Raises:
            IngestionError: If any pipeline step fails. The original
                exception is preserved as the cause via exception chaining.
        """
        doc_id = str(uuid.uuid4())
        added_to_store = False

        try:
            # Step 1: Extract text from PDF
            raw_text = self._extractor.extract_text(pdf_path)

            # Step 2: Preprocess the extracted text
            preprocessed_text = self._vectorizer.preprocess(raw_text)

            # Step 3: Create a document record and add to store
            record = DocumentRecord(
                doc_id=doc_id,
                filename=filename,
                text=preprocessed_text,
                vector=None,
                ingested_at=datetime.now(timezone.utc),
            )
            self._store.add_document(record)
            added_to_store = True

            # Step 4: Refit TF-IDF on the full corpus
            all_documents = self._store.get_all_documents()
            corpus = [doc.text for doc in all_documents]
            vectors = self._vectorizer.fit_transform(corpus)

            # Step 5: Update all vectors in the store
            vector_map = {
                doc.doc_id: vec for doc, vec in zip(all_documents, vectors)
            }
            self._store.update_vectors(vector_map)

        except Exception as exc:
            # Rollback: remove the document if it was added to the store
            if added_to_store:
                self._store.remove_document(doc_id)
            raise IngestionError(
                f"Ingestion failed for '{filename}': {exc}"
            ) from exc

        return IngestionResult(
            doc_id=doc_id,
            message=f"Document '{filename}' ingested successfully.",
        )

    def query(self, pdf_path: str) -> ComparisonResult:
        """Full query pipeline: extract text → preprocess → transform → compare.

        Does not modify the VectorStore. Uses the currently fitted
        TF-IDF model to transform the query document and compares
        it against all stored vectors.

        Args:
            pdf_path: Path to the query PDF file.

        Returns:
            ComparisonResult indicating duplicate status.

        Raises:
            IngestionError: If any pipeline step fails during query.
        """
        try:
            # Step 1: Extract text from PDF
            raw_text = self._extractor.extract_text(pdf_path)

            # Step 2: Preprocess the extracted text
            preprocessed_text = self._vectorizer.preprocess(raw_text)

            # Step 3: Check if the store is empty
            if self._store.is_empty():
                return ComparisonResult(
                    is_duplicate=False,
                    highest_score=0.0,
                    matching_doc_id=None,
                    store_empty=True,
                )

            # Step 4: Transform with the fitted model
            query_vector = self._vectorizer.transform(preprocessed_text)

            # Step 5: Compare against stored vectors
            stored_vectors = self._store.get_all_vectors()
            return self._detector.compare(query_vector, stored_vectors)

        except Exception as exc:
            raise IngestionError(
                f"Query failed for '{pdf_path}': {exc}"
            ) from exc
