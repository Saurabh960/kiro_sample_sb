"""In-memory vector store for document records with JSON persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scipy.sparse import spmatrix

from pdf_duplicate_detector.exceptions import DuplicateIdentifierError
from pdf_duplicate_detector.models import DocumentRecord


class VectorStore:
    """In-memory store for document records and their TF-IDF vectors.

    Provides CRUD operations for document records and bulk vector
    updates needed after TF-IDF refitting.
    """

    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}

    def add_document(self, record: DocumentRecord) -> str:
        """Store a document record. Raises if doc_id already exists.

        Args:
            record: The document record to store.

        Returns:
            The assigned document identifier.

        Raises:
            DuplicateIdentifierError: If a document with the same ID exists.
        """
        if record.doc_id in self._documents:
            raise DuplicateIdentifierError(
                f"Document with ID '{record.doc_id}' already exists in the store."
            )
        self._documents[record.doc_id] = record
        return record.doc_id

    def get_all_documents(self) -> list[DocumentRecord]:
        """Return all stored document records."""
        return list(self._documents.values())

    def get_all_vectors(self) -> list[tuple[str, spmatrix]]:
        """Return list of (doc_id, vector) tuples for documents that have vectors."""
        return [
            (doc_id, record.vector)
            for doc_id, record in self._documents.items()
            if record.vector is not None
        ]

    def update_vectors(self, vectors: dict[str, spmatrix]) -> None:
        """Bulk update vectors for existing documents (used after refitting).

        Args:
            vectors: Mapping of doc_id to new TF-IDF vector.
        """
        for doc_id, vector in vectors.items():
            if doc_id in self._documents:
                self._documents[doc_id].vector = vector

    def remove_document(self, doc_id: str) -> None:
        """Remove a document by ID. Used for rollback on failed ingestion.

        Silently ignores if the doc_id does not exist.
        """
        self._documents.pop(doc_id, None)

    def is_empty(self) -> bool:
        """Return True if the store contains no documents."""
        return len(self._documents) == 0

    def document_count(self) -> int:
        """Return the number of stored documents."""
        return len(self._documents)

    def save(self, filepath: str) -> None:
        """Persist all documents to a JSON file.

        Vectors are not serialized — they are recomputed on load by
        refitting the TF-IDF model on all stored texts.

        Args:
            filepath: Path to the JSON file to write.
        """
        documents = []
        for record in self._documents.values():
            documents.append(
                {
                    "doc_id": record.doc_id,
                    "filename": record.filename,
                    "text": record.text,
                    "ingested_at": record.ingested_at.isoformat(),
                }
            )
        data = {"documents": documents}
        Path(filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, filepath: str, vectorizer: "TFIDFVectorizer") -> None:
        """Load documents from a JSON file and refit the TF-IDF model.

        Reads persisted document records, reconstructs DocumentRecord
        instances, refits the vectorizer on all stored texts, and
        updates vectors in the store.

        Args:
            filepath: Path to the JSON file to read.
            vectorizer: A TFIDFVectorizer instance used to refit vectors.
        """
        from pdf_duplicate_detector.vectorizer import TFIDFVectorizer  # noqa: F811

        path = Path(filepath)
        if not path.exists():
            return

        raw = json.loads(path.read_text(encoding="utf-8"))
        self._documents.clear()

        for entry in raw.get("documents", []):
            record = DocumentRecord(
                doc_id=entry["doc_id"],
                filename=entry["filename"],
                text=entry["text"],
                vector=None,
                ingested_at=datetime.fromisoformat(entry["ingested_at"]),
            )
            self._documents[record.doc_id] = record

        # Refit TF-IDF on all stored texts to reconstruct vectors
        if self._documents:
            all_docs = self.get_all_documents()
            corpus = [doc.text for doc in all_docs]
            vectors = vectorizer.fit_transform(corpus)
            vector_map = {doc.doc_id: vec for doc, vec in zip(all_docs, vectors)}
            self.update_vectors(vector_map)
