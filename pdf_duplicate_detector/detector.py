"""Duplicate detection via cosine similarity comparison."""

from __future__ import annotations

import math

from scipy.sparse import spmatrix

from pdf_duplicate_detector.models import ComparisonResult


class DuplicateDetector:
    """Computes similarity between TF-IDF vectors and classifies duplicates.

    Uses cosine similarity to compare a query vector against all stored
    vectors, applying a threshold to determine duplicate status.
    """

    DUPLICATE_THRESHOLD: float = 0.98

    def cosine_similarity(self, vec_a: spmatrix, vec_b: spmatrix) -> float:
        """Compute cosine similarity between two sparse vectors.

        Args:
            vec_a: First sparse TF-IDF vector.
            vec_b: Second sparse TF-IDF vector.

        Returns:
            Float between 0.0 and 1.0.
        """
        dot_product = (vec_a.multiply(vec_b)).sum()
        norm_a = math.sqrt(vec_a.multiply(vec_a).sum())
        norm_b = math.sqrt(vec_b.multiply(vec_b).sum())

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = float(dot_product / (norm_a * norm_b))
        # Clamp to [0.0, 1.0] to handle floating-point imprecision
        return max(0.0, min(1.0, similarity))

    def compare(
        self,
        query_vector: spmatrix,
        stored_vectors: list[tuple[str, spmatrix]],
    ) -> ComparisonResult:
        """Compare a query vector against all stored vectors using cosine similarity.

        Args:
            query_vector: TF-IDF vector of the query document.
            stored_vectors: List of (doc_id, vector) tuples from the store.

        Returns:
            ComparisonResult with duplicate status, score, and matching ID.
        """
        if not stored_vectors:
            return ComparisonResult(
                is_duplicate=False,
                highest_score=0.0,
                matching_doc_id=None,
                store_empty=True,
            )

        best_score = 0.0
        best_doc_id: str | None = None

        for doc_id, vector in stored_vectors:
            score = self.cosine_similarity(query_vector, vector)
            if score > best_score:
                best_score = score
                best_doc_id = doc_id

        is_duplicate = best_score >= self.DUPLICATE_THRESHOLD

        return ComparisonResult(
            is_duplicate=is_duplicate,
            highest_score=best_score,
            matching_doc_id=best_doc_id if is_duplicate else None,
            store_empty=False,
        )
