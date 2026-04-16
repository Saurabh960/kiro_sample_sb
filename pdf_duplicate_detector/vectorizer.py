"""TF-IDF vectorization for the PDF Duplicate Detector system."""

from __future__ import annotations

import re

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

from pdf_duplicate_detector.exceptions import VectorizerNotFittedError


class TFIDFVectorizer:
    """Manages TF-IDF model fitting and vector generation.

    Wraps scikit-learn's TfidfVectorizer to provide a simple interface
    for fitting on a corpus and transforming individual documents.
    """

    def __init__(self) -> None:
        self._vectorizer: SklearnTfidfVectorizer | None = None

    def preprocess(self, text: str) -> str:
        """Apply text preprocessing: lowercasing and whitespace normalization.

        Args:
            text: Raw text string.

        Returns:
            Preprocessed text string with lowercase characters,
            no leading/trailing whitespace, and no consecutive whitespace.
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def fit_transform(self, corpus: list[str]) -> list[spmatrix]:
        """Fit the TF-IDF model on the full corpus and return vectors for all documents.

        Args:
            corpus: List of preprocessed text strings.

        Returns:
            List of sparse TF-IDF vectors, one per document.
        """
        self._vectorizer = SklearnTfidfVectorizer()
        matrix = self._vectorizer.fit_transform(corpus)
        return [matrix[i] for i in range(matrix.shape[0])]

    def transform(self, text: str) -> spmatrix:
        """Transform a single document using the currently fitted model.

        Args:
            text: Preprocessed text string.

        Returns:
            Sparse TF-IDF vector.

        Raises:
            VectorizerNotFittedError: If the model has not been fitted yet.
        """
        if self._vectorizer is None:
            raise VectorizerNotFittedError(
                "TFIDFVectorizer has not been fitted. Call fit_transform first."
            )
        return self._vectorizer.transform([text])[0]
