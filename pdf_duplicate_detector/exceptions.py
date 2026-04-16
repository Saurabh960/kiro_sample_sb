"""Custom exceptions for the PDF Duplicate Detector system."""


class PDFExtractionError(Exception):
    """Raised when a PDF file is invalid, corrupted, or unreadable.

    Raised by PDFTextExtractor when pdfplumber cannot parse the provided file.
    """


class EmptyDocumentError(Exception):
    """Raised when a PDF file contains no extractable text.

    Raised by PDFTextExtractor when the extracted text is empty or
    whitespace-only after processing all pages.
    """


class VectorizerNotFittedError(Exception):
    """Raised when transform is called before fit_transform.

    Raised by TFIDFVectorizer when attempting to transform a document
    before the model has been fitted on a corpus.
    """


class DuplicateIdentifierError(Exception):
    """Raised when a document with the same ID already exists in the store.

    Raised by VectorStore when attempting to add a document whose doc_id
    matches an existing record.
    """


class IngestionError(Exception):
    """Raised when any step in the ingestion pipeline fails.

    Raised by PDFIngestionService to wrap errors from extraction,
    vectorization, or storage steps. The original exception is preserved
    as the cause via exception chaining.
    """
