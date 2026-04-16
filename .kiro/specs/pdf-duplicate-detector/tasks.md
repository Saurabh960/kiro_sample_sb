# Implementation Plan: PDF Duplicate Detector

## Overview

Implement a Python-based PDF duplicate detection system using TF-IDF vectorization and cosine similarity. The system provides two workflows — ingestion (add a document to the store) and query (check for duplicates) — built on four core components: PDFTextExtractor, TFIDFVectorizer, VectorStore, and DuplicateDetector, with two orchestrators: PDFIngestionService and DuplicateDetector query flow.

## Tasks

- [x] 1. Set up project structure, dependencies, and custom exceptions
  - [x] 1.1 Create project directory structure and install dependencies
    - Create `pdf_duplicate_detector/` package with `__init__.py`
    - Create `tests/` directory with `__init__.py` and `conftest.py`
    - Create `pyproject.toml` (or `requirements.txt`) with dependencies: `pdfplumber`, `scikit-learn`, `scipy`, `hypothesis`, `pytest`
    - _Requirements: 1.1, 2.1, 3.1_

  - [x] 1.2 Define custom exception classes
    - Create `pdf_duplicate_detector/exceptions.py`
    - Implement `PDFExtractionError`, `EmptyDocumentError`, `VectorizerNotFittedError`, `DuplicateIdentifierError`, `IngestionError`
    - _Requirements: 1.3, 1.4, 5.3_

  - [x] 1.3 Define data model classes
    - Create `pdf_duplicate_detector/models.py`
    - Implement `DocumentRecord` dataclass with fields: `doc_id`, `filename`, `text`, `vector`, `ingested_at`
    - Implement `ComparisonResult` dataclass with fields: `is_duplicate`, `highest_score`, `matching_doc_id`, `store_empty`
    - Implement `IngestionResult` dataclass with fields: `doc_id`, `message`
    - _Requirements: 3.1, 4.3, 5.2_

- [-] 2. Implement PDFTextExtractor
  - [x] 2.1 Implement the PDFTextExtractor class
    - Create `pdf_duplicate_detector/extractor.py`
    - Implement `extract_text(pdf_path: str) -> str` using pdfplumber
    - Concatenate text from all pages in page order
    - Raise `PDFExtractionError` for invalid/corrupted files
    - Raise `EmptyDocumentError` when no text is extractable
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ] 2.2 Write unit tests for PDFTextExtractor
    - Create `tests/test_extractor.py`
    - Test extraction from a valid single-page PDF
    - Test extraction from a multi-page PDF (pages concatenated in order)
    - Test that `PDFExtractionError` is raised for corrupted/invalid files
    - Test that `EmptyDocumentError` is raised for PDFs with no text
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [-] 3. Implement TFIDFVectorizer
  - [x] 3.1 Implement the TFIDFVectorizer class
    - Create `pdf_duplicate_detector/vectorizer.py`
    - Implement `preprocess(text: str) -> str` for lowercasing and whitespace normalization
    - Implement `fit_transform(corpus: list[str]) -> list[spmatrix]` using scikit-learn's `TfidfVectorizer`
    - Implement `transform(text: str) -> spmatrix` for single-document vectorization
    - Raise `VectorizerNotFittedError` if `transform` is called before `fit_transform`
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ] 3.2 Write property test: text preprocessing normalizes case and whitespace
    - **Property 1: Text preprocessing normalizes case and whitespace**
    - Use Hypothesis `st.text()` to generate arbitrary strings
    - Assert output is entirely lowercase, has no leading/trailing whitespace, and no consecutive whitespace
    - **Validates: Requirements 2.2**

  - [ ] 3.3 Write property test: vectorizer produces valid sparse vectors
    - **Property 2: Vectorizer produces valid sparse vectors**
    - Use Hypothesis `st.text(min_size=1)` filtered to non-empty after preprocessing
    - Assert returned vector is a sparse matrix with non-negative values and correct dimensions
    - **Validates: Requirements 2.1**

  - [ ] 3.4 Write property test: vectorization round-trip preserves term weights
    - **Property 3: Vectorization round-trip preserves term weights**
    - Use Hypothesis `st.text(min_size=1)` with alphabetic characters
    - Vectorize, reconstruct term-weight pairs, re-vectorize, and assert equivalence
    - **Validates: Requirements 2.4**

  - [ ] 3.5 Write unit tests for TFIDFVectorizer
    - Test `preprocess` with known inputs (mixed case, extra whitespace)
    - Test `fit_transform` produces correct number of vectors
    - Test `transform` on a fitted model returns a valid vector
    - Test `VectorizerNotFittedError` is raised when `transform` is called before fitting
    - _Requirements: 2.1, 2.2, 2.3_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [-] 5. Implement VectorStore
  - [x] 5.1 Implement the VectorStore class
    - Create `pdf_duplicate_detector/store.py`
    - Implement `add_document(record: DocumentRecord) -> str` with duplicate ID check
    - Implement `get_all_documents() -> list[DocumentRecord]`
    - Implement `get_all_vectors() -> list[tuple[str, spmatrix]]`
    - Implement `update_vectors(vectors: dict[str, spmatrix]) -> None` for bulk vector updates after refitting
    - Implement `remove_document(doc_id: str) -> None` for rollback support
    - Implement `is_empty() -> bool` and `document_count() -> int`
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ] 5.2 Write property test: document store round-trip preserves all fields
    - **Property 4: Document store round-trip preserves all fields**
    - Use Hypothesis to generate random `DocumentRecord` instances with `st.uuids()`, `st.text()`, `st.datetimes()`
    - Assert stored and retrieved records have identical `doc_id`, `filename`, `text`, `ingested_at`
    - Assert storing N documents yields exactly N records on retrieval
    - **Validates: Requirements 3.1, 3.2, 3.4**

  - [ ] 5.3 Write property test: duplicate identifier rejection
    - **Property 5: Duplicate identifier rejection**
    - Use Hypothesis to generate `DocumentRecord` pairs sharing the same `doc_id`
    - Assert `DuplicateIdentifierError` is raised on the second store attempt
    - **Validates: Requirements 3.3**

  - [ ] 5.4 Write unit tests for VectorStore
    - Test `add_document` stores and returns the correct doc_id
    - Test `get_all_documents` returns all stored records
    - Test `get_all_vectors` returns correct (doc_id, vector) tuples
    - Test `update_vectors` correctly replaces vectors for existing documents
    - Test `remove_document` removes the specified record
    - Test `is_empty` and `document_count` return correct values
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [-] 6. Implement DuplicateDetector (comparison logic)
  - [x] 6.1 Implement the DuplicateDetector class
    - Create `pdf_duplicate_detector/detector.py`
    - Implement `cosine_similarity(vec_a, vec_b) -> float` using sparse matrix operations
    - Implement `compare(query_vector, stored_vectors) -> ComparisonResult`
    - Apply `DUPLICATE_THRESHOLD = 0.98` for classification
    - Return `store_empty=True` when stored_vectors is empty
    - Set `matching_doc_id` to the best match ID when duplicate, `None` otherwise
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ] 6.2 Write property test: cosine similarity score bounds
    - **Property 6: Cosine similarity score bounds**
    - Use Hypothesis to generate random non-negative sparse vectors
    - Assert cosine similarity is a float in [0.0, 1.0]
    - **Validates: Requirements 4.2**

  - [ ] 6.3 Write property test: threshold classification correctness
    - **Property 7: Threshold classification correctness**
    - Use Hypothesis to generate random float scores in [0.0, 1.0] with mock stored vectors
    - Assert `is_duplicate` is True iff `highest_score >= 0.98`
    - Assert `matching_doc_id` is non-None when `is_duplicate` is True, and None otherwise
    - **Validates: Requirements 4.3, 4.4**

  - [ ] 6.4 Write unit tests for DuplicateDetector
    - Test cosine similarity of identical vectors returns 1.0
    - Test cosine similarity of orthogonal vectors returns 0.0
    - Test `compare` returns `is_duplicate=True` when score >= 0.98
    - Test `compare` returns `is_duplicate=False` when score < 0.98
    - Test `compare` returns `store_empty=True` when no stored vectors
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [-] 8. Implement PDFIngestionService and query workflow
  - [x] 8.1 Implement the PDFIngestionService class
    - Create `pdf_duplicate_detector/ingestion.py`
    - Implement `ingest(pdf_path: str, filename: str) -> IngestionResult`
    - Pipeline: extract text → preprocess → add to store → refit TF-IDF on full corpus → update all vectors
    - Implement rollback via `remove_document` if vectorization or storage fails after text extraction
    - Raise `IngestionError` wrapping any pipeline step failure
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 8.2 Implement the query method on PDFIngestionService
    - Implement `query(pdf_path: str) -> ComparisonResult`
    - Pipeline: extract text → preprocess → transform with fitted model → compare against stored vectors
    - Ensure no modifications to the VectorStore during query
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ] 8.3 Write property test: query does not modify store
    - **Property 8: Query does not modify store**
    - Use Hypothesis to generate random store states and query vectors
    - Assert document count, doc_ids, and vectors are identical before and after query
    - **Validates: Requirements 6.3**

  - [ ] 8.4 Write unit tests for PDFIngestionService
    - Test successful ingestion returns doc_id and confirmation message
    - Test ingestion rollback: mock a failure at vectorization step and verify store is unchanged
    - Test ingestion rollback: mock a failure at storage step and verify store is unchanged
    - Test query returns correct `ComparisonResult` for a known duplicate
    - Test query returns `store_empty=True` when store is empty
    - Test corpus refitting: ingest multiple documents and verify vectors are recomputed
    - _Requirements: 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 2.3_

- [-] 9. Wire components together and add JSON persistence
  - [x] 9.1 Implement JSON persistence for VectorStore
    - Add `save(filepath: str)` and `load(filepath: str)` methods to VectorStore
    - Persist documents as JSON (doc_id, filename, text, ingested_at)
    - On load, refit TF-IDF model on all stored texts to reconstruct vectors
    - _Requirements: 3.1, 3.2_

  - [x] 9.2 Create main module with factory/wiring function
    - Create `pdf_duplicate_detector/main.py`
    - Implement a factory function that wires all components together (extractor, vectorizer, store, detector, ingestion service)
    - Provide a simple CLI or entry point for ingestion and query operations
    - _Requirements: 5.1, 6.1_

  - [ ] 9.3 Write integration tests for end-to-end workflows
    - Test full ingestion workflow: PDF file → stored document record
    - Test full query workflow: ingest a PDF, query with the same PDF, verify duplicate detected
    - Test full query workflow: ingest a PDF, query with a different PDF, verify no duplicate
    - Test persistence: ingest documents, save to JSON, reload, and verify query still works
    - _Requirements: 5.1, 5.2, 6.1, 6.2_

- [ ] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (Properties 1–8)
- Unit tests validate specific examples and edge cases
- Python with scikit-learn, pdfplumber, Hypothesis, and pytest as specified in the design
