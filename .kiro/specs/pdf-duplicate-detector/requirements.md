# Requirements Document

## Introduction

A simple PDF document duplicate detection system. The system accepts PDF documents, extracts text, generates TF-IDF vectors, and stores them in a vector database. When a new document is submitted for comparison, the system returns a cosine similarity score indicating whether a duplicate exists in the document store, using a 98% certainty threshold.

## Glossary

- **PDF_Ingestion_Service**: The component responsible for accepting PDF files and extracting raw text content from them.
- **TF_IDF_Vectorizer**: The component that transforms extracted text into TF-IDF (Term Frequency–Inverse Document Frequency) vector representations.
- **Vector_Store**: The vector database that persists TF-IDF vectors alongside document metadata for similarity search.
- **Duplicate_Detector**: The component that compares a query document's TF-IDF vector against all stored vectors and returns similarity scores.
- **Similarity_Score**: A cosine similarity value between 0.0 and 1.0, where 1.0 indicates identical documents.
- **Duplicate_Threshold**: The similarity score cutoff of 0.98 (98%) at or above which two documents are considered duplicates.
- **Document_Record**: A stored entry containing the document identifier, original filename, TF-IDF vector, and ingestion timestamp.

## Requirements

### Requirement 1: PDF Text Extraction

**User Story:** As a user, I want to upload a PDF document and have its text extracted, so that the system can analyze its content for duplicate detection.

#### Acceptance Criteria

1. WHEN a valid PDF file is provided, THE PDF_Ingestion_Service SHALL extract all text content from the PDF and return it as a single plain-text string.
2. WHEN a PDF file contains multiple pages, THE PDF_Ingestion_Service SHALL extract and concatenate text from all pages in page order.
3. IF an invalid or corrupted PDF file is provided, THEN THE PDF_Ingestion_Service SHALL return a descriptive error message indicating the file could not be parsed.
4. IF a PDF file contains no extractable text, THEN THE PDF_Ingestion_Service SHALL return an error message indicating the document contains no text content.

### Requirement 2: TF-IDF Vector Generation

**User Story:** As a user, I want the system to generate a TF-IDF vector from my document's text, so that the document can be numerically compared against other documents.

#### Acceptance Criteria

1. WHEN extracted text is provided, THE TF_IDF_Vectorizer SHALL generate a TF-IDF vector representation of the text.
2. THE TF_IDF_Vectorizer SHALL apply text preprocessing including lowercasing and whitespace normalization before vectorization.
3. WHEN a new document is ingested, THE TF_IDF_Vectorizer SHALL refit the TF-IDF model on the full corpus to maintain accurate term frequencies across all stored documents.
4. FOR ALL valid text inputs, vectorizing then reconstructing the term-weight pairs then re-vectorizing SHALL produce an equivalent vector (round-trip property).

### Requirement 3: Vector Storage

**User Story:** As a user, I want document vectors stored persistently, so that future documents can be compared against all previously ingested documents.

#### Acceptance Criteria

1. WHEN a TF-IDF vector is generated for a new document, THE Vector_Store SHALL persist a Document_Record containing the document identifier, original filename, TF-IDF vector, and ingestion timestamp.
2. THE Vector_Store SHALL support retrieval of all stored vectors for similarity comparison.
3. IF a document with the same identifier already exists in the Vector_Store, THEN THE Vector_Store SHALL return an error indicating a duplicate identifier.
4. WHEN a document is stored, THE Vector_Store SHALL confirm storage by returning the assigned document identifier.

### Requirement 4: Duplicate Detection via Similarity Search

**User Story:** As a user, I want to submit a new PDF and find out if a duplicate exists in the document store, so that I can identify redundant documents.

#### Acceptance Criteria

1. WHEN a query PDF is submitted for comparison, THE Duplicate_Detector SHALL compute the Similarity_Score between the query document's TF-IDF vector and every vector in the Vector_Store.
2. THE Duplicate_Detector SHALL use cosine similarity as the similarity metric, producing a score between 0.0 and 1.0.
3. WHEN the highest Similarity_Score meets or exceeds the Duplicate_Threshold of 0.98, THE Duplicate_Detector SHALL return a result indicating a duplicate was found, including the matching document identifier and the Similarity_Score.
4. WHEN no stored document has a Similarity_Score at or above the Duplicate_Threshold, THE Duplicate_Detector SHALL return a result indicating no duplicate was found, along with the highest Similarity_Score observed.
5. IF the Vector_Store contains no documents, THEN THE Duplicate_Detector SHALL return a result indicating the store is empty and no comparison was performed.

### Requirement 5: Document Ingestion Workflow

**User Story:** As a user, I want a single operation to ingest a PDF into the document store, so that I can build up my collection of documents for future comparison.

#### Acceptance Criteria

1. WHEN a PDF file is submitted for ingestion, THE PDF_Ingestion_Service SHALL extract text, generate a TF-IDF vector, and store the Document_Record in the Vector_Store in a single operation.
2. WHEN ingestion completes successfully, THE PDF_Ingestion_Service SHALL return the assigned document identifier and a confirmation message.
3. IF any step in the ingestion pipeline fails, THEN THE PDF_Ingestion_Service SHALL return an error describing which step failed without leaving partial data in the Vector_Store.

### Requirement 6: Query Workflow

**User Story:** As a user, I want a single operation to check a PDF against the document store, so that I can quickly determine if a duplicate exists.

#### Acceptance Criteria

1. WHEN a PDF file is submitted for duplicate checking, THE Duplicate_Detector SHALL extract text, generate a TF-IDF vector, and compare it against all stored vectors in a single operation.
2. THE Duplicate_Detector SHALL return the comparison result including: whether a duplicate was found (boolean), the highest Similarity_Score, and the matching document identifier if a duplicate was found.
3. THE Duplicate_Detector SHALL complete the query operation without modifying the contents of the Vector_Store.
