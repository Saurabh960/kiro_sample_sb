# PDF Duplicate Detector

Detects duplicate PDF documents using TF-IDF vectorization and cosine similarity. Documents scoring ≥ 0.98 similarity are flagged as duplicates.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## How to Run

**Compare two PDFs directly (simplest way):**
```bash
.venv/bin/python -m pdf_duplicate_detector.main compare file1.pdf file2.pdf
```

**Add a PDF to the store (for batch use):**
```bash
.venv/bin/python -m pdf_duplicate_detector.main ingest path/to/your.pdf
```

**Check a PDF against the store:**
```bash
.venv/bin/python -m pdf_duplicate_detector.main query path/to/another.pdf
```

## What to Expect

**Compare mode:**
- `Similarity score: 1.0000` / `DUPLICATE — these documents match (>= 0.98 threshold)`
- `Similarity score: 0.2928` / `NOT a duplicate (below 0.98 threshold)`

**Store mode:**
- **Duplicate found:** `DUPLICATE FOUND (score: 1.0000)` with the matching document ID
- **No duplicate:** `No duplicate found (highest score: 0.2928)`
- **Empty store:** `Store is empty — no documents to compare against.`

The document store defaults to `document_store.json` in your working directory and persists between runs.
