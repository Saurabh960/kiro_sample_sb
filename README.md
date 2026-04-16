# PDF Duplicate Detector

Detects duplicate PDF documents using TF-IDF vectorization and cosine similarity. Documents scoring ≥ 0.98 similarity are flagged as duplicates.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## How to Run

**Add a PDF to the store:**
```bash
.venv/bin/python -m pdf_duplicate_detector.main ingest path/to/your.pdf
```

**Check a PDF for duplicates:**
```bash
.venv/bin/python -m pdf_duplicate_detector.main query path/to/another.pdf
```

**Use a custom store file:**
```bash
.venv/bin/python -m pdf_duplicate_detector.main --store my_collection.json ingest doc.pdf
.venv/bin/python -m pdf_duplicate_detector.main --store my_collection.json query doc2.pdf
```

## What to Expect

- **Duplicate found:** `DUPLICATE FOUND (score: 1.0000)` with the matching document ID
- **No duplicate:** `No duplicate found (highest score: 0.2928)`
- **Empty store:** `Store is empty — no documents to compare against.`

The document store defaults to `document_store.json` in your working directory and persists between runs.
