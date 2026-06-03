# PDF Duplicate Detector

Detects duplicate PDF documents using TF-IDF vectorization and cosine similarity. Documents scoring ≥ 0.98 similarity are flagged as duplicates.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install pdfplumber scikit-learn scipy
```

## How to Run

```bash
python -m pdf_duplicate_detector.main
```

The program will:
1. Ask you for a folder path — all PDFs in that folder become your "repository"
2. Load and index every PDF in that folder
3. Prompt you for a file to compare against the repository
4. Tell you if it's a match (with score) or a new document

## What to Expect

```
Enter the path to your PDF repository folder: /path/to/pdfs
Loading 5 PDF(s) into the vector store...
  ✓ report1.pdf
  ✓ report2.pdf
  ...
Repository loaded. 5 document(s) in store.

Enter path to a PDF to check (or 'quit' to exit): /path/to/mystery.pdf
  MATCH FOUND! Score: 0.9923
  Closest match: report2.pdf

Enter path to a PDF to check (or 'quit' to exit): /path/to/something_new.pdf
  Looks like a new document :)
  (Highest similarity: 0.3412)
```

Type `quit` to exit.
