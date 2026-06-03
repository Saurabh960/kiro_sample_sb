# PDF Duplicate Detector

Detects duplicate PDF documents using TF-IDF vectorization and cosine similarity. Documents scoring ≥ 0.98 similarity are flagged as duplicates.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install pdfplumber scikit-learn scipy
```

### Azure Document Intelligence (optional)

If you want to use Azure instead of local extraction:

```bash
pip install azure-ai-documentintelligence
```

Set these environment variables before running:

```bash
export AZURE_DOC_INTEL_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
export AZURE_DOC_INTEL_KEY=your-key-here
```

On Windows:
```cmd
set AZURE_DOC_INTEL_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
set AZURE_DOC_INTEL_KEY=your-key-here
```

## How to Run

```bash
python -m pdf_duplicate_detector.main
```

The program will:
1. Ask which processing method to use (local or Azure)
2. Ask for a folder path — all PDFs in that folder become your "repository"
3. Prompt you for a file to compare against the repository
4. Tell you if it's a match (with score) or a new document

## What to Expect

```
Choose document processing method:
  1. Local (pdfplumber)
  2. Azure Document Intelligence
Enter 1 or 2: 1

Enter the path to your PDF repository folder: /path/to/pdfs
Loading PDFs...
Tried 5, ingested 5.

Enter path to a PDF to check (or 'quit' to exit): /path/to/mystery.pdf
  MATCH FOUND! Score: 0.9923
  Closest match: report2.pdf

Enter path to a PDF to check (or 'quit' to exit): /path/to/something_new.pdf
  Looks like a new document :)
  (Highest similarity: 0.3412)
```

Type `quit` to exit.
