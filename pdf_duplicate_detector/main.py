"""Interactive CLI for the PDF Duplicate Detector."""

from __future__ import annotations

import glob
import os
import sys

from pdf_duplicate_detector.detector import DuplicateDetector
from pdf_duplicate_detector.extractor import AzureDocIntelExtractor, PDFTextExtractor
from pdf_duplicate_detector.ingestion import PDFIngestionService
from pdf_duplicate_detector.store import VectorStore
from pdf_duplicate_detector.vectorizer import TFIDFVectorizer


def main() -> None:
    # Step 1: Choose extraction method
    print("Choose document processing method:")
    print("  1. Local (pdfplumber)")
    print("  2. Azure Document Intelligence")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        try:
            extractor = AzureDocIntelExtractor()
        except Exception as e:
            print(f"Error initializing Azure: {e}")
            sys.exit(1)
    else:
        extractor = PDFTextExtractor()

    vectorizer = TFIDFVectorizer()
    store = VectorStore()
    detector = DuplicateDetector()
    service = PDFIngestionService(extractor, vectorizer, store, detector)

    # Step 2: Ask for repository path
    repo_path = input("\nEnter the path to your PDF repository folder: ").strip()
    if not os.path.isdir(repo_path):
        print(f"Error: '{repo_path}' is not a valid directory.")
        sys.exit(1)

    # Load all PDFs from that path
    pdf_files = glob.glob(os.path.join(repo_path, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{repo_path}'.")
        sys.exit(1)

    print(f"\nLoading PDFs...")
    failed = 0
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            service.ingest(pdf_path, filename)
        except Exception as e:
            failed += 1
            print(f"  ✗ {filename} — {e}")

    ingested = store.document_count()
    print(f"Tried {len(pdf_files)}, ingested {ingested}.\n")

    # Step 3: Query loop
    while True:
        query_path = input("Enter path to a PDF to check (or 'quit' to exit): ").strip()
        if query_path.lower() in ("quit", "q", "exit"):
            print("Goodbye!")
            break

        if not os.path.isfile(query_path):
            print(f"  File not found: '{query_path}'\n")
            continue

        try:
            result = service.query(query_path)
        except Exception as e:
            print(f"  Error: {e}\n")
            continue

        if result.store_empty:
            print("  Store is empty — nothing to compare against.\n")
            continue

        # Find the closest matching filename
        match_name = None
        for doc in store.get_all_documents():
            if doc.doc_id == result.matching_doc_id:
                match_name = doc.filename
                break

        if result.is_duplicate:
            print(f"  MATCH FOUND!")
        else:
            print(f"  Looks like a new document :)")

        print(f"  Closest match: {match_name}")
        print(f"  Similarity score: {result.highest_score:.4f}\n")


if __name__ == "__main__":
    main()
