"""Interactive CLI for the PDF Duplicate Detector."""

from __future__ import annotations

import glob
import os
import sys

from pdf_duplicate_detector.detector import DuplicateDetector
from pdf_duplicate_detector.extractor import PDFTextExtractor
from pdf_duplicate_detector.ingestion import PDFIngestionService
from pdf_duplicate_detector.store import VectorStore
from pdf_duplicate_detector.vectorizer import TFIDFVectorizer


def main() -> None:
    extractor = PDFTextExtractor()
    vectorizer = TFIDFVectorizer()
    store = VectorStore()
    detector = DuplicateDetector()
    service = PDFIngestionService(extractor, vectorizer, store, detector)

    # Step 1: Ask for repository path
    repo_path = input("Enter the path to your PDF repository folder: ").strip()
    if not os.path.isdir(repo_path):
        print(f"Error: '{repo_path}' is not a valid directory.")
        sys.exit(1)

    # Load all PDFs from that path
    pdf_files = glob.glob(os.path.join(repo_path, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{repo_path}'.")
        sys.exit(1)

    print(f"\nLoading {len(pdf_files)} PDF(s) into the vector store...")
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            service.ingest(pdf_path, filename)
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ {filename} — {e}")

    print(f"\nRepository loaded. {store.document_count()} document(s) in store.\n")

    # Step 2: Query loop
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
        elif result.is_duplicate:
            # Find the filename for the matching doc
            match_name = result.matching_doc_id
            for doc in store.get_all_documents():
                if doc.doc_id == result.matching_doc_id:
                    match_name = doc.filename
                    break
            print(f"  MATCH FOUND! Score: {result.highest_score:.4f}")
            print(f"  Closest match: {match_name}\n")
        else:
            print(f"  Looks like a new document :)")
            print(f"  (Highest similarity: {result.highest_score:.4f})\n")


if __name__ == "__main__":
    main()
