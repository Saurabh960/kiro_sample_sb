"""CLI entry point for the PDF Duplicate Detector."""

from __future__ import annotations

import argparse
import sys

from pdf_duplicate_detector.detector import DuplicateDetector
from pdf_duplicate_detector.extractor import PDFTextExtractor
from pdf_duplicate_detector.ingestion import PDFIngestionService
from pdf_duplicate_detector.store import VectorStore
from pdf_duplicate_detector.vectorizer import TFIDFVectorizer

DEFAULT_STORE_PATH = "document_store.json"


def create_service(store_path: str = DEFAULT_STORE_PATH) -> PDFIngestionService:
    """Wire all components together and return a ready-to-use service."""
    extractor = PDFTextExtractor()
    vectorizer = TFIDFVectorizer()
    store = VectorStore()
    detector = DuplicateDetector()

    store.load(store_path, vectorizer)

    return PDFIngestionService(extractor, vectorizer, store, detector)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a PDF into the document store."""
    service = create_service(args.store)
    result = service.ingest(args.pdf, args.pdf.split("/")[-1])

    # Persist after successful ingestion
    service._store.save(args.store)

    print(f"Ingested: {result.doc_id}")
    print(result.message)


def cmd_query(args: argparse.Namespace) -> None:
    """Query a PDF against the document store."""
    service = create_service(args.store)
    result = service.query(args.pdf)

    if result.store_empty:
        print("Store is empty — no documents to compare against.")
        return

    if result.is_duplicate:
        print(f"DUPLICATE FOUND (score: {result.highest_score:.4f})")
        print(f"Matching document: {result.matching_doc_id}")
    else:
        print(f"No duplicate found (highest score: {result.highest_score:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Duplicate Detector")
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE_PATH,
        help=f"Path to the JSON store file (default: {DEFAULT_STORE_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF into the store")
    ingest_parser.add_argument("pdf", help="Path to the PDF file")

    query_parser = subparsers.add_parser("query", help="Check a PDF for duplicates")
    query_parser.add_argument("pdf", help="Path to the PDF file")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
