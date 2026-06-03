"""PDF text extraction using pdfplumber or Azure Document Intelligence."""

from __future__ import annotations

import os

import pdfplumber

from pdf_duplicate_detector.exceptions import EmptyDocumentError, PDFExtractionError


class PDFTextExtractor:
    """Extracts text using pdfplumber (local)."""

    def extract_text(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_texts: list[str] = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        page_texts.append(text)
        except Exception as exc:
            raise PDFExtractionError(
                f"Failed to extract text from '{pdf_path}': {exc}"
            ) from exc

        full_text = "\n".join(page_texts)
        if not full_text.strip():
            raise EmptyDocumentError(f"No extractable text found in '{pdf_path}'")
        return full_text


class AzureDocIntelExtractor:
    """Extracts text using Azure Document Intelligence.

    Requires environment variables:
      AZURE_DOC_INTEL_ENDPOINT
      AZURE_DOC_INTEL_KEY
    """

    def __init__(self) -> None:
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "Install azure-ai-documentintelligence: "
                "pip install azure-ai-documentintelligence"
            )

        endpoint = os.environ.get("AZURE_DOC_INTEL_ENDPOINT", "")
        key = os.environ.get("AZURE_DOC_INTEL_KEY", "")
        if not endpoint or not key:
            raise PDFExtractionError(
                "Set AZURE_DOC_INTEL_ENDPOINT and AZURE_DOC_INTEL_KEY env vars."
            )

        self._client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def extract_text(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as f:
                poller = self._client.begin_analyze_document(
                    "prebuilt-read", analyze_request=f, content_type="application/pdf"
                )
            result = poller.result()
        except Exception as exc:
            raise PDFExtractionError(
                f"Azure extraction failed for '{pdf_path}': {exc}"
            ) from exc

        full_text = result.content or ""
        if not full_text.strip():
            raise EmptyDocumentError(f"No extractable text found in '{pdf_path}'")
        return full_text
