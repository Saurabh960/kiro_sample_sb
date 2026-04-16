"""PDF text extraction using pdfplumber."""

from __future__ import annotations

import pdfplumber

from pdf_duplicate_detector.exceptions import EmptyDocumentError, PDFExtractionError


class PDFTextExtractor:
    """Extracts text content from PDF files.

    Uses pdfplumber to parse PDF pages and concatenate their text
    in page order into a single plain-text string.
    """

    def extract_text(self, pdf_path: str) -> str:
        """Extract all text from a PDF file, concatenating pages in order.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Concatenated plain-text string from all pages.

        Raises:
            PDFExtractionError: If the file is invalid, corrupted, or unreadable.
            EmptyDocumentError: If the PDF contains no extractable text.
        """
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
            raise EmptyDocumentError(
                f"No extractable text found in '{pdf_path}'"
            )

        return full_text
