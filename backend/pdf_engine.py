"""PDF text extraction and analysis engine."""

from __future__ import annotations
import os
import tempfile
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_bytes: bytes) -> dict:
    """Extract text content from a PDF file.
    
    Returns:
        dict with 'text', 'pages', 'page_texts', 'word_count'
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        page_texts = []
        full_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                page_texts.append({
                    "page": page_num + 1,
                    "text": text,
                })
                full_text.append(text)

        doc.close()
        combined = "\n\n".join(full_text)

        return {
            "text": combined,
            "pages": len(page_texts),
            "page_texts": page_texts,
            "word_count": len(combined.split()),
        }
    finally:
        os.unlink(tmp_path)
