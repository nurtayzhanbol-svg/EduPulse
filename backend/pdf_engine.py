"""PDF text extraction and analysis engine."""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Union

import pymupdf as fitz
from fastapi import HTTPException, UploadFile

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 64 * 1024


def max_upload_bytes() -> int:
    """Upload cap in bytes, from ``EDUPULSE_MAX_UPLOAD_MB`` (default 10 MB)."""
    raw = os.environ.get("EDUPULSE_MAX_UPLOAD_MB")
    if raw:
        try:
            return int(float(raw) * 1024 * 1024)
        except ValueError:
            pass
    return DEFAULT_MAX_UPLOAD_BYTES


def _too_large(limit: int) -> HTTPException:
    return HTTPException(413, f"PDF is too large. Maximum upload size is {limit // (1024 * 1024)} MB.")


async def save_upload_to_tempfile(file: UploadFile, limit: int | None = None) -> str:
    """Stream ``file`` to a temp file in fixed-size chunks, never holding the whole body.

    Raises 413 as soon as the declared ``Content-Length`` or the bytes actually received
    exceed ``limit``. The caller must ``os.unlink`` the returned path.
    """
    limit = max_upload_bytes() if limit is None else limit
    declared = file.size if file.size is not None else None
    if declared is not None and declared > limit:
        raise _too_large(limit)

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise _too_large(limit)
                out.write(chunk)
    except BaseException:
        os.unlink(tmp_path)
        raise
    return tmp_path


def extract_text_from_pdf(source: Union[bytes, str, Path]) -> dict:
    """Extract text content from a PDF given as raw bytes or a filesystem path.

    Returns:
        dict with 'text', 'pages', 'page_texts', 'word_count'

    Raises HTTPException(400) when the file is not a readable PDF.
    """
    if isinstance(source, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        try:
            return extract_text_from_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)

    try:
        doc = fitz.open(str(source))
    except Exception:
        raise HTTPException(400, "The uploaded file is not a valid PDF or is corrupted. Please upload a readable PDF.")

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise HTTPException(400, "The uploaded PDF is password-protected. Please upload an unencrypted PDF.")
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
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "The uploaded file is not a valid PDF or is corrupted. Please upload a readable PDF.")
    finally:
        doc.close()

    combined = "\n\n".join(full_text)
    return {
        "text": combined,
        "pages": len(page_texts),
        "page_texts": page_texts,
        "word_count": len(combined.split()),
    }
