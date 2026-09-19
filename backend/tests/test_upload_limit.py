"""Upload size cap (413), corrupt PDFs (400) and streamed temp-file writes."""

from __future__ import annotations

import io

import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

import pdf_engine
from main import app
from test_api import LONG_PARAGRAPHS, _remember_tokens, create, make_pdf, th


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", event_hooks={"response": [_remember_tokens]},
    ) as c:
        yield c

def test_default_limit_is_10mb_and_env_overrides(monkeypatch):
    monkeypatch.delenv("EDUPULSE_MAX_UPLOAD_MB", raising=False)
    assert pdf_engine.max_upload_bytes() == 10 * 1024 * 1024
    monkeypatch.setenv("EDUPULSE_MAX_UPLOAD_MB", "2")
    assert pdf_engine.max_upload_bytes() == 2 * 1024 * 1024
    monkeypatch.setenv("EDUPULSE_MAX_UPLOAD_MB", "garbage")
    assert pdf_engine.max_upload_bytes() == 10 * 1024 * 1024


@pytest.mark.asyncio
async def test_declared_content_length_over_limit_is_413_before_body_is_read(monkeypatch):
    """The middleware rejects on the Content-Length header alone; the body is never consumed."""
    monkeypatch.setenv("EDUPULSE_MAX_UPLOAD_MB", "1")
    body_read = False

    async def body_stream():
        nonlocal body_read
        body_read = True
        yield b"x"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/api/sessions/create-from-pdf",
            content=body_stream(),
            headers={
                "content-type": "multipart/form-data; boundary=abc",
                "content-length": str(3 * 1024 * 1024),
            },
        )
    assert r.status_code == 413
    assert r.json()["detail"] == "PDF is too large. Maximum upload size is 1 MB."
    assert body_read is False


@pytest.mark.asyncio
async def test_upload_over_limit_is_413_json(client, monkeypatch):
    monkeypatch.setenv("EDUPULSE_MAX_UPLOAD_MB", "1")
    sid = await create(client)
    big = make_pdf(LONG_PARAGRAPHS) + b"\n%" + b"0" * (2 * 1024 * 1024)
    r = await client.post(
        f"/api/sessions/{sid}/upload-pdf",
        files={"file": ("big.pdf", big, "application/pdf")},
        headers=th(sid),
    )
    assert r.status_code == 413
    assert r.json() == {"detail": "PDF is too large. Maximum upload size is 1 MB."}


@pytest.mark.asyncio
async def test_create_from_pdf_over_limit_is_413(client, monkeypatch):
    monkeypatch.setenv("EDUPULSE_MAX_UPLOAD_MB", "1")
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("big.pdf", b"%PDF-1.4" + b"0" * (2 * 1024 * 1024), "application/pdf")},
    )
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_upload_under_limit_still_works(client, monkeypatch):
    monkeypatch.setenv("EDUPULSE_MAX_UPLOAD_MB", "1")
    sid = await create(client)
    r = await client.post(
        f"/api/sessions/{sid}/upload-pdf",
        files={"file": ("ok.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
        headers=th(sid),
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_streamed_save_rejects_when_bytes_exceed_limit_without_content_length(tmp_path):
    """A chunked body (no size known up front) is cut off as soon as the cap is crossed."""
    data = io.BytesIO(b"a" * (pdf_engine.UPLOAD_CHUNK_BYTES * 3))
    upload = UploadFile(file=data, filename="x.pdf", headers=Headers({"content-type": "application/pdf"}))
    assert upload.size is None
    with pytest.raises(HTTPException) as exc:
        await pdf_engine.save_upload_to_tempfile(upload, limit=pdf_engine.UPLOAD_CHUNK_BYTES * 2)
    assert exc.value.status_code == 413
    assert data.tell() <= pdf_engine.UPLOAD_CHUNK_BYTES * 3  # stopped before the end at the latest
    # Nothing left behind in the temp dir.
    import glob, tempfile  # noqa: E401
    leftovers = [p for p in glob.glob(f"{tempfile.gettempdir()}/*.pdf") if open(p, "rb").read(1) == b"a"]
    assert leftovers == []


@pytest.mark.asyncio
async def test_streamed_save_writes_in_chunks():
    payload = make_pdf(LONG_PARAGRAPHS)
    reads: list[int] = []

    class Spy(UploadFile):
        async def read(self, size: int = -1) -> bytes:  # type: ignore[override]
            reads.append(size)
            return await super().read(size)

    upload = Spy(file=io.BytesIO(payload), filename="x.pdf")
    path = await pdf_engine.save_upload_to_tempfile(upload)
    try:
        assert all(r == pdf_engine.UPLOAD_CHUNK_BYTES for r in reads)
        assert open(path, "rb").read() == payload
        assert pdf_engine.extract_text_from_pdf(path)["word_count"] >= 20
    finally:
        import os
        os.unlink(path)


# ── corrupt PDFs ──────────────────────────────────────────────────


def test_extract_text_from_corrupt_bytes_raises_400():
    with pytest.raises(HTTPException) as exc:
        pdf_engine.extract_text_from_pdf(b"%PDF-1.7 this is not really a pdf \x00\x01\x02")
    assert exc.value.status_code == 400
    assert "not a valid PDF" in exc.value.detail


def test_extract_text_from_valid_bytes_still_works():
    result = pdf_engine.extract_text_from_pdf(make_pdf(LONG_PARAGRAPHS))
    assert result["pages"] == 1 and result["word_count"] >= 20


@pytest.mark.asyncio
async def test_corrupt_pdf_upload_is_400_json(client):
    sid = await create(client)
    r = await client.post(
        f"/api/sessions/{sid}/upload-pdf",
        files={"file": ("bad.pdf", b"%PDF-1.4\ngarbage garbage garbage", "application/pdf")},
        headers=th(sid),
    )
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert "not a valid PDF" in r.json()["detail"]


@pytest.mark.asyncio
async def test_corrupt_pdf_create_from_pdf_is_400_json(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("bad.pdf", b"\x00" * 512, "application/pdf")},
    )
    assert r.status_code == 400
    assert "not a valid PDF" in r.json()["detail"]
