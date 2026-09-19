"""Teacher dashboard resume: GET /api/sessions/{id} with the teacher token, plus the
bundled sample PDFs under samples/.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import pytest_asyncio

import main
import session_manager
from main import app
from test_api import _remember_tokens, create, th

pytestmark = pytest.mark.asyncio

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", event_hooks={"response": [_remember_tokens]},
    ) as c:
        yield c


@pytest.fixture
def sio_spy(monkeypatch):
    emitted: list[tuple[str, dict, dict]] = []
    rooms: list[tuple[str, str]] = []

    async def fake_emit(event, data=None, **kwargs):
        emitted.append((event, data, kwargs))

    async def fake_enter_room(sid, room):
        rooms.append((sid, room))

    monkeypatch.setattr(main.sio, "emit", fake_emit)
    monkeypatch.setattr(main.sio, "enter_room", fake_enter_room)
    main._student_sockets.clear()
    return emitted, rooms


# ── GET /api/sessions/{id} as the teacher ─────────────────────────


async def test_public_get_session_is_unchanged_without_token(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    session_manager.get_session(sid).students["Alice"].current_code = "print('hi')\nprint('there')"

    r = await client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert "role" not in body
    assert "current_code" not in body["students"]["Alice"]
    assert body["students"]["Alice"]["current_code_lines"] == 2


async def test_teacher_token_gets_full_dashboard_payload(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Bob"})
    session_manager.get_session(sid).students["Alice"].current_code = "x = 1"

    r = await client.get(f"/api/sessions/{sid}", headers=th(sid))
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "teacher"
    assert body["active"] is True
    assert set(body["students"]) == {"Alice", "Bob"}
    assert body["students"]["Alice"]["current_code"] == "x = 1"


async def test_wrong_teacher_token_is_401(client):
    sid = await create(client)
    other = await create(client)
    r = await client.get(f"/api/sessions/{sid}", headers=th(other))
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid teacher token"

    r = await client.get(f"/api/sessions/{sid}", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401


async def test_student_token_cannot_read_teacher_payload(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    student_token = r.json()["student_token"]
    r = await client.get(f"/api/sessions/{sid}", headers={"Authorization": f"Bearer {student_token}"})
    assert r.status_code == 401


async def test_unknown_session_is_404_with_or_without_token(client):
    r = await client.get("/api/sessions/does-not-exist")
    assert r.status_code == 404
    r = await client.get("/api/sessions/does-not-exist", headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 404
    assert r.json()["detail"] == "Session not found"


async def test_ended_session_resumes_as_inactive(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/end", headers=th(sid))
    assert r.status_code == 200
    r = await client.get(f"/api/sessions/{sid}", headers=th(sid))
    assert r.status_code == 200
    assert r.json()["active"] is False


async def test_resume_survives_server_restart(client):
    """Session + teacher token still resolve after the in-memory cache is dropped."""
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    session_manager.reset_cache()

    r = await client.get(f"/api/sessions/{sid}", headers=th(sid))
    assert r.status_code == 200
    assert r.json()["role"] == "teacher"
    assert list(r.json()["students"]) == ["Alice"]


async def test_teacher_only_endpoints_still_require_token(client):
    sid = await create(client)
    for path in (f"/api/sessions/{sid}/end", f"/api/sessions/{sid}/generate-quiz"):
        r = await client.post(path)
        assert r.status_code == 401, path
    r = await client.get(f"/api/sessions/{sid}/report")
    assert r.status_code == 401


async def test_teacher_can_rejoin_socket_room_after_reload(client, sio_spy):
    """The token a reloaded dashboard restores from sessionStorage re-enters the teacher room."""
    emitted, rooms = sio_spy
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    token = th(sid)["Authorization"].split(" ", 1)[1]

    await main.join_room("sid-t1", {"session_id": sid, "role": "teacher", "teacher_token": token})
    await main.join_room("sid-t2", {"session_id": sid, "role": "teacher", "teacher_token": token})
    assert rooms == [
        ("sid-t1", sid), ("sid-t1", main.teacher_room(sid)),
        ("sid-t2", sid), ("sid-t2", main.teacher_room(sid)),
    ]
    updates = [(d, kw) for e, d, kw in emitted if e == "dashboard_update"]
    assert len(updates) == 2
    assert updates[-1][1] == {"to": "sid-t2"}
    assert "current_code" in updates[-1][0]["students"]["Alice"]


# ── samples/ fixtures ─────────────────────────────────────────────


async def test_sample_files_exist():
    for name in ("sample_assignment.pdf", "sample_assignment.txt", "long_sample.pdf", "too_short.pdf"):
        assert (SAMPLES_DIR / name).is_file(), name


async def test_sample_assignment_pdf_creates_a_session(client):
    pdf = (SAMPLES_DIR / "sample_assignment.pdf").read_bytes()
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("sample_assignment.pdf", pdf, "application/pdf")},
        data={"task_level": "medium", "mode": "practical"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["word_count"] >= 200
    assert body["pages"] >= 2
    assert body["teacher_token"]


async def test_long_sample_pdf_meets_the_minimum(client):
    pdf = (SAMPLES_DIR / "long_sample.pdf").read_bytes()
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("long_sample.pdf", pdf, "application/pdf")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["word_count"] >= 20


async def test_too_short_pdf_is_rejected(client):
    pdf = (SAMPLES_DIR / "too_short.pdf").read_bytes()
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("too_short.pdf", pdf, "application/pdf")},
    )
    assert r.status_code == 400
    assert "Could not extract enough text" in r.json()["detail"]
