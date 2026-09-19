"""The teacher dashboard's Live Code View gets the student's full code; nobody else does."""

from __future__ import annotations

import httpx
import pytest

import main
import session_manager
from main import app

pytestmark = pytest.mark.asyncio

CODE = "def total(xs):\n    return sum(xs)\n"


@pytest.fixture
def sio_spy(monkeypatch):
    emitted: list[tuple[str, dict, dict]] = []

    async def fake_emit(event, data=None, **kwargs):
        emitted.append((event, data, kwargs))

    async def fake_enter_room(sid, room):
        pass

    monkeypatch.setattr(main.sio, "emit", fake_emit)
    monkeypatch.setattr(main.sio, "enter_room", fake_enter_room)
    main._student_sockets.clear()
    return emitted


def dashboards(emitted):
    return [(d, kw) for e, d, kw in emitted if e == "dashboard_update"]


async def join_student(session_id, student, token):
    await main.join_room(f"sid-{student.name}", {
        "session_id": session_id, "role": "student",
        "student_id": student.student_id, "student_token": token,
    })


async def test_code_update_reaches_the_teacher_dashboard(sio_spy):
    emitted = sio_spy
    session, _ = session_manager.create_session("Sum a list", "easy")
    alice, token = session_manager.join_session(session.session_id, "Alice")
    await join_student(session.session_id, alice, token)

    await main.telemetry("sid-Alice", {
        "session_id": session.session_id,
        "student_id": alice.student_id,
        "student_token": token,
        "event": {"event_type": "code_update", "payload": {"code": CODE}},
    })

    payload, kwargs = dashboards(emitted)[-1]
    assert payload["students"][alice.student_id]["current_code"] == CODE
    assert kwargs == {"room": main.teacher_room(session.session_id)}


async def test_dashboard_with_code_is_not_broadcast_to_the_student_room(sio_spy):
    emitted = sio_spy
    session, _ = session_manager.create_session("Sum a list", "easy")
    alice, token = session_manager.join_session(session.session_id, "Alice")
    await join_student(session.session_id, alice, token)

    rooms = {kw.get("room") for _, kw in dashboards(emitted)}
    assert rooms == {main.teacher_room(session.session_id)}
    assert session.session_id not in rooms


async def test_unauthenticated_session_endpoint_omits_code():
    session, _ = session_manager.create_session("Sum a list", "easy")
    alice, _ = session_manager.join_session(session.session_id, "Alice")
    alice.current_code = CODE

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get(f"/api/sessions/{session.session_id}")
    payload = r.json()["students"][alice.student_id]
    assert payload["name"] == "Alice"
    assert "current_code" not in payload
    assert payload["current_code_lines"] == 3
