"""Teacher control (council report §4 items 4 and 6): launch gate after task review,
one-student teacher nudges, and student-socket isolation."""

from __future__ import annotations

import json
import sqlite3

import pytest

import main
import session_manager
from models import StudentState
from test_api import client, create, th, sh, make_pdf, LONG_PARAGRAPHS  # noqa: F401 (fixtures)
from test_socket_auth import sio_spy, setup_session, sid_of, errors  # noqa: F401 (fixtures)

pytestmark = pytest.mark.asyncio


async def create_from_pdf(client) -> dict:  # noqa: F811
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
        data={"task_level": "medium", "mode": "practical", "quiz_difficulty": "medium"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def student_payload(name: str) -> dict:
    return {"student_name": name, "consent": True}


# ── Launch gate ───────────────────────────────────────────────────


async def test_pdf_session_starts_unlaunched_and_rejects_joins(client):  # noqa: F811
    body = await create_from_pdf(client)
    sid = body["session_id"]
    assert body["launched"] is False
    assert (await client.get(f"/api/sessions/{sid}")).json()["launched"] is False

    r = await client.post(f"/api/sessions/{sid}/join", json=student_payload("Ann"))
    assert r.status_code == 409
    assert "not launched" in r.json()["detail"]
    assert session_manager.get_session(sid).students == {}


async def test_manual_session_is_launched_immediately(client):  # noqa: F811
    sid = await create(client)
    assert (await client.get(f"/api/sessions/{sid}")).json()["launched"] is True
    r = await client.post(f"/api/sessions/{sid}/join", json=student_payload("Ann"))
    assert r.status_code == 200


async def test_launch_saves_final_task_and_opens_joins(client):  # noqa: F811
    body = await create_from_pdf(client)
    sid = body["session_id"]

    r = await client.post(f"/api/sessions/{sid}/launch", headers=th(sid),
                          json={"task_description": "  Task: reviewed by the teacher  "})
    assert r.status_code == 200, r.text
    assert r.json() == {
        "launched": True,
        "task_description": "Task: reviewed by the teacher",
        "join_url": f"/student.html?session={sid}",
    }
    state = (await client.get(f"/api/sessions/{sid}")).json()
    assert state["launched"] is True
    assert state["task_description"] == "Task: reviewed by the teacher"

    r = await client.post(f"/api/sessions/{sid}/join", json=student_payload("Ann"))
    assert r.status_code == 200, r.text

    # The launch survives a cache reset (persisted).
    session_manager.reset_cache()
    assert session_manager.get_session(sid).launched is True


async def test_launch_without_task_keeps_generated_description(client):  # noqa: F811
    body = await create_from_pdf(client)
    sid = body["session_id"]
    r = await client.post(f"/api/sessions/{sid}/launch", headers=th(sid), json={})
    assert r.status_code == 200
    assert r.json()["task_description"] == body["task_description"]


async def test_launch_requires_teacher_and_non_empty_task(client):  # noqa: F811
    sid = (await create_from_pdf(client))["session_id"]
    r = await client.post(f"/api/sessions/{sid}/launch", json={})
    assert r.status_code == 401
    r = await client.post(f"/api/sessions/{sid}/launch", headers={"Authorization": "Bearer nope"}, json={})
    assert r.status_code == 403
    r = await client.post(f"/api/sessions/{sid}/launch", headers=th(sid), json={"task_description": "   "})
    assert r.status_code == 400
    assert session_manager.get_session(sid).launched is False
    r = await client.post("/api/sessions/nope/launch", json={})
    assert r.status_code == 404


async def test_launch_after_end_is_409(client):  # noqa: F811
    sid = (await create_from_pdf(client))["session_id"]
    assert (await client.post(f"/api/sessions/{sid}/end", headers=th(sid))).status_code == 200
    r = await client.post(f"/api/sessions/{sid}/launch", headers=th(sid), json={})
    assert r.status_code == 409


async def test_socket_join_is_refused_before_launch(sio_spy):  # noqa: F811
    emitted, rooms = sio_spy
    session, teacher_token, alice_token, _ = setup_session()
    session.launched = False
    alice_id = sid_of(session, "Alice")
    await main.join_room("sock-a", {"session_id": session.session_id, "role": "student",
                                    "student_id": alice_id, "student_token": alice_token})
    assert errors(emitted) == ["Your teacher has not launched this session yet."]
    assert rooms == []
    assert "sock-a" not in main._student_sockets

    # The teacher can still connect to review/launch.
    emitted.clear()
    await main.join_room("sock-t", {"session_id": session.session_id, "role": "teacher",
                                    "teacher_token": teacher_token})
    assert errors(emitted) == []
    assert rooms == [("sock-t", main.teacher_room(session.session_id))]


def test_records_without_launched_field_load_as_launched():
    session, _ = session_manager.create_session("Old session", "easy")
    session_manager.persist_session(session)
    conn = sqlite3.connect(session_manager.db_path())
    (raw,) = conn.execute("SELECT data FROM sessions WHERE session_id = ?", (session.session_id,)).fetchone()
    data = json.loads(raw)
    assert data.pop("launched") is True
    conn.execute("UPDATE sessions SET data = ? WHERE session_id = ?", (json.dumps(data), session.session_id))
    conn.commit()
    conn.close()
    session_manager.reset_cache()
    assert session_manager.get_session(session.session_id).launched is True


# ── Nudge ─────────────────────────────────────────────────────────


async def test_nudge_reaches_only_the_target_student_and_is_logged(client, sio_spy):  # noqa: F811
    emitted, _rooms = sio_spy
    sid = await create(client)
    ann = (await client.post(f"/api/sessions/{sid}/join", json=student_payload("Ann"))).json()["student_id"]
    bob = (await client.post(f"/api/sessions/{sid}/join", json=student_payload("Bob"))).json()["student_id"]
    emitted.clear()

    r = await client.post(f"/api/sessions/{sid}/students/{ann}/nudge", headers=th(sid),
                          json={"message": "  You're doing fine,\n keep going  "})
    assert r.status_code == 200, r.text
    assert r.json() == {"student_id": ann, "message": "You're doing fine, keep going", "teacher_nudges": 1}

    messages = [(d, kw) for e, d, kw in emitted if e == "teacher_message"]
    assert len(messages) == 1
    data, kw = messages[0]
    assert data["message"] == "You're doing fine, keep going"
    assert kw["room"] == main.student_room(sid, ann)
    assert kw["room"] != main.student_room(sid, bob)
    for event, _d, kw in emitted:
        target = kw.get("room") or kw.get("to")
        assert target not in (sid, main.student_room(sid, bob)), event
        if event == "dashboard_update":
            assert target == main.teacher_room(sid)

    session = session_manager.get_session(sid)
    assert len(session.students[ann].teacher_nudges) == 1
    assert session.students[bob].teacher_nudges == []
    assert session.to_dict()["students"][ann]["teacher_nudges_count"] == 1

    # Persisted and counted in the report.
    session_manager.reset_cache()
    assert len(session_manager.get_session(sid).students[ann].teacher_nudges) == 1
    assert (await client.post(f"/api/sessions/{sid}/end", headers=th(sid))).status_code == 200
    report = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()
    assert report["analytics"]["teacher_intervention_count"] == 1
    by_name = {s["name"]: s for s in report["students"]}
    assert by_name["Ann"]["teacher_nudges"] == 1
    assert by_name["Bob"]["teacher_nudges"] == 0


async def test_nudge_auth_and_validation(client):  # noqa: F811
    sid = await create(client)
    ann = (await client.post(f"/api/sessions/{sid}/join", json=student_payload("Ann"))).json()["student_id"]
    url = f"/api/sessions/{sid}/students/{ann}/nudge"

    assert (await client.post(url, json={"message": "hi"})).status_code == 401
    assert (await client.post(url, headers=sh(sid, "Ann"), json={"message": "hi"})).status_code == 403
    assert (await client.post(url, headers=th(sid), json={"message": "   "})).status_code == 400
    assert (await client.post(url, headers=th(sid), json={"message": "x" * 281})).status_code == 422
    assert (await client.post(url, headers=th(sid), json={})).status_code == 422
    assert (await client.post(f"/api/sessions/{sid}/students/ghost/nudge", headers=th(sid),
                              json={"message": "hi"})).status_code == 404
    assert (await client.post("/api/sessions/nope/students/x/nudge", headers=th(sid),
                              json={"message": "hi"})).status_code == 404
    assert session_manager.get_session(sid).students[ann].teacher_nudges == []

    assert (await client.post(f"/api/sessions/{sid}/end", headers=th(sid))).status_code == 200
    assert (await client.post(url, headers=th(sid), json={"message": "hi"})).status_code == 409


def test_teacher_nudges_are_bounded():
    student = StudentState(name="Ann")
    for i in range(StudentState.MAX_TEACHER_NUDGES + 10):
        student.log_teacher_nudge(f"m{i}", float(i))
    assert len(student.teacher_nudges) == StudentState.MAX_TEACHER_NUDGES
    assert student.teacher_nudges[0]["message"] == "m10"
    assert student.to_dict()["teacher_nudges_count"] == StudentState.MAX_TEACHER_NUDGES


# ── Student isolation ─────────────────────────────────────────────


async def test_student_socket_never_receives_another_students_name_or_state(client, sio_spy):  # noqa: F811
    """Across a lesson (joins, help, paste, idle, code, quiz, nudge, task edit, end), everything
    Bob's socket could receive is inspected: it must never mention Alice or carry class state."""
    emitted, rooms = sio_spy
    sid = await create(client)
    session = session_manager.get_session(sid)
    alice_id = (await client.post(f"/api/sessions/{sid}/join", json=student_payload("Alice"))).json()["student_id"]
    bob_id = (await client.post(f"/api/sessions/{sid}/join", json=student_payload("Bob"))).json()["student_id"]
    teacher_token = th(sid)["Authorization"].split()[1]
    alice_token = sh(sid, "Alice")["Authorization"].split()[1]
    bob_token = sh(sid, "Bob")["Authorization"].split()[1]
    session.quiz = [{"question": "2+2?", "options": ["3", "4"], "correct": "4", "task_description": "sum"}]

    await main.join_room("sock-t", {"session_id": sid, "role": "teacher", "teacher_token": teacher_token})
    await main.join_room("sock-a", {"session_id": sid, "role": "student",
                                    "student_id": alice_id, "student_token": alice_token})
    await main.join_room("sock-b", {"session_id": sid, "role": "student",
                                    "student_id": bob_id, "student_token": bob_token})

    def alice_event(event_type, **payload):
        return {"session_id": sid, "student_id": alice_id, "student_token": alice_token,
                "event": {"event_type": event_type, "payload": payload}}

    await main.telemetry("sock-a", alice_event("keystroke", count=3))
    await main.telemetry("sock-a", alice_event("help", message="stuck on the loop", current_code="for x in xs:"))
    await main.telemetry("sock-a", alice_event("paste", length=5000))
    await main.telemetry("sock-a", alice_event("idle", idle_seconds=300))
    await main.telemetry("sock-a", alice_event("code_update", code="print('alice secret')"))
    r = await client.post(f"/api/sessions/{sid}/students/{alice_id}/nudge", headers=th(sid),
                          json={"message": "Keep going Alice"})
    assert r.status_code == 200
    r = await client.post(f"/api/sessions/{sid}/submit-quiz", headers=sh(sid, "Alice"),
                          json={"student_id": alice_id, "answers": {"0": "4"}})
    assert r.status_code == 200, r.text
    r = await client.patch(f"/api/sessions/{sid}/task", headers=th(sid), json={"task_description": "Task: v2"})
    assert r.status_code == 200
    assert (await client.post(f"/api/sessions/{sid}/end", headers=th(sid))).status_code == 200

    bob_targets = {"sock-b", sid, main.student_room(sid, bob_id)}
    bob_received = [(e, d) for e, d, kw in emitted if (kw.get("room") or kw.get("to")) in bob_targets]
    # Bob does receive the class-wide events, so the privacy check below is not vacuous.
    assert {e for e, _ in bob_received} == {"quiz_available", "task_updated", "session_ended"}
    for event, data in bob_received:
        raw = json.dumps(data).lower()
        assert "alice" not in raw and alice_id.lower() not in raw, (event, data)
        for forbidden in ("students", "frustration", "status", "current_code", "alert", "secret", "for x in xs"):
            assert forbidden not in raw, (event, forbidden)

    # And everything about Alice went to her own room or the teacher room.
    for event, data, kw in emitted:
        target = kw.get("room") or kw.get("to")
        if event in ("quiz_available", "task_updated", "session_ended"):
            continue
        assert target in ("sock-t", main.teacher_room(sid), main.student_room(sid, alice_id)), (event, kw)
