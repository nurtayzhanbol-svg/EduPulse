"""Privacy & data contract (council report §4 item 4): data minimisation, room scoping,
consent, delete and retention."""

from __future__ import annotations

import json
import sqlite3

import pytest

import main
import session_manager
from conftest import make_event, student_id
from models import StudentState
from telemetry import process_telemetry
from test_api import client, create, th, sh  # noqa: F401 (fixtures)
from test_socket_auth import sio_spy, setup_session, sid_of  # noqa: F401 (fixtures)



# ── Data minimisation ─────────────────────────────────────────────


@pytest.mark.parametrize("event_type", ["keystroke", "idle", "paste", "backspace", "mystery"])
def test_code_is_ignored_on_non_code_events(session, student, event_type):
    process_telemetry(session, student_id(session), make_event(event_type, current_code="x = 1", code="y = 2", length=3))
    assert student.current_code == ""


def test_help_accepts_current_code_and_code_update_accepts_code(session, student):
    process_telemetry(session, student_id(session), make_event("help", message="stuck", current_code="a = 1"))
    assert student.current_code == "a = 1"
    process_telemetry(session, student_id(session), make_event("help", message="stuck", current_answer="b = 2"))
    assert student.current_code == "b = 2"
    process_telemetry(session, student_id(session), make_event("code_update", code="c = 3"))
    assert student.current_code == "c = 3"


def test_stored_code_is_capped(session, student):
    process_telemetry(session, student_id(session), make_event("code_update", code="x" * 25_000))
    assert len(student.current_code) == StudentState.MAX_CODE_CHARS == 20_000


def test_events_ring_buffer_holds_only_type_and_ts(session, student):
    for i in range(600):
        process_telemetry(session, student_id(session), make_event("keystroke", count=1, current_code="secret", extra=i))
    assert len(student.events) == 500
    assert all(set(e) == {"type", "ts"} for e in student.events)


def test_paste_stores_length_and_timestamp_only(session, student):
    ev = make_event("paste", length=12, content_preview="import os", preview="import os")
    process_telemetry(session, student_id(session), ev)
    assert student.paste_events == [{"length": 12, "timestamp": ev.timestamp}]


def test_persisted_student_json_has_no_code_or_preview(session, student):
    process_telemetry(session, student_id(session), make_event("paste", length=400, content_preview="stolen code"))
    process_telemetry(session, student_id(session), make_event("code_update", code="print('hi')"))
    process_telemetry(session, student_id(session), make_event("help", message="?", current_code="print('hi')"))
    student.consented_at = 1234.5
    session_manager.persist_session(session)

    conn = sqlite3.connect(session_manager.db_path())
    rows = conn.execute("SELECT data FROM students WHERE session_id = ?", (session.session_id,)).fetchall()
    conn.close()
    assert rows
    raw = rows[0][0]
    for forbidden in ("current_code", "preview", "content_preview", "print('hi')", "stolen code"):
        assert forbidden not in raw
    data = json.loads(raw)
    assert data["consented_at"] == 1234.5
    assert data["paste_events"] == [{"length": 400, "timestamp": student.paste_events[0]["timestamp"]}]


def test_to_dict_never_exposes_consented_at(student):
    student.consented_at = 1.0
    assert "consented_at" not in student.to_dict()
    assert "consented_at" not in student.to_dict(include_code=True)


# ── Socket room scoping ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_private_events_reach_teacher_room_only(sio_spy):  # noqa: F811
    emitted, rooms = sio_spy
    session, teacher_token, alice_token, bob_token = setup_session()
    sid_ = session.session_id
    await main.join_room("sock-t", {"session_id": sid_, "role": "teacher", "teacher_token": teacher_token})
    alice_id, bob_id = sid_of(session, "Alice"), sid_of(session, "Bob")
    await main.join_room("sock-a", {"session_id": sid_, "role": "student",
                                    "student_id": alice_id, "student_token": alice_token})
    await main.join_room("sock-b", {"session_id": sid_, "role": "student",
                                    "student_id": bob_id, "student_token": bob_token})

    bob_rooms = {r for s, r in rooms if s == "sock-b"}
    assert bob_rooms == {sid_, main.student_room(sid_, bob_id)}
    assert {r for s, r in rooms if s == "sock-t"} == {main.teacher_room(sid_)}

    emitted.clear()
    # Alice asks for help (-> hint + hint_given), pastes a large block (-> alert).
    await main.telemetry("sock-a", {
        "session_id": sid_, "student_id": alice_id, "student_token": alice_token,
        "event": {"event_type": "help", "payload": {"message": "stuck", "current_code": "x = 1"}},
    })
    await main.telemetry("sock-a", {
        "session_id": sid_, "student_id": alice_id, "student_token": alice_token,
        "event": {"event_type": "paste", "payload": {"length": 5000}},
    })

    names = {e for e, _, _ in emitted}
    assert {"hint", "hint_given", "alert", "dashboard_update"} <= names

    teacher_room = main.teacher_room(sid_)
    alice_room = main.student_room(sid_, alice_id)
    for event, _data, kw in emitted:
        target = kw.get("room") or kw.get("to")
        if event == "hint":
            assert target == alice_room
        elif event in ("hint_given", "alert", "dashboard_update", "quiz_result"):
            assert target == teacher_room
        else:
            pytest.fail(f"unexpected event {event} -> {kw}")
        # Nothing private is ever addressed to the shared session room or to Bob.
        assert target not in (sid_, "sock-b", main.student_room(sid_, bob_id))


@pytest.mark.asyncio
async def test_quiz_result_goes_to_teacher_room(client, sio_spy):  # noqa: F811
    emitted, _ = sio_spy
    sid_ = await create(client)
    await client.post(f"/api/sessions/{sid_}/join", json={"student_name": "Alice", "consent": True})
    await client.post(f"/api/sessions/{sid_}/join", json={"student_name": "Bob", "consent": True})
    session = session_manager.get_session(sid_)
    session.quiz = [{"question": "2+2?", "options": ["3", "4"], "correct": "4", "task_description": ""}]
    session_manager.persist_session(session)
    r = await client.post(f"/api/sessions/{sid_}/submit-quiz",
                          json={"student_id": sid_of(session, "Alice"), "answers": {"0": "4"}}, headers=sh(sid_, "Alice"))
    assert r.status_code == 200, r.text
    results = [kw for e, _, kw in emitted if e == "quiz_result"]
    assert results and all(kw.get("room") == main.teacher_room(sid_) for kw in results)
    assert all(kw.get("room") == sid_ for e, _, kw in emitted if e == "quiz_available")


# ── Consent ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_join_requires_consent(client):  # noqa: F811
    sid_ = await create(client)
    for body in ({"student_name": "Alice"}, {"student_name": "Alice", "consent": False},
                 {"student_name": "Alice", "consent": "yes"}):
        r = await client.post(f"/api/sessions/{sid_}/join", json=body)
        assert r.status_code in (400, 422), (body, r.text)
    assert session_manager.get_session(sid_).students == {}

    r = await client.post(f"/api/sessions/{sid_}/join", json={"student_name": "Alice", "consent": True})
    assert r.status_code == 200
    alice = session_manager.get_session(sid_).student_by_name("Alice")
    assert isinstance(alice.consented_at, float) and alice.consented_at > 0
    assert "consented_at" in alice.to_record()

    # Consent timestamp survives a restart; the public payload never shows it.
    session_manager.reset_cache()
    assert session_manager.get_session(sid_).student_by_name("Alice").consented_at == alice.consented_at
    r = await client.get(f"/api/sessions/{sid_}")
    assert "consented_at" not in r.text


def test_session_manager_join_does_not_grant_consent():
    session, _ = session_manager.create_session("x")
    alice, _ = session_manager.join_session(session.session_id, "Alice")
    assert alice.consented_at is None


# ── Config ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_config_shape_and_provider(client, monkeypatch):  # noqa: F811
    r = await client.get("/api/config")
    assert r.status_code == 200
    assert r.json() == {"ai_provider": "none (mock hints)", "session_retention_hours": 24.0}

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("SESSION_RETENTION_HOURS", "48")
    r = await client.get("/api/config")
    assert r.json() == {"ai_provider": "OpenAI", "session_retention_hours": 48.0}

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    assert (await client.get("/api/config")).json()["ai_provider"] == "Azure OpenAI"


# ── Delete + TTL ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_session_requires_teacher_and_purges(client, sio_spy):  # noqa: F811
    emitted, _ = sio_spy
    sid_ = await create(client)
    await client.post(f"/api/sessions/{sid_}/join", json={"student_name": "Alice", "consent": True})

    assert (await client.delete(f"/api/sessions/{sid_}")).status_code == 401
    assert (await client.delete(f"/api/sessions/{sid_}", headers=sh(sid_, "Alice"))).status_code in (401, 403)
    assert (await client.delete(f"/api/sessions/{sid_}",
                                headers={"Authorization": "Bearer nope"})).status_code in (401, 403)
    assert session_manager.get_session(sid_) is not None

    r = await client.delete(f"/api/sessions/{sid_}", headers=th(sid_))
    assert r.status_code == 204 and r.content == b""
    ended = [kw for e, _, kw in emitted if e == "session_ended"]
    assert ended == [{"room": sid_}, {"room": main.teacher_room(sid_)}]

    assert (await client.get(f"/api/sessions/{sid_}")).status_code == 404
    session_manager.reset_cache()
    assert (await client.get(f"/api/sessions/{sid_}")).status_code == 404
    assert (await client.delete(f"/api/sessions/{sid_}", headers=th(sid_))).status_code == 404

    conn = sqlite3.connect(session_manager.db_path())
    assert conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    conn.close()


def test_retention_is_configurable_via_env(monkeypatch):
    monkeypatch.delenv("SESSION_RETENTION_HOURS", raising=False)
    assert session_manager.retention_hours() == 24.0
    monkeypatch.setenv("SESSION_RETENTION_HOURS", "2")
    assert session_manager.retention_seconds() == 2 * 3600
    monkeypatch.setenv("SESSION_RETENTION_HOURS", "garbage")
    assert session_manager.retention_hours() == 24.0
    monkeypatch.setenv("SESSION_RETENTION_HOURS", "0")
    assert session_manager.retention_hours() == 24.0


def test_cleanup_honours_configured_ttl_and_keeps_active(monkeypatch):
    import time

    monkeypatch.setenv("SESSION_RETENTION_HOURS", "1")
    live, _ = session_manager.create_session("live")
    ended, _ = session_manager.create_session("ended")
    session_manager.end_session(ended.session_id)
    ended.ended_at = time.time() - 2 * 3600
    session_manager.persist_session(ended)
    # An active session with an ancient ended_at must still be kept.
    live.ended_at = time.time() - 99 * 3600
    session_manager.persist_session(live)

    assert session_manager.cleanup_expired_sessions() == 1
    assert session_manager.get_session(ended.session_id) is None
    assert session_manager.get_session(live.session_id) is not None


@pytest.mark.asyncio
async def test_retention_sweeper_is_started_in_lifespan(monkeypatch):
    calls = []
    monkeypatch.setattr(session_manager, "cleanup_expired_sessions", lambda: calls.append(1) or 0)
    monkeypatch.setattr(main, "RETENTION_SWEEP_SECONDS", 0)
    import asyncio
    async with main.lifespan(main.app):
        await asyncio.sleep(0.01)
    assert calls
