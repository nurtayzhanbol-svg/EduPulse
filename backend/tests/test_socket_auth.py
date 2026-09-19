"""Auth checks on the Socket.IO ``join_room`` and ``telemetry`` handlers.

The handlers are plain coroutines, so they are driven directly with fake sids while
``sio.emit`` / ``sio.enter_room`` are captured.
"""

from __future__ import annotations

import pytest

import main
import telemetry
import session_manager

pytestmark = pytest.mark.asyncio


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


def errors(emitted):
    return [d["message"] for e, d, _ in emitted if e == "error"]


def setup_session():
    session, teacher_token = session_manager.create_session("Sum a list", "easy")
    alice, alice_token = session_manager.join_session(session.session_id, "Alice")
    bob, bob_token = session_manager.join_session(session.session_id, "Bob")
    return session, teacher_token, alice_token, bob_token


def sid_of(session, name):
    return session.student_by_name(name).student_id


def keystroke(session_id, student_id, token=None, **payload):
    data = {
        "session_id": session_id,
        "student_id": student_id,
        "event": {"event_type": "keystroke", "payload": payload or {"count": 1}},
    }
    if token is not None:
        data["student_token"] = token
    return data


# ── join_room ─────────────────────────────────────────────────────


async def test_student_join_room_with_valid_token_enters_room_and_binds_sid(sio_spy):
    emitted, rooms = sio_spy
    session, _, alice_token, _ = setup_session()
    await main.join_room("sid-a", {
        "session_id": session.session_id, "role": "student",
        "student_id": sid_of(session, "Alice"), "student_token": alice_token,
    })
    assert rooms == [
        ("sid-a", session.session_id),
        ("sid-a", main.student_room(session.session_id, sid_of(session, "Alice"))),
    ]
    assert main._student_sockets["sid-a"] == (session.session_id, sid_of(session, "Alice"))
    assert session.student_by_name("Alice").sid == "sid-a"
    assert errors(emitted) == []
    assert [e for e, _, _ in emitted] == ["dashboard_update"]


async def test_student_join_room_without_token_is_rejected(sio_spy):
    emitted, rooms = sio_spy
    session, *_ = setup_session()
    await main.join_room("sid-a", {"session_id": session.session_id, "role": "student", "student_id": sid_of(session, "Alice")})
    assert rooms == []
    assert "sid-a" not in main._student_sockets
    assert errors(emitted) == ["Invalid student token"]


async def test_student_join_room_with_another_students_token_is_rejected(sio_spy):
    emitted, rooms = sio_spy
    session, _, _, bob_token = setup_session()
    await main.join_room("sid-a", {
        "session_id": session.session_id, "role": "student",
        "student_id": sid_of(session, "Alice"), "student_token": bob_token,
    })
    assert rooms == []
    assert errors(emitted) == ["Invalid student token"]


async def test_join_room_cannot_invent_a_student(sio_spy):
    emitted, rooms = sio_spy
    session, _, alice_token, _ = setup_session()
    await main.join_room("sid-x", {
        "session_id": session.session_id, "role": "student",
        "student_id": "mallory-does-not-exist", "student_token": alice_token,
    })
    assert rooms == []
    assert session.student_by_name("Mallory") is None


async def test_teacher_join_room_requires_teacher_token(sio_spy):
    emitted, rooms = sio_spy
    session, teacher_token, alice_token, _ = setup_session()

    await main.join_room("sid-t", {"session_id": session.session_id, "role": "teacher"})
    await main.join_room("sid-t", {"session_id": session.session_id, "role": "teacher", "teacher_token": alice_token})
    assert rooms == []
    assert errors(emitted) == ["Invalid teacher token", "Invalid teacher token"]

    await main.join_room("sid-t", {"session_id": session.session_id, "role": "teacher", "teacher_token": teacher_token})
    assert rooms == [("sid-t", main.teacher_room(session.session_id))]
    assert emitted[-1][0] == "dashboard_update"
    assert emitted[-1][2] == {"to": "sid-t"}


async def test_join_room_unknown_session(sio_spy):
    emitted, rooms = sio_spy
    await main.join_room("sid-a", {"session_id": "nope", "role": "student", "student_id": "whoever", "student_token": "x"})
    assert rooms == []
    assert errors(emitted) == ["Session not found"]


# ── telemetry ─────────────────────────────────────────────────────


async def test_telemetry_from_bound_socket_is_processed(sio_spy):
    emitted, _ = sio_spy
    session, _, alice_token, _ = setup_session()
    await main.join_room("sid-a", {
        "session_id": session.session_id, "role": "student",
        "student_id": sid_of(session, "Alice"), "student_token": alice_token,
    })
    await main.telemetry("sid-a", keystroke(session.session_id, sid_of(session, "Alice"), alice_token, count=3))
    assert session.student_by_name("Alice").total_keystrokes == 3
    assert errors(emitted) == []


async def test_telemetry_without_join_room_is_rejected(sio_spy):
    emitted, _ = sio_spy
    session, _, alice_token, _ = setup_session()
    await main.telemetry("sid-unbound", keystroke(session.session_id, sid_of(session, "Alice"), alice_token, count=3))
    assert session.student_by_name("Alice").total_keystrokes == 0
    assert errors(emitted) == ["Unauthorized telemetry"]


async def test_telemetry_with_mismatched_student_id_is_rejected(sio_spy):
    emitted, _ = sio_spy
    session, _, alice_token, bob_token = setup_session()
    await main.join_room("sid-b", {
        "session_id": session.session_id, "role": "student",
        "student_id": sid_of(session, "Bob"), "student_token": bob_token,
    })
    # Bob's socket claims to be Alice (with Bob's own token).
    await main.telemetry("sid-b", keystroke(session.session_id, sid_of(session, "Alice"), bob_token, count=5))
    # ...and even with Alice's stolen token, the socket is bound to Bob.
    await main.telemetry("sid-b", keystroke(session.session_id, sid_of(session, "Alice"), alice_token, count=5))
    assert session.student_by_name("Alice").total_keystrokes == 0
    assert session.student_by_name("Bob").total_keystrokes == 0
    assert errors(emitted) == ["Unauthorized telemetry", "Unauthorized telemetry"]


async def test_telemetry_with_wrong_token_on_bound_socket_is_rejected(sio_spy):
    emitted, _ = sio_spy
    session, _, alice_token, bob_token = setup_session()
    await main.join_room("sid-a", {
        "session_id": session.session_id, "role": "student",
        "student_id": sid_of(session, "Alice"), "student_token": alice_token,
    })
    await main.telemetry("sid-a", keystroke(session.session_id, sid_of(session, "Alice"), bob_token, count=2))
    await main.telemetry("sid-a", keystroke(session.session_id, sid_of(session, "Alice"), count=2))
    assert session.student_by_name("Alice").total_keystrokes == 0
    assert errors(emitted) == ["Invalid student token", "Invalid student token"]


async def test_telemetry_for_other_session_is_rejected(sio_spy):
    emitted, _ = sio_spy
    session, _, alice_token, _ = setup_session()
    other, _, other_alice_token, _ = setup_session()
    await main.join_room("sid-a", {
        "session_id": session.session_id, "role": "student",
        "student_id": sid_of(session, "Alice"), "student_token": alice_token,
    })
    await main.telemetry("sid-a", keystroke(other.session_id, sid_of(other, "Alice"), other_alice_token, count=2))
    assert other.student_by_name("Alice").total_keystrokes == 0
    assert errors(emitted) == ["Unauthorized telemetry"]


async def test_disconnect_clears_socket_binding(sio_spy):
    _, _ = sio_spy
    session, _, alice_token, _ = setup_session()
    await main.join_room("sid-a", {
        "session_id": session.session_id, "role": "student",
        "student_id": sid_of(session, "Alice"), "student_token": alice_token,
    })
    await main.disconnect("sid-a")
    assert "sid-a" not in main._student_sockets


# ── confusion spike dedup ─────────────────────────────────────────


def _spike_alerts(emitted):
    return [d for e, d, _ in emitted if e == "alert" and d.get("type") == "confusion_spike"]


async def test_confusion_spike_alert_fires_once_per_episode(sio_spy, monkeypatch):
    emitted, _ = sio_spy
    session, _, alice_token, _ = setup_session()
    for name in ("Carol", "Dave"):
        session_manager.join_session(session.session_id, name)
    for name in ("Bob", "Carol", "Dave"):
        session.student_by_name(name).status = "yellow"
    await main.join_room("sid-a", {
        "session_id": session.session_id, "role": "student",
        "student_id": sid_of(session, "Alice"), "student_token": alice_token,
    })

    def send():
        return main.telemetry("sid-a", keystroke(session.session_id, sid_of(session, "Alice"), alice_token, count=1))

    # The other three stay yellow, so the episode is detected on every event; only
    # its start may alert.
    await send()
    await send()
    assert len(_spike_alerts(emitted)) == 1
    assert len([a for a in session.alerts if a["type"] == "confusion_spike"]) == 1

    # A large-paste alert in between does not restart the episode.
    await main.telemetry("sid-a", {
        "session_id": session.session_id, "student_id": sid_of(session, "Alice"), "student_token": alice_token,
        "event": {"event_type": "paste", "payload": {"length": 500}},
    })
    await send()
    assert len(_spike_alerts(emitted)) == 1

    # The episode ends when the class recovers, and a fresh one alerts again.
    for name in ("Bob", "Carol", "Dave"):
        session.student_by_name(name).status = "green"
    await send()
    assert session.confusion_episode_active is False
    session.confusion_episode_ended_at -= telemetry.CONFUSION_SPIKE_QUIET_SECONDS + 1
    for name in ("Bob", "Carol", "Dave"):
        session.student_by_name(name).status = "yellow"
    await send()
    assert len(_spike_alerts(emitted)) == 2
    assert len([a for a in session.alerts if a["type"] == "confusion_spike"]) == 2
