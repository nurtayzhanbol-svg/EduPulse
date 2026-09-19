"""SQLite persistence: state survives a simulated restart, cleanup, student ids."""

from __future__ import annotations

import os
import sqlite3
import time

import session_manager
from models import TelemetryEvent
from telemetry import process_telemetry


def restart() -> None:
    """Simulate a process restart: drop every in-memory object and DB handle."""
    session_manager.reset_cache()
    assert session_manager._sessions == {}


def test_db_path_comes_from_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom.db"
    monkeypatch.setenv("EDUPULSE_DB", str(custom))
    session_manager.reset_cache()
    session_manager.create_session("x")
    assert custom.exists()
    monkeypatch.delenv("EDUPULSE_DB")
    assert session_manager.db_path() == "./edupulse.db"


def test_session_and_students_survive_restart():
    session, teacher_token = session_manager.create_session("Sum a list", "hard")
    sid = session.session_id
    alice, alice_token = session_manager.join_session(sid, "Alice")
    alice.hints_given = 2
    alice.current_code = "print(1)\nprint(2)"
    alice.sid = "socket-123"
    session.pdf_text = "material"
    session.quiz = [{"question": "q", "options": ["a"], "correct": "a"}]
    session_manager.persist_session(session)

    restart()

    loaded = session_manager.get_session(sid)
    assert loaded is not None and loaded is not session
    assert loaded.task_level == "hard"
    assert loaded.pause_threshold_seconds == 120
    assert loaded.teacher_token_hash == session.teacher_token_hash
    assert loaded.pdf_text == "material"
    assert loaded.quiz == session.quiz
    assert set(loaded.students) == {alice.student_id}
    a = loaded.students[alice.student_id]
    assert a.student_id == alice.student_id
    assert a.hints_given == 2
    assert a.current_code == ""  # code is live-only, never persisted
    assert a.sid is None  # transient, never persisted

    # Tokens still work after the restart.
    assert session_manager.authenticate_student(sid, alice.student_id, alice_token) is a
    assert session_manager.list_sessions(teacher_token) == [{
        "session_id": sid, "task_description": "Sum a list", "active": True, "student_count": 1,
    }]
    # And a re-join with the original token is honoured.
    again, tok = session_manager.join_session(sid, "Alice", alice_token)
    assert again is a and tok == alice_token
    # Name collision without the token is still rejected.
    assert session_manager.join_session(sid, "Alice") == (a, None)


def test_hot_telemetry_is_not_written_per_keystroke():
    session, _ = session_manager.create_session("x")
    bob, _ = session_manager.join_session(session.session_id, "Bob")
    for _ in range(50):
        process_telemetry(session, bob.student_id, TelemetryEvent(event_type="keystroke", payload={"count": 1}))
    assert bob.total_keystrokes == 50

    restart()
    loaded = session_manager.get_session(session.session_id)
    assert loaded.students[bob.student_id].total_keystrokes == 0  # not persisted until a transition

    loaded.students[bob.student_id].total_keystrokes = 7
    session_manager.end_session(session.session_id)  # transition -> persisted
    restart()
    reloaded = session_manager.get_session(session.session_id)
    assert reloaded.active is False
    assert reloaded.ended_at is not None
    assert reloaded.students[bob.student_id].total_keystrokes == 7


def test_ended_sessions_are_evicted_after_24h():
    old, _ = session_manager.create_session("old")
    session_manager.join_session(old.session_id, "S")
    recent, _ = session_manager.create_session("recent")
    live, _ = session_manager.create_session("live")
    session_manager.end_session(old.session_id)
    session_manager.end_session(recent.session_id)
    old.ended_at = time.time() - 25 * 3600
    session_manager.persist_session(old)

    removed = session_manager.cleanup_expired_sessions()
    assert removed == 1
    assert session_manager.get_session(old.session_id) is None
    assert session_manager.get_session(recent.session_id) is not None
    assert session_manager.get_session(live.session_id) is not None

    conn = sqlite3.connect(session_manager.db_path())
    assert conn.execute("SELECT COUNT(*) FROM students WHERE session_id = ?", (old.session_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
    conn.close()

    # Boundary: exactly at the cutoff is kept, just past it is removed.
    recent.ended_at = time.time() - 24 * 3600 - 1
    session_manager.persist_session(recent)
    assert session_manager.cleanup_expired_sessions() == 1


def test_students_are_keyed_by_generated_id_in_db():
    session, _ = session_manager.create_session("x")
    a, _ = session_manager.join_session(session.session_id, "Alice")
    b, _ = session_manager.join_session(session.session_id, "Bob")
    assert a.student_id != b.student_id and len(a.student_id) >= 12

    conn = sqlite3.connect(session_manager.db_path())
    rows = conn.execute("SELECT student_id, name FROM students WHERE session_id = ? ORDER BY name",
                        (session.session_id,)).fetchall()
    conn.close()
    assert rows == [(a.student_id, "Alice"), (b.student_id, "Bob")]

    # Renaming the display label keeps the same record.
    a.name = "Alice R."
    session.students = {s.name: s for s in session.students.values()}
    session_manager.persist_session(session)
    restart()
    loaded = session_manager.get_session(session.session_id)
    assert {s.student_id: s.name for s in loaded.students.values()} == {a.student_id: "Alice R.", b.student_id: "Bob"}


def test_unknown_session_is_none_and_db_file_is_created(tmp_path):
    assert session_manager.get_session("nope") is None
    assert session_manager.get_session("") is None
    assert os.path.exists(session_manager.db_path())
