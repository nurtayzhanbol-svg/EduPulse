"""Session store and lifecycle management, backed by SQLite.

Sessions are kept in a process-local cache for hot per-keystroke telemetry and written
through to SQLite on meaningful transitions (``persist_session``): join, hint given,
quiz submitted, session end. ``get_session`` falls back to the database, so state
survives a restart and is visible to other processes after their next lookup.

The DB path comes from ``EDUPULSE_DB`` (default ``./edupulse.db``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Optional

from auth import hash_token, new_token, token_matches
from models import SessionState, StudentState


DEFAULT_DB_PATH = "./edupulse.db"
SESSION_RETENTION_SECONDS = 24 * 60 * 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    teacher_token_hash  TEXT NOT NULL,
    active              INTEGER NOT NULL,
    created_at          REAL NOT NULL,
    ended_at            REAL,
    data                TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS students (
    student_id  TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    token_hash  TEXT NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_students_session ON students(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_ended ON sessions(active, ended_at);
"""

# ── Process-local cache + connection ───────────────────────────────
_sessions: dict[str, SessionState] = {}
_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None
_conn_path: Optional[str] = None


def db_path() -> str:
    return os.environ.get("EDUPULSE_DB", DEFAULT_DB_PATH)


def _connect() -> sqlite3.Connection:
    global _conn, _conn_path
    path = db_path()
    if _conn is None or _conn_path != path:
        if _conn is not None:
            _conn.close()
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        _conn, _conn_path = conn, path
    return _conn


def reset_cache() -> None:
    """Drop the in-memory cache and DB connection (simulates a process restart)."""
    global _conn, _conn_path
    with _lock:
        _sessions.clear()
        if _conn is not None:
            _conn.close()
        _conn, _conn_path = None, None


# ── Serialisation ──────────────────────────────────────────────────

def _dumps(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _row_to_session(row: sqlite3.Row, student_rows: list[sqlite3.Row]) -> SessionState:
    students = [StudentState.from_record(json.loads(r["data"])) for r in student_rows]
    return SessionState.from_record(json.loads(row["data"]), students)


def _load_session(session_id: str) -> Optional[SessionState]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    student_rows = conn.execute(
        "SELECT * FROM students WHERE session_id = ? ORDER BY rowid", (session_id,)
    ).fetchall()
    return _row_to_session(row, student_rows)


def persist_session(session: SessionState) -> None:
    """Write the session and all of its students to SQLite (upsert)."""
    conn = _connect()
    with _lock:
        conn.execute("BEGIN")
        try:
            conn.execute(
                """INSERT INTO sessions (session_id, teacher_token_hash, active, created_at, ended_at, data)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     teacher_token_hash = excluded.teacher_token_hash,
                     active = excluded.active,
                     created_at = excluded.created_at,
                     ended_at = excluded.ended_at,
                     data = excluded.data""",
                (
                    session.session_id,
                    session.teacher_token_hash,
                    1 if session.active else 0,
                    session.created_at,
                    session.ended_at,
                    _dumps(session.to_record()),
                ),
            )
            for student in session.students.values():
                conn.execute(
                    """INSERT INTO students (student_id, session_id, name, token_hash, data)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(student_id) DO UPDATE SET
                         name = excluded.name,
                         token_hash = excluded.token_hash,
                         data = excluded.data""",
                    (
                        student.student_id,
                        session.session_id,
                        student.name,
                        student.token_hash,
                        _dumps(student.to_record()),
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def cleanup_expired_sessions(now: Optional[float] = None,
                             retention_seconds: float = SESSION_RETENTION_SECONDS) -> int:
    """Delete sessions that ended more than ``retention_seconds`` ago. Returns rows removed."""
    cutoff = (now if now is not None else time.time()) - retention_seconds
    conn = _connect()
    with _lock:
        rows = conn.execute(
            "SELECT session_id FROM sessions WHERE active = 0 AND ended_at IS NOT NULL AND ended_at < ?",
            (cutoff,),
        ).fetchall()
        ids = [r[0] for r in rows]
        if not ids:
            return 0
        conn.execute("BEGIN")
        try:
            for sid in ids:
                conn.execute("DELETE FROM students WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
                _sessions.pop(sid, None)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return len(ids)


# ── Public API ─────────────────────────────────────────────────────

def create_session(task_description: str, task_level: str = "medium") -> tuple[SessionState, str]:
    """Create a session and return it with the raw teacher token (only the hash is kept)."""
    session = SessionState(task_description=task_description, task_level=task_level)
    teacher_token = new_token()
    session.teacher_token_hash = hash_token(teacher_token)
    with _lock:
        _sessions[session.session_id] = session
    persist_session(session)
    return session, teacher_token


def get_session(session_id: str) -> Optional[SessionState]:
    if not session_id:
        return None
    with _lock:
        session = _sessions.get(session_id)
        if session is not None:
            return session
        session = _load_session(session_id)
        if session is not None:
            _sessions[session_id] = session
        return session


def join_session(
    session_id: str, student_name: str, student_token: Optional[str] = None,
) -> tuple[Optional[StudentState], Optional[str]]:
    """Join (or re-join) a session.

    Returns ``(student, raw_token)``. A new student gets a fresh token; an existing name
    is only re-joined when the caller presents that student's token, so nobody can take
    over a name by simply joining again. ``(None, None)`` means not found / inactive,
    ``(student, None)`` means the name is taken and the token did not match.

    Each student is identified by a generated ``student_id`` (the DB primary key); the
    name is a display label that is kept unique per session so the socket protocol
    (which addresses students by name) stays unambiguous.
    """
    session = get_session(session_id)
    if session is None or not session.active:
        return None, None
    existing = session.students.get(student_name)
    if existing is not None:
        if token_matches(student_token, existing.token_hash):
            return existing, student_token
        return existing, None
    token = new_token()
    student = StudentState(name=student_name)
    student.token_hash = hash_token(token)
    session.students[student_name] = student
    persist_session(session)
    return student, token


def authenticate_student(session_id: str, student_name: str, student_token: Optional[str]) -> Optional[StudentState]:
    session = get_session(session_id)
    if session is None:
        return None
    student = session.students.get(student_name or "")
    if student is None or not token_matches(student_token, student.token_hash):
        return None
    return student


def end_session(session_id: str) -> Optional[SessionState]:
    session = get_session(session_id)
    if session is None:
        return None
    session.active = False
    if session.ended_at is None:
        session.ended_at = time.time()
    persist_session(session)
    cleanup_expired_sessions()
    return session


def list_sessions(teacher_token: str) -> list[dict]:
    """Sessions owned by ``teacher_token`` (tokens are per-session, so usually 0 or 1)."""
    if not teacher_token:
        return []
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT session_id FROM sessions WHERE teacher_token_hash = ? ORDER BY created_at",
        (hash_token(teacher_token),),
    ).fetchall()
    result = []
    for row in rows:
        s = get_session(row["session_id"])
        if s is None or not token_matches(teacher_token, s.teacher_token_hash):
            continue
        result.append({
            "session_id": s.session_id,
            "task_description": s.task_description[:80],
            "active": s.active,
            "student_count": len(s.students),
        })
    return result
