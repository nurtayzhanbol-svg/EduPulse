"""In-memory session store and lifecycle management."""

from typing import Optional

from models import SessionState, StudentState


# ── Global session store ───────────────────────────────────────────
_sessions: dict[str, SessionState] = {}


def create_session(task_description: str, task_level: str = "medium") -> SessionState:
    session = SessionState(task_description=task_description, task_level=task_level)
    _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> Optional[SessionState]:
    return _sessions.get(session_id)


def join_session(session_id: str, student_name: str, sid: Optional[str] = None) -> Optional[StudentState]:
    session = get_session(session_id)
    if session is None or not session.active:
        return None
    if student_name in session.students:
        # Reconnect — update socket sid
        session.students[student_name].sid = sid
        return session.students[student_name]
    student = StudentState(name=student_name, sid=sid)
    session.students[student_name] = student
    return student


def end_session(session_id: str) -> Optional[SessionState]:
    session = get_session(session_id)
    if session is None:
        return None
    session.active = False
    return session


def list_sessions() -> list[dict]:
    return [
        {
            "session_id": s.session_id,
            "task_description": s.task_description[:80],
            "active": s.active,
            "student_count": len(s.students),
        }
        for s in _sessions.values()
    ]
