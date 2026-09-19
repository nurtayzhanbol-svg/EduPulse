"""In-memory session store and lifecycle management."""

from typing import Optional

from auth import hash_token, new_token, token_matches
from models import SessionState, StudentState


# ── Global session store ───────────────────────────────────────────
_sessions: dict[str, SessionState] = {}


def create_session(task_description: str, task_level: str = "medium") -> tuple[SessionState, str]:
    """Create a session and return it with the raw teacher token (only the hash is kept)."""
    session = SessionState(task_description=task_description, task_level=task_level)
    teacher_token = new_token()
    session.teacher_token_hash = hash_token(teacher_token)
    _sessions[session.session_id] = session
    return session, teacher_token


def get_session(session_id: str) -> Optional[SessionState]:
    return _sessions.get(session_id)


def join_session(
    session_id: str, student_name: str, student_token: Optional[str] = None,
) -> tuple[Optional[StudentState], Optional[str]]:
    """Join (or re-join) a session.

    Returns ``(student, raw_token)``. A new student gets a fresh token; an existing name
    is only re-joined when the caller presents that student's token, so nobody can take
    over a name by simply joining again. ``(None, None)`` means not found / inactive,
    ``(student, None)`` means the name is taken and the token did not match.
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
    return session


def list_sessions(teacher_token: str) -> list[dict]:
    """Sessions owned by ``teacher_token`` (tokens are per-session, so usually 0 or 1)."""
    return [
        {
            "session_id": s.session_id,
            "task_description": s.task_description[:80],
            "active": s.active,
            "student_count": len(s.students),
        }
        for s in _sessions.values()
        if token_matches(teacher_token, s.teacher_token_hash)
    ]
