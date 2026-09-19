"""Token helpers: opaque bearer tokens, stored only as SHA-256 hashes."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional

from fastapi import HTTPException, Request


def new_session_id() -> str:
    return secrets.token_urlsafe(12)


def new_student_id() -> str:
    return secrets.token_urlsafe(9)


def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(token: Optional[str], token_hash: Optional[str]) -> bool:
    if not token or not token_hash:
        return False
    return hmac.compare_digest(hash_token(token), token_hash)


def bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def require_teacher(request: Request, session) -> None:
    """Raise 401 when no bearer token is sent, 403 when it is not this session's teacher token."""
    token = bearer_token(request)
    if token is None:
        raise HTTPException(401, "Teacher token required")
    if not token_matches(token, session.teacher_token_hash):
        raise HTTPException(403, "Invalid teacher token")


def require_student(request: Request, session, student_name: str) -> None:
    """Raise 401/403 unless the bearer token belongs to ``student_name`` in ``session``."""
    token = bearer_token(request)
    if token is None:
        raise HTTPException(401, "Student token required")
    student = session.students.get(student_name)
    if student is None or not token_matches(token, student.token_hash):
        raise HTTPException(403, "Invalid student token")
