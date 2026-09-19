"""Shared fixtures for EduPulse backend tests.

The backend modules import each other as top-level modules (``from models import ...``),
so ``backend/`` must be on ``sys.path`` before anything under test is imported.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Make sure the mock AI path is always exercised, regardless of the host environment.
for _var in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
    os.environ.pop(_var, None)

from models import SessionState, StudentState, TelemetryEvent  # noqa: E402
import ai_engine  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ai_client(monkeypatch):
    """Guarantee no real OpenAI client is ever built during tests."""
    for var in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ai_engine, "_client", None)
    yield


def make_session(n_students: int = 0, task_level: str = "medium",
                 task_description: str = "Write a function") -> SessionState:
    """Build a SessionState with ``n_students`` joined students, bypassing HTTP."""
    session = SessionState(task_description=task_description, task_level=task_level)
    for i in range(n_students):
        name = f"student{i}"
        session.students[name] = StudentState(name=name)
    return session


@pytest.fixture
def session_factory():
    return make_session


@pytest.fixture
def session():
    """Medium-level session (pause_threshold_seconds == 90) with one student."""
    return make_session(1)


@pytest.fixture
def student(session):
    return session.students["student0"]


def make_event(event_type: str, **payload) -> TelemetryEvent:
    return TelemetryEvent(event_type=event_type, payload=payload)


@pytest.fixture
def event():
    return make_event


def now_ts() -> float:
    return datetime.now().timestamp()


def start_work(student: StudentState, keystrokes: int = 5) -> None:
    """Mark a student as having started working so idle logic is not suppressed."""
    student.total_keystrokes = keystrokes
