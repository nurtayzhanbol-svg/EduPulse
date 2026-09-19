"""Pydantic models for EduPulse sessions, students, and telemetry."""

from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime

from auth import new_session_id, new_student_id


# ── Request / Response Models ──────────────────────────────────────

class CreateSessionRequest(BaseModel):
    task_description: str = ""
    task_level: str = "medium"  # easy | medium | hard


class CreateSessionResponse(BaseModel):
    session_id: str
    join_url: str
    teacher_token: str



class JoinSessionRequest(BaseModel):
    student_name: str


class EndSessionResponse(BaseModel):
    summary: str
    analytics: dict = Field(default_factory=dict)


# ── Telemetry Event ────────────────────────────────────────────────

class TelemetryEvent(BaseModel):
    """A single behavioural telemetry event sent from the student client."""
    event_type: str  # "keystroke" | "idle" | "paste" | "backspace" | "help" | "code_update"
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    payload: dict = Field(default_factory=dict)
    # payload examples:
    #   keystroke:  {"keys_per_second": 3.2}
    #   paste:      {"length": 523, "content_preview": "def foo..."}
    #   idle:       {"idle_seconds": 65}
    #   backspace:  {"rate": 0.4}   (ratio of backspace to total keys)
    #   help:       {"message": "I don't understand arrays"}
    #   code_update: {"code": "...current code...", "line_count": 12}


# ── Student State ──────────────────────────────────────────────────

class StudentState:
    """Mutable in-memory state for a connected student."""

    # Attributes that are transient (per-process) and never persisted.
    TRANSIENT_FIELDS = ("sid",)

    def __init__(self, name: str, sid: str | None = None, student_id: str | None = None):
        self.student_id: str = student_id or new_student_id()  # stable identity; name is a display label
        self.name = name
        self.sid = sid  # Socket.IO session id
        self.token_hash: str = ""  # sha256 of the student's bearer token
        self.status: str = "green"  # green | yellow | red
        self.progress: float = 0.0
        # Quiz correctness — the only direct evidence of what the student knows.
        # quiz_score stays None until the student submits a quiz.
        self.quiz_score: float | None = None
        self.quiz_correct: int = 0
        self.quiz_total: int = 0
        self.total_keystrokes: int = 0
        self.total_backspaces: int = 0
        self.paste_events: list[dict] = []
        self.idle_seconds: float = 0.0
        self.hints_given: int = 0
        self.hint_level: int = 0
        self.last_activity: float = datetime.now().timestamp()
        self.help_requests: list[str] = []
        self.current_code: str = ""
        self.joined_at: float = datetime.now().timestamp()
        self.frustration_score: float = 0.0
        self.events: list[dict] = []  # raw event log
        self.last_keypress_at: float = datetime.now().timestamp()
        self.last_pause_hint_at: float = 0.0

    @property
    def support_signals(self) -> int:
        """How many times this student needed support: hints delivered + help asked for.

        A count of observed events, not a mastery estimate.
        """
        return self.hints_given + len(self.help_requests)

    def to_dict(self, include_code: bool = False) -> dict:
        code_preview = ""
        if self.current_code:
            compact = " ".join(self.current_code.strip().split())
            code_preview = compact[:140]
        payload = {
            "student_id": self.student_id,
            "name": self.name,
            "status": self.status,
            "progress": round(self.progress, 1),
            "quiz_score": self.quiz_score,
            "quiz_correct": self.quiz_correct,
            "quiz_total": self.quiz_total,
            "help_requests_count": len(self.help_requests),
            "support_signals": self.support_signals,
            "total_keystrokes": self.total_keystrokes,
            "paste_events_count": len(self.paste_events),
            "idle_seconds": round(self.idle_seconds, 1),
            "hints_given": self.hints_given,
            "hint_level": self.hint_level,
            "frustration_score": round(self.frustration_score, 2),
            "last_activity": self.last_activity,
            "last_keypress_at": self.last_keypress_at,
            "help_requests": self.help_requests[-3:],  # last 3
            "current_code_lines": self.current_code.count("\n") + 1 if self.current_code else 0,
            "current_code_preview": code_preview,
        }
        if include_code:
            payload["current_code"] = self.current_code
        return payload

    def to_record(self) -> dict:
        """Full persistable state (everything except transient fields)."""
        return {k: v for k, v in vars(self).items() if k not in self.TRANSIENT_FIELDS}

    @classmethod
    def from_record(cls, record: dict) -> "StudentState":
        student = cls(name=record.get("name", ""), student_id=record.get("student_id"))
        for key, value in record.items():
            if key in cls.TRANSIENT_FIELDS:
                continue
            setattr(student, key, value)
        return student


# ── Session State ──────────────────────────────────────────────────

class SessionState:
    """Mutable in-memory state for a lab session."""

    def __init__(self, task_description: str, task_level: str = "medium"):
        level = (task_level or "medium").lower()
        if level not in ("easy", "medium", "hard"):
            level = "medium"

        pause_threshold_map = {
            "easy": 60,
            "medium": 90,
            "hard": 120,
        }

        self.session_id: str = new_session_id()
        self.teacher_token_hash: str = ""
        self.task_description: str = task_description
        self.task_level: str = level
        self.pause_threshold_seconds: int = pause_threshold_map[level]
        self.quiz_mode_preference: str = "practical"
        self.quiz_difficulty_preference: str = level
        self.created_at: float = datetime.now().timestamp()
        self.active: bool = True
        self.students: dict[str, StudentState] = {}
        self.alerts: list[dict] = []  # {"type": ..., "message": ..., "timestamp": ...}
        self.summary: str | None = None
        self.analytics: dict = {}
        self.ended_at: float | None = None
        # PDF & Quiz
        self.pdf_text: str = ""
        self.pdf_analysis: str = ""
        self.pdf_filename: str = ""
        self.quiz: list[dict] = []
        self.quiz_results: dict[str, dict] = {}

    def to_dict(self, include_code: bool = False) -> dict:
        return {
            "session_id": self.session_id,
            "task_description": self.task_description,
            "task_level": self.task_level,
            "pause_threshold_seconds": self.pause_threshold_seconds,
            "has_material": bool(self.pdf_text or self.pdf_analysis),
            "active": self.active,
            "summary": self.summary,
            "student_count": len(self.students),
            "students": {
                name: s.to_dict(include_code=include_code)
                for name, s in self.students.items()
            },
            "alerts": self.alerts[-20:],  # last 20
        }

    def to_record(self) -> dict:
        """Persistable session state, excluding students (stored separately)."""
        return {k: v for k, v in vars(self).items() if k != "students"}

    @classmethod
    def from_record(cls, record: dict, students: list[StudentState]) -> "SessionState":
        session = cls(task_description=record.get("task_description", ""),
                      task_level=record.get("task_level", "medium"))
        for key, value in record.items():
            if key == "students":
                continue
            setattr(session, key, value)
        session.students = {s.name: s for s in students}
        return session
