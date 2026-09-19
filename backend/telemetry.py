"""Telemetry processing: support signals, confusion detection, large-paste observations."""

from __future__ import annotations
from datetime import datetime
from models import SessionState, StudentState, TelemetryEvent
from ai_engine import code_at_last_hint


# ── Thresholds ─────────────────────────────────────────────────────
IDLE_WARNING_SECONDS = 60
IDLE_CRITICAL_SECONDS = 120
PASTE_LENGTH_THRESHOLD = 200
BACKSPACE_RATE_THRESHOLD = 0.35
CONFUSION_SPIKE_MIN_STUDENTS = 3
CONFUSION_SPIKE_RATIO = 0.5  # 50 % of class
HELP_ATTENTION_WINDOW_SECONDS = 180
RECOVERY_KEYSTROKES = 20
RECOVERY_CODE_DELTA_CHARS = 20
LEFT_ROOM_AFTER_SECONDS = 600
# Minimum idle time before the single automatic Level-1 nudge; a session's
# pause_threshold_seconds (teacher-chosen via task level) can only raise it.
AUTO_HINT_MIN_IDLE_SECONDS = 90
# A confusion episode must have been over for this long before a new one can alert;
# while an episode is ongoing the teacher is never re-alerted.
CONFUSION_SPIKE_QUIET_SECONDS = 60
CONFUSION_SPIKE_DEDUP_SECONDS = 300
MAX_HINTS_PER_STUDENT = 5

def _pause_threshold(session: SessionState) -> int:
    return max(30, int(session.pause_threshold_seconds))


def _auto_hint_idle_threshold(session: SessionState, student: StudentState) -> int:
    return _pause_threshold(session)


def _auto_hint_allowed(student: StudentState, now: float) -> bool:
    """Silence alone earns at most one Level-1 nudge; anything deeper needs the
    student to have changed their code since the last hint (or to ask)."""
    return student.auto_hints_given == 0 and student.hints_given == 0


def _propose_auto_hint(actions: dict, student: StudentState, reason: str, now: float) -> None:
    actions["should_hint"] = True
    actions["hint_reason"] = reason
    actions["force_hint_level"] = 1
    student.auto_hints_given += 1


def process_telemetry(session: SessionState, student_id: str, event: TelemetryEvent) -> dict:
    """Process a single telemetry event and return actions to take."""
    student = session.students.get(student_id)
    if student is None:
        return {}

    actions: dict = {"dashboard_update": True}
    now = datetime.now().timestamp()
    student.last_activity = now
    student.log_event(event.event_type, event.timestamp)
    # Code is accepted only from an explicit help request or a code_update; any
    # other event type (keystroke, idle, ...) must not carry editor contents.
    if event.event_type == "help":
        for key in ("current_code", "current_answer"):
            code = event.payload.get(key)
            if isinstance(code, str) and code.strip():
                student.set_code(code)
    has_started_work = bool(student.current_code.strip()) or student.total_keystrokes > 0

    # ── Per-event-type processing ──────────────────────────────────
    if event.event_type == "keystroke":
        count = event.payload.get("count", 1)
        student.total_keystrokes += count
        student.keystrokes_since_stall += count
        if student.keystrokes_since_stall >= RECOVERY_KEYSTROKES:
            student.last_help_at = 0
        student.idle_seconds = 0
        ts = event.payload.get("key_ts")
        student.last_keypress_at = ts if isinstance(ts, (int, float)) else now

    elif event.event_type == "backspace":
        count = event.payload.get("count", 1)
        student.total_backspaces += count
        student.total_keystrokes += count
        ts = event.payload.get("key_ts")
        student.last_keypress_at = ts if isinstance(ts, (int, float)) else now
        student.keystrokes_since_stall += count
        if student.keystrokes_since_stall >= RECOVERY_KEYSTROKES:
            student.last_help_at = 0

    elif event.event_type == "idle":
        secs = event.payload.get("idle_seconds", 0)
        paused_for = max(0.0, now - float(student.last_keypress_at or now))
        # Ignore idle before the student has actually started working.
        if not has_started_work:
            student.idle_seconds = 0
            secs = 0
            paused_for = 0
        else:
            student.idle_seconds = max(float(secs or 0), paused_for)
        pause_threshold = max(30, int(getattr(session, "pause_threshold_seconds", IDLE_CRITICAL_SECONDS)))
        critical_threshold = _auto_hint_idle_threshold(session, student)

        if student.idle_seconds >= critical_threshold:
            if has_started_work and _auto_hint_allowed(student, now):
                _propose_auto_hint(actions, student, "idle_threshold_exceeded", now)

    elif event.event_type == "paste":
        length = event.payload.get("length", 0)
        student.paste_events.append({"length": length, "timestamp": event.timestamp})
        # A large paste is a neutral, teacher-only observation: it does not change
        # status or frustration, and the teacher decides whether it means anything.
        if length >= PASTE_LENGTH_THRESHOLD:
            actions["large_paste_alert"] = {
                "student_id": student.student_id,
                "student_name": student.name,
                "paste_length": length,
                "message": (
                    f"{student.name} pasted {length} characters at once. "
                    "This is an observation, not a judgement — it may be their own notes "
                    "or an example from the material."
                ),
            }

    elif event.event_type == "help":
        msg = event.payload.get("message", "")
        student.help_requests.append(msg)
        student.last_support_at = now
        student.last_help_at = now
        student.keystrokes_since_stall = 0
        actions["should_hint"] = True
        actions["hint_reason"] = "help_request"
        actions["help_message"] = msg

    elif event.event_type == "code_update":
        code = event.payload.get("code", "")
        previous = student.current_code
        student.set_code(code if isinstance(code, str) else "")
        if student.current_code.strip() != previous.strip():
            if _code_delta(student.current_code, code_at_last_hint(student)) >= RECOVERY_CODE_DELTA_CHARS:
                student.last_help_at = 0
            student.idle_seconds = 0
            student.last_keypress_at = now

    elif event.event_type == "pause_wait":
        secs = float(event.payload.get("idle_seconds", 0) or 0)
        paused_for = max(0.0, now - float(student.last_keypress_at or now))
        if has_started_work:
            student.idle_seconds = max(secs, paused_for)
        else:
            student.idle_seconds = 0
            secs = 0
            paused_for = 0

        pause_threshold = _auto_hint_idle_threshold(session, student)
        if has_started_work and student.idle_seconds >= pause_threshold:
            if _auto_hint_allowed(student, now):
                _propose_auto_hint(actions, student, "pause_threshold_exceeded", now)

    # ── Recalculate status ─────────────────────────────────────────
    _update_status(student, session, now)

    # ── Check class-wide confusion ─────────────────────────────────
    spike = track_confusion_episode(session, now)
    if spike:
        actions["confusion_spike"] = spike

    return actions


def refresh_status(student: StudentState, session: SessionState, now: float | None = None):
    _update_status(student, session, now)


def _code_delta(current: str, previous: str) -> int:
    return sum(1 for a, b in zip(current, previous) if a != b) + abs(len(current) - len(previous))


def _update_status(student: StudentState, session: SessionState, now: float | None = None):
    """Traffic light for the student's *current* state, not their history.

    Every branch is driven by something that is true right now — ongoing idle or
    support the student has not worked past yet — so a student who resumes work
    goes back to green instead of staying red for the rest of the session. Large
    pastes never colour a student; they are only surfaced to the teacher.
    """
    now = datetime.now().timestamp() if now is None else now

    threshold = _pause_threshold(session)
    help_recent = student.last_help_at > 0 and now - student.last_help_at < HELP_ATTENTION_WINDOW_SECONDS
    idle_stalled = student.idle_seconds >= threshold
    if student.idle_seconds >= 2 * threshold or (help_recent and idle_stalled):
        student.status = "red"
    elif idle_stalled or help_recent:
        student.status = "yellow"
    else:
        student.status = "green"
    if student.status == "green":
        student.needs_attention_since = None
        student.attention_reason = ""
    else:
        if student.needs_attention_since is None:
            student.needs_attention_since = now
            student.keystrokes_since_stall = 0
        student.attention_reason = "idle" if idle_stalled else "help"


GENERIC_HELP_MESSAGES = {"", "student is confused", "i'm confused", "im confused", "confused", "help"}


def _suggested_action(stuck: list[StudentState]) -> str:
    """A concrete next step for the teacher, built from what stuck students asked."""
    latest_ts, latest_msg = -1.0, ""
    for s in stuck:
        for msg in reversed(s.help_requests):
            text = (msg or "").strip()
            if text.lower() in GENERIC_HELP_MESSAGES:
                continue
            if s.last_support_at > latest_ts:
                latest_ts, latest_msg = s.last_support_at, text
            break
    if latest_msg:
        return f"Suggested: pause and re-explain what students are asking about — “{latest_msg[:80]}”."
    return "Suggested: pause and re-explain the current step."


def detect_confusion_spike(session: SessionState, now: float | None = None) -> dict | None:
    """Check if enough students are stuck *right now* (status is current-state only)."""
    now = datetime.now().timestamp() if now is None else now
    if len(session.students) < 2:
        return None

    stuck = [s for s in session.students.values()
             if s.status in ("yellow", "red") and now - s.last_activity < LEFT_ROOM_AFTER_SECONDS]

    threshold = max(CONFUSION_SPIKE_MIN_STUDENTS,
                    int(len(session.students) * CONFUSION_SPIKE_RATIO))

    if len(stuck) >= threshold:
        total = len(session.students)
        return {
            "type": "confusion_spike",
            "struggling_count": len(stuck),
            "total_count": total,
            "students": [s.name for s in stuck],
            "timestamp": datetime.now().timestamp(),
            "message": (
                f"{len(stuck)}/{total} students are stuck right now. "
                + _suggested_action(stuck)
            ),
        }
    return None


def is_duplicate_confusion_spike(session: SessionState, now: float | None = None) -> bool:
    now = datetime.now().timestamp() if now is None else now
    return any(a.get("type") == "confusion_spike" and now - a.get("timestamp", 0) < CONFUSION_SPIKE_DEDUP_SECONDS for a in session.alerts)


def track_confusion_episode(session: SessionState, now: float | None = None) -> dict | None:
    """Return a spike alert only when a confusion episode *starts*.

    While the same episode is ongoing nothing is returned; once fewer students are
    stuck the episode ends, and a new one may alert after CONFUSION_SPIKE_QUIET_SECONDS.
    """
    now = datetime.now().timestamp() if now is None else now
    spike = detect_confusion_spike(session, now)
    if spike is None:
        if session.confusion_episode_active:
            session.confusion_episode_active = False
            session.confusion_episode_ended_at = now
        return None
    if session.confusion_episode_active:
        return None
    if now - session.confusion_episode_ended_at < CONFUSION_SPIKE_QUIET_SECONDS:
        return None
    session.confusion_episode_active = True
    return spike
