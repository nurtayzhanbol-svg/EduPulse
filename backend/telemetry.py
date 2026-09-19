"""Telemetry processing: support signals, confusion detection, large-paste observations."""

from __future__ import annotations
from datetime import datetime
from models import SessionState, StudentState, TelemetryEvent
from ai_engine import code_changed_since_last_hint


# ── Thresholds ─────────────────────────────────────────────────────
IDLE_WARNING_SECONDS = 60
IDLE_CRITICAL_SECONDS = 120
PASTE_LENGTH_THRESHOLD = 200
BACKSPACE_RATE_THRESHOLD = 0.35
CONFUSION_SPIKE_MIN_STUDENTS = 3
CONFUSION_SPIKE_RATIO = 0.5  # 50 % of class
PAUSE_HINT_COOLDOWN_SECONDS = 45
# Minimum idle time before the single automatic Level-1 nudge; a session's
# pause_threshold_seconds (teacher-chosen via task level) can only raise it.
AUTO_HINT_MIN_IDLE_SECONDS = 90
# A confusion episode must have been over for this long before a new one can alert;
# while an episode is ongoing the teacher is never re-alerted.
CONFUSION_SPIKE_QUIET_SECONDS = 60
# How long a hint/help request keeps colouring a student who has gone back to work.
SUPPORT_RECOVERY_SECONDS = 120


def _next_hint_level(student: StudentState) -> int:
    return min(3, max(1, student.hint_level + 1))


def _pause_interval_for_next_hint(session: SessionState, student: StudentState) -> int:
    """Pause interval in seconds before the next hint level is allowed."""
    base = max(30, int(getattr(session, "pause_threshold_seconds", IDLE_WARNING_SECONDS)))
    level = _next_hint_level(student)
    multiplier = {1: 1.0, 2: 1.5, 3: 2.0}.get(level, 1.0)
    return int(base * multiplier)


def _auto_hint_idle_threshold(session: SessionState, student: StudentState) -> int:
    return max(AUTO_HINT_MIN_IDLE_SECONDS, _pause_interval_for_next_hint(session, student))


def _auto_hint_allowed(student: StudentState, now: float) -> bool:
    """Silence alone earns at most one Level-1 nudge; anything deeper needs the
    student to have changed their code since the last hint (or to ask)."""
    if student.hint_level >= 3:
        return False
    if now - student.last_pause_hint_at < PAUSE_HINT_COOLDOWN_SECONDS:
        return False
    return student.hint_level == 0 or code_changed_since_last_hint(student)


def _propose_auto_hint(actions: dict, student: StudentState, reason: str, now: float) -> None:
    actions["should_hint"] = True
    actions["hint_reason"] = reason
    actions["force_hint_level"] = _next_hint_level(student)
    student.last_pause_hint_at = now


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
        student.total_keystrokes += event.payload.get("count", 1)
        student.idle_seconds = 0
        ts = event.payload.get("key_ts")
        student.last_keypress_at = ts if isinstance(ts, (int, float)) else now

    elif event.event_type == "backspace":
        count = event.payload.get("count", 1)
        student.total_backspaces += count
        student.total_keystrokes += count
        ts = event.payload.get("key_ts")
        student.last_keypress_at = ts if isinstance(ts, (int, float)) else now
        if student.total_keystrokes > 0:
            rate = student.total_backspaces / student.total_keystrokes
            if rate > BACKSPACE_RATE_THRESHOLD:
                student.frustration_score = min(1.0, student.frustration_score + 0.1)

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
        warning_threshold = max(15, int(pause_threshold * 0.5))
        critical_threshold = _auto_hint_idle_threshold(session, student)

        if student.idle_seconds >= critical_threshold:
            student.frustration_score = min(1.0, student.frustration_score + 0.12)
            if has_started_work and _auto_hint_allowed(student, now):
                _propose_auto_hint(actions, student, "idle_threshold_exceeded", now)
        elif student.idle_seconds >= warning_threshold:
            student.frustration_score = min(1.0, student.frustration_score + 0.05)

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
        student.frustration_score = min(1.0, student.frustration_score + 0.25)
        actions["should_hint"] = True
        actions["hint_reason"] = "help_request"
        actions["help_message"] = msg

    elif event.event_type == "code_update":
        code = event.payload.get("code", "")
        previous = student.current_code
        student.set_code(code if isinstance(code, str) else "")
        if student.current_code.strip() != previous.strip():
            # Changed code is work, whether or not keystrokes were reported.
            student.idle_seconds = 0
            student.last_keypress_at = now
        lines = student.current_code.count("\n") + 1
        student.progress = min(100.0, lines * 5.0)  # rough heuristic

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
            student.frustration_score = min(1.0, student.frustration_score + 0.08)
            if _auto_hint_allowed(student, now):
                _propose_auto_hint(actions, student, "pause_threshold_exceeded", now)

    # ── Recalculate status ─────────────────────────────────────────
    _update_status(student, now)

    # ── Check class-wide confusion ─────────────────────────────────
    spike = track_confusion_episode(session, now)
    if spike:
        actions["confusion_spike"] = spike

    return actions


def refresh_status(student: StudentState, now: float | None = None):
    """Recompute status outside the telemetry loop (e.g. once a hint is delivered)."""
    _update_status(student, now)


def _stalled_since_support(student: StudentState) -> bool:
    """True when support arrived and the student has not typed anything since."""
    return student.last_support_at > 0 and student.last_keypress_at < student.last_support_at


def _update_status(student: StudentState, now: float | None = None):
    """Traffic light for the student's *current* state, not their history.

    Every branch is driven by something that is true right now — ongoing idle or
    support the student has not worked past yet — so a student who resumes work
    goes back to green instead of staying red for the rest of the session. Large
    pastes never colour a student; they are only surfaced to the teacher.
    """
    now = datetime.now().timestamp() if now is None else now

    stuck = _stalled_since_support(student)
    if student.idle_seconds >= IDLE_CRITICAL_SECONDS or (stuck and student.idle_seconds >= IDLE_WARNING_SECONDS):
        student.status = "red"
        return

    if student.idle_seconds >= IDLE_WARNING_SECONDS or (
        stuck and now - student.last_support_at <= SUPPORT_RECOVERY_SECONDS
    ):
        student.status = "yellow"
        return

    student.status = "green"


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


def detect_confusion_spike(session: SessionState) -> dict | None:
    """Check if enough students are stuck *right now* (status is current-state only)."""
    if len(session.students) < 2:
        return None

    stuck = [s for s in session.students.values() if s.status in ("yellow", "red")]

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


def track_confusion_episode(session: SessionState, now: float | None = None) -> dict | None:
    """Return a spike alert only when a confusion episode *starts*.

    While the same episode is ongoing nothing is returned; once fewer students are
    stuck the episode ends, and a new one may alert after CONFUSION_SPIKE_QUIET_SECONDS.
    """
    now = datetime.now().timestamp() if now is None else now
    spike = detect_confusion_spike(session)
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
