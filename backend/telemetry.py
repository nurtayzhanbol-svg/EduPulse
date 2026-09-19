"""Telemetry processing: understanding scores, confusion detection, plagiarism flags."""

from __future__ import annotations
from datetime import datetime
from models import SessionState, StudentState, TelemetryEvent


# ── Thresholds ─────────────────────────────────────────────────────
IDLE_WARNING_SECONDS = 60
IDLE_CRITICAL_SECONDS = 120
PASTE_LENGTH_THRESHOLD = 200
BACKSPACE_RATE_THRESHOLD = 0.35
CONFUSION_SPIKE_MIN_STUDENTS = 3
CONFUSION_SPIKE_RATIO = 0.5  # 50 % of class
PAUSE_HINT_COOLDOWN_SECONDS = 45
CONFUSION_SPIKE_DEDUP_SECONDS = 30

# ── Understanding score weights (see compute_understanding_score) ──
HINT_PENALTY_PER_HINT = 18.0
IDLE_PENALTY_MAX = 25.0
IDLE_PENALTY_CAP_SECONDS = 300
FRUSTRATION_PENALTY_MAX = 20.0
PASTE_PENALTY = 20.0


def _next_hint_level(student: StudentState) -> int:
    return min(3, max(1, student.hint_level + 1))


def _pause_interval_for_next_hint(session: SessionState, student: StudentState) -> int:
    """Pause interval in seconds before the next hint level is allowed."""
    base = max(30, int(getattr(session, "pause_threshold_seconds", IDLE_WARNING_SECONDS)))
    level = _next_hint_level(student)
    multiplier = {1: 1.0, 2: 1.5, 3: 2.0}.get(level, 1.0)
    return int(base * multiplier)


def process_telemetry(session: SessionState, student_name: str, event: TelemetryEvent) -> dict:
    """Process a single telemetry event and return actions to take."""
    student = session.students.get(student_name)
    if student is None:
        return {}

    actions: dict = {"dashboard_update": True}
    now = datetime.now().timestamp()
    student.last_activity = now
    student.events.append({"type": event.event_type, "ts": event.timestamp, **event.payload})
    current_code = event.payload.get("current_code")
    if isinstance(current_code, str):
        student.current_code = current_code
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
        critical_threshold = _pause_interval_for_next_hint(session, student)

        if student.idle_seconds >= critical_threshold:
            student.frustration_score = min(1.0, student.frustration_score + 0.12)
            if has_started_work and now - student.last_pause_hint_at >= PAUSE_HINT_COOLDOWN_SECONDS:
                next_level = _next_hint_level(student)
                actions["should_hint"] = True
                actions["hint_reason"] = "idle_threshold_exceeded"
                actions["force_hint_level"] = next_level
                student.last_pause_hint_at = now
        elif student.idle_seconds >= warning_threshold:
            student.frustration_score = min(1.0, student.frustration_score + 0.05)

    elif event.event_type == "paste":
        length = event.payload.get("length", 0)
        student.paste_events.append({
            "length": length,
            "timestamp": event.timestamp,
            "preview": event.payload.get("content_preview", "")[:100],
        })
        if length >= PASTE_LENGTH_THRESHOLD:
            actions["plagiarism_alert"] = {
                "student_name": student_name,
                "paste_length": length,
                "message": f"⚠️ {student_name} pasted {length} characters at once. High plagiarism risk.",
            }
            student.frustration_score = 0  # they're not frustrated, they're cheating

    elif event.event_type == "help":
        msg = event.payload.get("message", "")
        student.help_requests.append(msg)
        answer = event.payload.get("current_answer")
        if isinstance(answer, str) and answer.strip():
            student.current_code = answer
        student.frustration_score = min(1.0, student.frustration_score + 0.25)
        actions["should_hint"] = True
        actions["hint_reason"] = "help_request"
        actions["help_message"] = msg

    elif event.event_type == "code_update":
        student.current_code = event.payload.get("code", "")
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

        answer = event.payload.get("current_answer")
        if isinstance(answer, str) and answer.strip():
            student.current_code = answer

        pause_threshold = _pause_interval_for_next_hint(session, student)
        if has_started_work and student.idle_seconds >= pause_threshold:
            student.frustration_score = min(1.0, student.frustration_score + 0.08)
            if now - student.last_pause_hint_at >= PAUSE_HINT_COOLDOWN_SECONDS:
                next_level = _next_hint_level(student)
                actions["should_hint"] = True
                actions["hint_reason"] = "pause_threshold_exceeded"
                actions["force_hint_level"] = next_level
                student.last_pause_hint_at = now

    # ── Recalculate scores ─────────────────────────────────────────
    _update_understanding_score(student)
    _update_status(student)

    # ── Check class-wide confusion ─────────────────────────────────
    spike = detect_confusion_spike(session)
    if spike:
        actions["confusion_spike"] = spike

    return actions


def compute_understanding_score(student: StudentState) -> float:
    """The single "Understanding" score (0-100) used by the live dashboard AND the
    end-of-session report, so both always agree for the same student state.

    It starts at 100 and subtracts four independent penalties:

    * **Hints** — ``HINT_PENALTY_PER_HINT`` (18) per hint taken, uncapped. A student
      needing 3 hints is at most 46; 6 hints reach 0. Hints are the strongest signal
      because each one means the student could not progress alone.
    * **Idle** — linear in ``idle_seconds`` up to ``IDLE_PENALTY_CAP_SECONDS`` (300s),
      at most ``IDLE_PENALTY_MAX`` (25). Applied only once the student has started
      working (typed or has code): idle time before starting is not a signal.
    * **Frustration** — ``frustration_score`` (0-1) x ``FRUSTRATION_PENALTY_MAX`` (20).
      Frustration accumulates from high backspace rates, long pauses and help requests.
    * **Large paste** — flat ``PASTE_PENALTY`` (20) if any of the last three pastes was
      >= ``PASTE_LENGTH_THRESHOLD`` chars (probable copy/paste of a solution). Applied
      regardless of whether the student typed anything first.

    The result is clamped to [0, 100]. The report's ``avg_understanding_score`` is the
    mean of this value over students at the moment the session ended.
    """
    has_started_work = bool(student.current_code.strip()) or student.total_keystrokes > 0

    hint_penalty = student.hints_given * HINT_PENALTY_PER_HINT
    idle_penalty = 0.0
    if has_started_work:
        idle_penalty = min(student.idle_seconds, IDLE_PENALTY_CAP_SECONDS) / IDLE_PENALTY_CAP_SECONDS * IDLE_PENALTY_MAX
    frustration_penalty = max(0.0, min(1.0, student.frustration_score)) * FRUSTRATION_PENALTY_MAX
    paste_penalty = 0.0
    if any(p.get("length", 0) >= PASTE_LENGTH_THRESHOLD for p in student.paste_events[-3:]):
        paste_penalty = PASTE_PENALTY

    score = 100.0 - hint_penalty - idle_penalty - frustration_penalty - paste_penalty
    return max(0.0, min(100.0, score))


def _update_understanding_score(student: StudentState):
    student.understanding_score = compute_understanding_score(student)


def _update_status(student: StudentState):
    """Traffic-light status based on hint usage + sustained idle time."""
    # Immediate red for plagiarism
    if any(p["length"] >= PASTE_LENGTH_THRESHOLD for p in student.paste_events[-3:]):
        student.status = "red"
        return

    if student.hints_given >= 3 or student.idle_seconds >= (IDLE_CRITICAL_SECONDS * 2):
        student.status = "red"
        return

    if student.hints_given >= 2 and student.idle_seconds >= IDLE_WARNING_SECONDS:
        student.status = "red"
        return

    if student.hints_given >= 1 or student.idle_seconds >= IDLE_CRITICAL_SECONDS:
        student.status = "yellow"
        return

    if student.frustration_score >= 0.95 and student.idle_seconds >= IDLE_CRITICAL_SECONDS:
        student.status = "yellow"
    else:
        student.status = "green"


def is_duplicate_confusion_spike(session: SessionState, now: float | None = None,
                                 window_seconds: float = CONFUSION_SPIKE_DEDUP_SECONDS) -> bool:
    """True when a confusion_spike alert was already raised within ``window_seconds``."""
    now = datetime.now().timestamp() if now is None else now
    for alert in reversed(session.alerts):
        if alert.get("type") != "confusion_spike":
            continue
        ts = alert.get("timestamp")
        if isinstance(ts, (int, float)) and now - ts < window_seconds:
            return True
        return False
    return False


def detect_confusion_spike(session: SessionState) -> dict | None:
    """Check if enough students are struggling simultaneously."""
    if len(session.students) < 2:
        return None

    struggling = [
        name for name, s in session.students.items()
        if s.status in ("yellow", "red")
    ]

    threshold = max(CONFUSION_SPIKE_MIN_STUDENTS,
                    int(len(session.students) * CONFUSION_SPIKE_RATIO))

    if len(struggling) >= threshold:
        return {
            "type": "confusion_spike",
            "struggling_count": len(struggling),
            "total_count": len(session.students),
            "students": struggling,
            "timestamp": datetime.now().timestamp(),
            "message": f"🚨 Class-wide confusion detected! {len(struggling)}/{len(session.students)} students are struggling.",
        }
    return None
