"""Unit tests for backend/telemetry.py."""

from __future__ import annotations

import pytest

import ai_engine
import telemetry
from telemetry import (
    IDLE_CRITICAL_SECONDS,
    IDLE_WARNING_SECONDS,
    PASTE_LENGTH_THRESHOLD,
    PAUSE_HINT_COOLDOWN_SECONDS,
    SUPPORT_RECOVERY_SECONDS,
    _next_hint_level,
    _pause_interval_for_next_hint,
    _update_status,
    detect_confusion_spike,
    process_telemetry,
    track_confusion_episode,
)
from models import StudentState

from conftest import make_event, make_session, now_ts, start_work, student_id


# ── process_telemetry: generic behaviour ──────────────────────────


def test_unknown_student_returns_empty_dict(session):
    assert process_telemetry(session, "nobody", make_event("keystroke")) == {}


def test_event_is_logged_and_activity_updated(session, student):
    before = student.last_activity
    ev = make_event("keystroke", count=2, extra="x")
    actions = process_telemetry(session, student_id(session, "student0"), ev)
    assert actions["dashboard_update"] is True
    assert student.last_activity >= before
    assert student.events == [{"type": "keystroke", "ts": ev.timestamp}]


def test_unknown_event_type_only_updates_dashboard(session, student):
    actions = process_telemetry(session, student_id(session, "student0"), make_event("mystery"))
    assert actions == {"dashboard_update": True}
    assert student.status == "green"
    assert student.support_signals == 0


def test_current_code_in_payload_is_ignored_for_non_help_events(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("mystery", current_code="x = 1"))
    assert student.current_code == ""


def test_non_string_current_code_is_ignored(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", current_code=123))
    assert student.current_code == ""


# ── keystroke ─────────────────────────────────────────────────────


def test_keystroke_increments_and_resets_idle(session, student):
    student.idle_seconds = 50
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", count=3))
    assert student.total_keystrokes == 3
    assert student.idle_seconds == 0


def test_keystroke_default_count_is_one(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke"))
    assert student.total_keystrokes == 1


def test_keystroke_uses_key_ts_when_numeric(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", key_ts=1234.5))
    assert student.last_keypress_at == 1234.5


def test_keystroke_falls_back_to_now_for_non_numeric_key_ts(session, student):
    before = now_ts()
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", key_ts="bad"))
    assert student.last_keypress_at >= before


# ── backspace ─────────────────────────────────────────────────────


def test_backspace_counts_towards_keystrokes(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", count=10))
    process_telemetry(session, student_id(session, "student0"), make_event("backspace", count=2))
    assert student.total_backspaces == 2
    assert student.total_keystrokes == 12


def test_backspace_below_rate_threshold_no_frustration(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", count=10))
    # 3 / 13 ≈ 0.23 <= 0.35
    process_telemetry(session, student_id(session, "student0"), make_event("backspace", count=3))
    assert student.frustration_score == 0.0


def test_backspace_above_rate_threshold_adds_frustration(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", count=10))
    # 7 / 17 ≈ 0.41 > 0.35
    process_telemetry(session, student_id(session, "student0"), make_event("backspace", count=7))
    assert student.frustration_score == pytest.approx(0.1)


def test_backspace_frustration_caps_at_one(session, student):
    student.frustration_score = 0.95
    process_telemetry(session, student_id(session, "student0"), make_event("backspace", count=1))  # rate 1.0
    assert student.frustration_score == 1.0


def test_backspace_uses_key_ts(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("backspace", key_ts=42))
    assert student.last_keypress_at == 42


# ── idle + has_started_work guard ─────────────────────────────────


def test_idle_before_any_typing_does_not_accrue(session, student):
    actions = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=500))
    assert student.idle_seconds == 0
    assert "should_hint" not in actions
    assert student.frustration_score == 0.0
    assert student.status == "green"


def test_idle_before_typing_resets_previously_set_idle(session, student):
    student.idle_seconds = 30
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=500))
    assert student.idle_seconds == 0


def test_idle_ignores_stale_keypress_before_typing(session, student):
    student.last_keypress_at = now_ts() - 1000
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=0))
    assert student.idle_seconds == 0


def test_code_in_editor_counts_as_started_work(session, student):
    student.current_code = "print(1)"
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=40))
    assert student.idle_seconds == 40


def test_whitespace_only_code_does_not_count_as_started(session, student):
    student.current_code = "   \n\t"
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=40))
    assert student.idle_seconds == 0


def test_idle_takes_max_of_payload_and_wall_clock_pause(session, student):
    start_work(student)
    student.last_keypress_at = now_ts() - 200
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=10))
    assert student.idle_seconds == pytest.approx(200, abs=2)


def test_idle_below_warning_threshold_no_frustration(session, student):
    start_work(student)
    # medium: pause_threshold 90 -> warning 45
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=44))
    assert student.frustration_score == 0.0


def test_idle_at_warning_threshold_adds_small_frustration(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=45))
    assert student.frustration_score == pytest.approx(0.05)


def test_idle_just_below_critical_is_warning_only(session, student):
    start_work(student)
    actions = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=89))
    assert student.frustration_score == pytest.approx(0.05)
    assert "should_hint" not in actions


def test_idle_at_critical_threshold_triggers_hint(session, student):
    start_work(student)
    actions = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=90))
    assert actions["should_hint"] is True
    assert actions["hint_reason"] == "idle_threshold_exceeded"
    assert actions["force_hint_level"] == 1
    assert student.frustration_score == pytest.approx(0.12)
    assert student.last_pause_hint_at > 0


def test_idle_warning_threshold_floor_is_15s():
    session = make_session(1, task_level="easy")
    session.pause_threshold_seconds = 10  # below the 30s floor -> base 30, warning 15
    student = session.student_by_name("student0")
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=15))
    assert student.frustration_score == pytest.approx(0.05)


# ── hint escalation ───────────────────────────────────────────────


@pytest.mark.parametrize("current,expected", [(0, 1), (1, 2), (2, 3), (3, 3), (10, 3), (-5, 1)])
def test_next_hint_level_caps_between_1_and_3(current, expected):
    s = StudentState("x")
    s.hint_level = current
    assert _next_hint_level(s) == expected


def test_idle_hint_cooldown_suppresses_second_hint(session, student):
    start_work(student)
    first = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=300))
    assert first["should_hint"] is True
    second = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=400))
    assert "should_hint" not in second
    assert "force_hint_level" not in second
    # frustration still accrues even without a hint
    assert student.frustration_score == pytest.approx(0.24)


def test_idle_alone_never_escalates_past_level_one(session, student):
    """Silence earns one Level-1 nudge; Level 2 needs a help click or changed code."""
    start_work(student)
    first = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=300))
    assert first["force_hint_level"] == 1
    student.last_pause_hint_at = now_ts() - PAUSE_HINT_COOLDOWN_SECONDS - 1
    student.hint_level = 1
    ai_engine._remember_hint_delivery(student, now=0.0)
    again = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=300))
    assert "should_hint" not in again
    assert "force_hint_level" not in again


def test_idle_after_changed_code_may_propose_next_level(session, student):
    start_work(student)
    student.hint_level = 1
    student.set_code("x = 1")
    ai_engine._remember_hint_delivery(student, now=0.0)
    student.set_code("x = 1\ny = 2")
    student.last_pause_hint_at = now_ts() - PAUSE_HINT_COOLDOWN_SECONDS - 1
    again = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=300))
    assert again["should_hint"] is True
    assert again["force_hint_level"] == 2


def test_auto_hint_needs_longer_idle_than_status_warning(session, student):
    start_work(student)
    assert telemetry.AUTO_HINT_MIN_IDLE_SECONDS > 60
    actions = process_telemetry(
        session, student_id(session, "student0"),
        make_event("idle", idle_seconds=telemetry.AUTO_HINT_MIN_IDLE_SECONDS - 1),
    )
    assert "should_hint" not in actions


def test_idle_hint_just_inside_cooldown_is_suppressed(session, student):
    start_work(student)
    student.last_pause_hint_at = now_ts() - (PAUSE_HINT_COOLDOWN_SECONDS - 5)
    actions = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=300))
    assert "should_hint" not in actions


def test_no_automatic_hint_after_level_three(session, student):
    """Hard cap: once Level 3 was given, idle never re-proposes a hint every 45 s."""
    start_work(student)
    student.hint_level = 3
    ai_engine._remember_hint_delivery(student, now=0.0)
    student.set_code("changed")
    for _ in range(3):
        student.last_pause_hint_at = 0
        actions = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=999))
        assert "should_hint" not in actions
        assert "force_hint_level" not in actions


# ── _pause_interval_for_next_hint ─────────────────────────────────


@pytest.mark.parametrize("level,base", [("easy", 60), ("medium", 90), ("hard", 120)])
@pytest.mark.parametrize("hint_level,multiplier", [(0, 1.0), (1, 1.5), (2, 2.0), (3, 2.0), (7, 2.0)])
def test_pause_interval_multipliers(level, base, hint_level, multiplier):
    session = make_session(1, task_level=level)
    assert session.pause_threshold_seconds == base
    s = session.student_by_name("student0")
    s.hint_level = hint_level
    assert _pause_interval_for_next_hint(session, s) == int(base * multiplier)


def test_pause_interval_has_30s_floor():
    session = make_session(1)
    session.pause_threshold_seconds = 5
    assert _pause_interval_for_next_hint(session, session.student_by_name("student0")) == 30


def test_pause_interval_falls_back_to_idle_warning_without_attribute():
    class Bare:
        pass

    s = StudentState("x")
    assert _pause_interval_for_next_hint(Bare(), s) == IDLE_WARNING_SECONDS


def test_critical_idle_threshold_scales_with_hint_level(session, student):
    """After one hint (level 1) the next idle hint needs 1.5x the base pause (135s on medium),
    and only once the code has changed since that hint."""
    start_work(student)
    student.hint_level = 1
    student.set_code("x = 1")
    ai_engine._remember_hint_delivery(student, now=0.0)
    student.set_code("x = 1\ny = 2")
    actions = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=134))
    assert "should_hint" not in actions
    actions = process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=135))
    assert actions["should_hint"] is True
    assert actions["force_hint_level"] == 2


# ── paste / large-paste observation ───────────────────────────────
# The old tests here encoded the "High plagiarism risk" behaviour (red status,
# frustration reset). Those were replaced: a large paste is a neutral, teacher-only
# observation that never changes status or frustration.


def test_small_paste_is_recorded_without_alert(session, student):
    actions = process_telemetry(
        session, student_id(session, "student0"),
        make_event("paste", length=PASTE_LENGTH_THRESHOLD - 1, content_preview="abc"),
    )
    assert "large_paste_alert" not in actions
    assert "plagiarism_alert" not in actions
    assert student.paste_events[0]["length"] == PASTE_LENGTH_THRESHOLD - 1
    assert set(student.paste_events[0]) == {"length", "timestamp"}
    assert student.status == "green"


def test_large_paste_is_a_neutral_observation(session, student):
    start_work(student)
    student.frustration_score = 0.7
    actions = process_telemetry(
        session, student_id(session, "student0"),
        make_event("paste", length=PASTE_LENGTH_THRESHOLD, content_preview="z" * 500),
    )
    alert = actions["large_paste_alert"]
    assert alert["student_id"] == student.student_id
    assert alert["student_name"] == "student0"
    assert alert["paste_length"] == PASTE_LENGTH_THRESHOLD
    assert "student0" in alert["message"]
    for word in ("plagiarism", "cheat", "risk", "⚠"):
        assert word not in alert["message"].lower()
    assert "plagiarism_alert" not in actions
    assert student.frustration_score == 0.7
    assert student.status == "green"
    assert "preview" not in student.paste_events[0]


def test_large_paste_never_changes_status_by_itself(session, student):
    sid = student_id(session, "student0")
    process_telemetry(session, sid, make_event("paste", length=500))
    assert student.status == "green"
    assert student.support_signals == 0
    for _ in range(3):
        process_telemetry(session, sid, make_event("paste", length=5000))
    assert student.status == "green"
    assert len(student.paste_events) == 4


# ── help ──────────────────────────────────────────────────────────


def test_help_requests_hint_and_records_message(session, student):
    actions = process_telemetry(session, student_id(session, "student0"), make_event("help", message="stuck on loops"))
    assert actions["should_hint"] is True
    assert actions["hint_reason"] == "help_request"
    assert actions["help_message"] == "stuck on loops"
    assert student.help_requests == ["stuck on loops"]
    assert student.frustration_score == pytest.approx(0.25)


def test_help_stores_current_answer_as_code(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("help", message="?", current_answer="for i in x:"))
    assert student.current_code == "for i in x:"


def test_help_ignores_blank_current_answer(session, student):
    student.current_code = "keep"
    process_telemetry(session, student_id(session, "student0"), make_event("help", message="?", current_answer="   "))
    assert student.current_code == "keep"


def test_help_frustration_caps_at_one(session, student):
    for _ in range(5):
        process_telemetry(session, student_id(session, "student0"), make_event("help", message="?"))
    assert student.frustration_score == 1.0


# ── code_update ───────────────────────────────────────────────────


def test_code_update_stores_code_but_never_changes_progress(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("code_update", code="a\nb\nc"))
    assert student.current_code == "a\nb\nc"
    assert student.progress == 0.0
    process_telemetry(session, student_id(session, "student0"), make_event("code_update", code="\n" * 40))
    assert student.progress == 0.0


def test_code_update_with_missing_code_clears_editor(session, student):
    student.current_code = "old"
    process_telemetry(session, student_id(session, "student0"), make_event("code_update"))
    assert student.current_code == ""
    assert student.progress == 0.0


# ── task steps ────────────────────────────────────────────────────


def test_progress_is_completed_steps_over_total(student):
    assert student.mark_step_done(0, 3) is True
    assert student.completed_steps == [0]
    assert student.progress == pytest.approx(100 / 3)
    assert student.mark_step_done(0, 3) is True  # idempotent
    assert student.completed_steps == [0]
    assert student.mark_step_done(2, 3) is True
    assert student.completed_steps == [0, 2]
    assert student.progress == pytest.approx(200 / 3)
    assert student.mark_step_done(1, 3) is True
    assert student.progress == 100.0


def test_mark_step_done_rejects_invalid_indexes(student):
    for bad in (-1, 3, "0", None, True, 1.0):
        assert student.mark_step_done(bad, 3) is False
    assert student.completed_steps == []
    assert student.progress == 0.0
    assert student.mark_step_done(0, 0) is False


def test_reset_steps_clears_progress(student):
    student.mark_step_done(1, 2)
    student.reset_steps()
    assert student.completed_steps == []
    assert student.progress == 0.0


def test_completed_steps_are_persisted_and_restored(student):
    student.mark_step_done(1, 2)
    record = student.to_record()
    assert record["completed_steps"] == [1]
    restored = student.__class__.from_record(record)
    assert restored.completed_steps == [1]
    assert restored.progress == 50.0
    assert restored.to_dict()["completed_steps"] == [1]


# ── pause_wait ────────────────────────────────────────────────────


def test_pause_wait_before_typing_is_ignored(session, student):
    actions = process_telemetry(session, student_id(session, "student0"), make_event("pause_wait", idle_seconds=999))
    assert student.idle_seconds == 0
    assert "should_hint" not in actions
    assert student.frustration_score == 0.0


def test_pause_wait_at_threshold_hints_with_reason(session, student):
    start_work(student)
    actions = process_telemetry(session, student_id(session, "student0"), make_event("pause_wait", idle_seconds=90))
    assert actions["should_hint"] is True
    assert actions["hint_reason"] == "pause_threshold_exceeded"
    assert actions["force_hint_level"] == 1
    assert student.frustration_score == pytest.approx(0.08)


def test_pause_wait_below_threshold_no_hint(session, student):
    start_work(student)
    actions = process_telemetry(session, student_id(session, "student0"), make_event("pause_wait", idle_seconds=89))
    assert "should_hint" not in actions
    assert student.frustration_score == 0.0
    assert student.idle_seconds == 89


def test_pause_wait_respects_cooldown(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("pause_wait", idle_seconds=300))
    second = process_telemetry(session, student_id(session, "student0"), make_event("pause_wait", idle_seconds=300))
    assert "should_hint" not in second
    assert student.frustration_score == pytest.approx(0.16)


def test_pause_wait_does_not_accept_code(session, student):
    process_telemetry(session, student_id(session, "student0"), make_event("pause_wait", idle_seconds=0, current_answer="x=1"))
    assert student.current_code == ""


def test_pause_wait_handles_none_idle_seconds(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("pause_wait", idle_seconds=None))
    assert student.idle_seconds == pytest.approx(0, abs=1)


# ── support signals & quiz evidence ───────────────────────────────


def test_support_signals_count_hints_and_help_requests():
    s = StudentState("x")
    assert s.support_signals == 0
    s.hints_given = 2
    s.help_requests = ["stuck", "still stuck"]
    assert s.support_signals == 4


def test_telemetry_never_infers_a_quiz_score(session, student):
    start_work(student)
    student.hints_given = 4
    student.frustration_score = 1.0
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=300))
    process_telemetry(session, student_id(session, "student0"), make_event("paste", length=PASTE_LENGTH_THRESHOLD))
    assert student.quiz_score is None
    assert student.to_dict()["quiz_score"] is None


# ── _update_status ────────────────────────────────────────────────


def _status(idle=0.0, frustration=0.0, hints=0, support_ago=None,
            typed_since_support=False, pastes=(), paste_age=0.0):
    now = now_ts()
    s = StudentState("x")
    s.hints_given = hints
    s.idle_seconds = idle
    s.frustration_score = frustration
    s.last_keypress_at = now - 1
    if support_ago is not None:
        s.last_support_at = now - support_ago
        if not typed_since_support:
            s.last_keypress_at = s.last_support_at - 1
    s.paste_events = [{"length": p, "timestamp": now - paste_age} for p in pastes]
    _update_status(s, now)
    return s.status


def test_status_green_by_default():
    assert _status() == "green"


@pytest.mark.parametrize("pastes", [(PASTE_LENGTH_THRESHOLD,), (500,), (1, 500, 1, 1), (5000, 5000, 5000)])
def test_status_ignores_large_pastes(pastes):
    """Large pastes are a teacher-only observation; they never colour a student."""
    assert _status(pastes=pastes) == "green"
    assert _status(pastes=pastes, paste_age=1000) == "green"


def test_status_large_paste_does_not_mask_real_signals():
    assert _status(pastes=(500,), idle=IDLE_CRITICAL_SECONDS) == "red"


@pytest.mark.parametrize("hints", [1, 2, 5])
def test_status_ignores_lifetime_hint_count(hints):
    """Hints are history; a student typing away right now is green whatever they used."""
    assert _status(hints=hints) == "green"


def test_status_yellow_while_support_is_unresolved():
    assert _status(support_ago=10) == "yellow"


def test_status_clears_once_the_student_types_again():
    assert _status(support_ago=10, typed_since_support=True) == "green"


def test_status_unresolved_support_expires_after_recovery_window():
    assert _status(support_ago=SUPPORT_RECOVERY_SECONDS + 1) == "green"


def test_status_red_when_still_idle_after_support():
    assert _status(support_ago=10, idle=IDLE_WARNING_SECONDS - 1) == "yellow"
    assert _status(support_ago=10, idle=IDLE_WARNING_SECONDS) == "red"


def test_status_idle_boundaries():
    assert _status(idle=IDLE_WARNING_SECONDS - 1) == "green"
    assert _status(idle=IDLE_WARNING_SECONDS) == "yellow"
    assert _status(idle=IDLE_CRITICAL_SECONDS - 1) == "yellow"
    assert _status(idle=IDLE_CRITICAL_SECONDS) == "red"


def test_status_frustration_alone_stays_green():
    assert _status(frustration=1.0) == "green"


def test_status_uses_global_idle_constants_not_session_threshold():
    """A hard-level session (pause 120s) and an easy one (60s) share the same status cut-offs."""
    for level in ("easy", "medium", "hard"):
        session = make_session(1, task_level=level)
        s = session.student_by_name("student0")
        start_work(s)
        process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=119))
        assert s.status == "yellow", level


def test_status_recovers_when_a_student_resumes_after_asking_for_help(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("help", message="stuck"))
    assert student.status == "yellow"
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke"))
    assert student.status == "green"


def test_status_recovers_when_a_student_resumes_after_going_idle(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=IDLE_CRITICAL_SECONDS))
    assert student.status == "red"
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke"))
    assert student.status == "green"


# ── detect_confusion_spike ────────────────────────────────────────


def _mark_struggling(session, count, status="yellow"):
    for name in list(session.students)[:count]:
        session.students[name].status = status


def test_confusion_spike_none_with_fewer_than_two_students():
    assert detect_confusion_spike(make_session(0)) is None
    s1 = make_session(1)
    _mark_struggling(s1, 1, "red")
    assert detect_confusion_spike(s1) is None


def test_confusion_spike_two_students_can_never_fire():
    """threshold = max(3, int(2*0.5)) = 3 > class size."""
    session = make_session(2)
    _mark_struggling(session, 2, "red")
    assert detect_confusion_spike(session) is None


def test_confusion_spike_min_threshold_is_three():
    session = make_session(4)  # int(4*0.5)=2 -> threshold 3
    _mark_struggling(session, 2)
    assert detect_confusion_spike(session) is None
    _mark_struggling(session, 3)
    spike = detect_confusion_spike(session)
    assert spike is not None
    assert spike["type"] == "confusion_spike"
    assert spike["struggling_count"] == 3
    assert spike["total_count"] == 4
    assert spike["students"] == ["student0", "student1", "student2"]
    assert "3/4" in spike["message"]


@pytest.mark.parametrize("n,threshold", [(3, 3), (5, 3), (6, 3), (7, 3), (8, 4), (10, 5), (11, 5)])
def test_confusion_spike_threshold_is_max_of_3_and_half_class(n, threshold):
    session = make_session(n)
    _mark_struggling(session, threshold - 1)
    assert detect_confusion_spike(session) is None
    _mark_struggling(session, threshold)
    assert detect_confusion_spike(session)["struggling_count"] == threshold


def test_confusion_spike_counts_yellow_and_red_only():
    session = make_session(6)
    session.student_by_name("student0").status = "red"
    session.student_by_name("student1").status = "yellow"
    session.student_by_name("student2").status = "green"
    assert detect_confusion_spike(session) is None
    session.student_by_name("student2").status = "red"
    assert detect_confusion_spike(session)["struggling_count"] == 3


def test_process_telemetry_attaches_confusion_spike(session_factory):
    session = session_factory(3)
    for name in ("student0", "student1"):
        session.student_by_name(name).status = "yellow"
    actions = process_telemetry(session, student_id(session, "student2"), make_event("help", message="lost"))
    # An unanswered help request makes student2 yellow, completing the spike.
    assert actions["confusion_spike"]["struggling_count"] == 3
    # Typing again clears student2, so the class is no longer spiking.
    actions = process_telemetry(session, student_id(session, "student2"), make_event("keystroke"))
    assert "confusion_spike" not in actions


# ── confusion spike dedup window ─────────────────────────────────────


def test_confusion_spike_carries_timestamp():
    session = make_session(4)
    _mark_struggling(session, 3)
    spike = detect_confusion_spike(session)
    assert spike["timestamp"] == pytest.approx(now_ts(), abs=2)


def test_confusion_episode_alerts_once_until_it_ends():
    session = make_session(4)
    _mark_struggling(session, 3)
    t0 = 1_000_000.0
    assert track_confusion_episode(session, now=t0) is not None
    # Same ongoing episode, however long it lasts: no re-alert.
    for dt in (5, 31, 120, 900):
        assert track_confusion_episode(session, now=t0 + dt) is None
    # Students recover -> episode ends.
    for s in session.students.values():
        s.status = "green"
    assert track_confusion_episode(session, now=t0 + 1000) is None
    assert session.confusion_episode_active is False
    # Flapping straight back is not a new episode yet...
    _mark_struggling(session, 3)
    assert track_confusion_episode(session, now=t0 + 1001) is None
    # ...but after the quiet period a fresh episode alerts again.
    assert track_confusion_episode(session, now=t0 + 1000 + telemetry.CONFUSION_SPIKE_QUIET_SECONDS) is not None


def test_confusion_spike_counts_only_currently_stuck():
    session = make_session(4)
    _mark_struggling(session, 3)
    # Two students were stuck earlier but are green now: not counted.
    for name in ("student0", "student1"):
        s = session.student_by_name(name)
        s.status = "green"
        s.help_requests.extend(["help"] * 5)
        s.hints_given = 3
    assert detect_confusion_spike(session) is None


def test_confusion_spike_message_suggests_concrete_action():
    session = make_session(4)
    _mark_struggling(session, 3)
    stuck = session.student_by_name("student0")
    stuck.help_requests.append("I don't get the for loop")
    stuck.last_support_at = now_ts()
    spike = detect_confusion_spike(session)
    assert "stuck right now" in spike["message"]
    assert "pause and re-explain" in spike["message"]
    assert "for loop" in spike["message"]


def test_confusion_spike_generic_help_falls_back_to_current_step():
    session = make_session(4)
    _mark_struggling(session, 3)
    session.student_by_name("student0").help_requests.append("Student is confused")
    spike = detect_confusion_spike(session)
    assert "pause and re-explain the current step" in spike["message"]


def test_module_constants_are_as_documented():
    assert telemetry.IDLE_WARNING_SECONDS == 60
    assert telemetry.IDLE_CRITICAL_SECONDS == 120
    assert telemetry.PASTE_LENGTH_THRESHOLD == 200
    assert telemetry.BACKSPACE_RATE_THRESHOLD == 0.35
    assert telemetry.CONFUSION_SPIKE_MIN_STUDENTS == 3
    assert telemetry.CONFUSION_SPIKE_RATIO == 0.5
    assert telemetry.PAUSE_HINT_COOLDOWN_SECONDS == 45


# ── status = "needs attention now": recovery in every path ─────────────


def _stall(session, student):
    """Drive a student to red via a long idle."""
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=IDLE_CRITICAL_SECONDS + 5))
    assert student.status == "red"


def test_status_recovers_after_resumed_typing(session, student):
    _stall(session, student)
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", count=1))
    assert student.status == "green"


def test_status_recovers_after_meaningful_code_update_without_keystrokes(session, student):
    _stall(session, student)
    student.set_code("x = 1")
    process_telemetry(session, student_id(session, "student0"), make_event("code_update", code="x = 1\nprint(x)"))
    assert student.idle_seconds == 0
    assert student.status == "green"


def test_unchanged_code_update_does_not_clear_a_stall(session, student):
    _stall(session, student)
    idle_before = student.idle_seconds
    process_telemetry(session, student_id(session, "student0"), make_event("code_update", code=student.current_code))
    assert student.idle_seconds == idle_before
    assert student.status == "red"


def test_status_recovers_after_three_hints(session, student):
    _stall(session, student)
    student.hint_level = 3
    student.hints_given = 3
    student.last_support_at = now_ts()
    _update_status(student)
    assert student.status in ("yellow", "red")
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", count=1))
    assert student.status == "green"
    assert student.hint_level == 3  # recovery does not erase hint history


def test_status_recovers_after_large_paste(session, student):
    _stall(session, student)
    process_telemetry(session, student_id(session, "student0"), make_event("paste", length=PASTE_LENGTH_THRESHOLD * 5))
    assert student.status == "red"  # the paste itself changes nothing
    student.set_code("x")
    process_telemetry(session, student_id(session, "student0"), make_event("code_update", code="x = [1, 2, 3]"))
    assert student.status == "green"


def test_help_request_never_lowers_scores_or_locks_status(session, student):
    start_work(student)
    student.quiz_score = 100
    student.progress = 40.0
    frustration_before = student.frustration_score
    process_telemetry(session, student_id(session, "student0"), make_event("help", message="stuck on loops"))
    assert student.quiz_score == 100
    assert student.progress == 40.0
    assert student.frustration_score >= frustration_before
    assert student.status == "yellow"  # attention needed *now*
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", count=1))
    assert student.status == "green"
    assert student.quiz_score == 100
