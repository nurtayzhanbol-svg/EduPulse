"""Unit tests for backend/telemetry.py."""

from __future__ import annotations

import pytest

import telemetry
from telemetry import (
    IDLE_CRITICAL_SECONDS,
    IDLE_WARNING_SECONDS,
    PASTE_LENGTH_THRESHOLD,
    PAUSE_HINT_COOLDOWN_SECONDS,
    _next_hint_level,
    _pause_interval_for_next_hint,
    _update_status,
    _update_understanding_score,
    compute_understanding_score,
    detect_confusion_spike,
    is_duplicate_confusion_spike,
    process_telemetry,
)
from models import StudentState

from conftest import make_event, make_session, now_ts, start_work


# ── process_telemetry: generic behaviour ──────────────────────────


def test_unknown_student_returns_empty_dict(session):
    assert process_telemetry(session, "nobody", make_event("keystroke")) == {}


def test_event_is_logged_and_activity_updated(session, student):
    before = student.last_activity
    ev = make_event("keystroke", count=2, extra="x")
    actions = process_telemetry(session, "student0", ev)
    assert actions["dashboard_update"] is True
    assert student.last_activity >= before
    assert student.events == [{"type": "keystroke", "ts": ev.timestamp, "count": 2, "extra": "x"}]


def test_unknown_event_type_only_updates_dashboard(session, student):
    actions = process_telemetry(session, "student0", make_event("mystery"))
    assert actions == {"dashboard_update": True}
    assert student.status == "green"
    assert student.understanding_score == 100.0


def test_current_code_in_payload_is_stored_for_any_event(session, student):
    process_telemetry(session, "student0", make_event("mystery", current_code="x = 1"))
    assert student.current_code == "x = 1"


def test_non_string_current_code_is_ignored(session, student):
    process_telemetry(session, "student0", make_event("keystroke", current_code=123))
    assert student.current_code == ""


# ── keystroke ─────────────────────────────────────────────────────


def test_keystroke_increments_and_resets_idle(session, student):
    student.idle_seconds = 50
    process_telemetry(session, "student0", make_event("keystroke", count=3))
    assert student.total_keystrokes == 3
    assert student.idle_seconds == 0


def test_keystroke_default_count_is_one(session, student):
    process_telemetry(session, "student0", make_event("keystroke"))
    assert student.total_keystrokes == 1


def test_keystroke_uses_key_ts_when_numeric(session, student):
    process_telemetry(session, "student0", make_event("keystroke", key_ts=1234.5))
    assert student.last_keypress_at == 1234.5


def test_keystroke_falls_back_to_now_for_non_numeric_key_ts(session, student):
    before = now_ts()
    process_telemetry(session, "student0", make_event("keystroke", key_ts="bad"))
    assert student.last_keypress_at >= before


# ── backspace ─────────────────────────────────────────────────────


def test_backspace_counts_towards_keystrokes(session, student):
    process_telemetry(session, "student0", make_event("keystroke", count=10))
    process_telemetry(session, "student0", make_event("backspace", count=2))
    assert student.total_backspaces == 2
    assert student.total_keystrokes == 12


def test_backspace_below_rate_threshold_no_frustration(session, student):
    process_telemetry(session, "student0", make_event("keystroke", count=10))
    # 3 / 13 ≈ 0.23 <= 0.35
    process_telemetry(session, "student0", make_event("backspace", count=3))
    assert student.frustration_score == 0.0


def test_backspace_above_rate_threshold_adds_frustration(session, student):
    process_telemetry(session, "student0", make_event("keystroke", count=10))
    # 7 / 17 ≈ 0.41 > 0.35
    process_telemetry(session, "student0", make_event("backspace", count=7))
    assert student.frustration_score == pytest.approx(0.1)


def test_backspace_frustration_caps_at_one(session, student):
    student.frustration_score = 0.95
    process_telemetry(session, "student0", make_event("backspace", count=1))  # rate 1.0
    assert student.frustration_score == 1.0


def test_backspace_uses_key_ts(session, student):
    process_telemetry(session, "student0", make_event("backspace", key_ts=42))
    assert student.last_keypress_at == 42


# ── idle + has_started_work guard ─────────────────────────────────


def test_idle_before_any_typing_does_not_accrue(session, student):
    actions = process_telemetry(session, "student0", make_event("idle", idle_seconds=500))
    assert student.idle_seconds == 0
    assert "should_hint" not in actions
    assert student.frustration_score == 0.0
    assert student.status == "green"
    assert student.understanding_score == 100.0


def test_idle_before_typing_resets_previously_set_idle(session, student):
    student.idle_seconds = 30
    process_telemetry(session, "student0", make_event("idle", idle_seconds=500))
    assert student.idle_seconds == 0


def test_idle_ignores_stale_keypress_before_typing(session, student):
    student.last_keypress_at = now_ts() - 1000
    process_telemetry(session, "student0", make_event("idle", idle_seconds=0))
    assert student.idle_seconds == 0


def test_code_in_editor_counts_as_started_work(session, student):
    student.current_code = "print(1)"
    process_telemetry(session, "student0", make_event("idle", idle_seconds=40))
    assert student.idle_seconds == 40


def test_whitespace_only_code_does_not_count_as_started(session, student):
    student.current_code = "   \n\t"
    process_telemetry(session, "student0", make_event("idle", idle_seconds=40))
    assert student.idle_seconds == 0


def test_idle_takes_max_of_payload_and_wall_clock_pause(session, student):
    start_work(student)
    student.last_keypress_at = now_ts() - 200
    process_telemetry(session, "student0", make_event("idle", idle_seconds=10))
    assert student.idle_seconds == pytest.approx(200, abs=2)


def test_idle_below_warning_threshold_no_frustration(session, student):
    start_work(student)
    # medium: pause_threshold 90 -> warning 45
    process_telemetry(session, "student0", make_event("idle", idle_seconds=44))
    assert student.frustration_score == 0.0


def test_idle_at_warning_threshold_adds_small_frustration(session, student):
    start_work(student)
    process_telemetry(session, "student0", make_event("idle", idle_seconds=45))
    assert student.frustration_score == pytest.approx(0.05)


def test_idle_just_below_critical_is_warning_only(session, student):
    start_work(student)
    actions = process_telemetry(session, "student0", make_event("idle", idle_seconds=89))
    assert student.frustration_score == pytest.approx(0.05)
    assert "should_hint" not in actions


def test_idle_at_critical_threshold_triggers_hint(session, student):
    start_work(student)
    actions = process_telemetry(session, "student0", make_event("idle", idle_seconds=90))
    assert actions["should_hint"] is True
    assert actions["hint_reason"] == "idle_threshold_exceeded"
    assert actions["force_hint_level"] == 1
    assert student.frustration_score == pytest.approx(0.12)
    assert student.last_pause_hint_at > 0


def test_idle_warning_threshold_floor_is_15s():
    session = make_session(1, task_level="easy")
    session.pause_threshold_seconds = 10  # below the 30s floor -> base 30, warning 15
    student = session.students["student0"]
    start_work(student)
    process_telemetry(session, "student0", make_event("idle", idle_seconds=15))
    assert student.frustration_score == pytest.approx(0.05)


# ── hint escalation ───────────────────────────────────────────────


@pytest.mark.parametrize("current,expected", [(0, 1), (1, 2), (2, 3), (3, 3), (10, 3), (-5, 1)])
def test_next_hint_level_caps_between_1_and_3(current, expected):
    s = StudentState("x")
    s.hint_level = current
    assert _next_hint_level(s) == expected


def test_idle_hint_cooldown_suppresses_second_hint(session, student):
    start_work(student)
    first = process_telemetry(session, "student0", make_event("idle", idle_seconds=300))
    assert first["should_hint"] is True
    second = process_telemetry(session, "student0", make_event("idle", idle_seconds=400))
    assert "should_hint" not in second
    assert "force_hint_level" not in second
    # frustration still accrues even without a hint
    assert student.frustration_score == pytest.approx(0.24)


def test_idle_hint_fires_again_after_cooldown(session, student):
    start_work(student)
    process_telemetry(session, "student0", make_event("idle", idle_seconds=300))
    student.last_pause_hint_at = now_ts() - PAUSE_HINT_COOLDOWN_SECONDS - 1
    student.hint_level = 1
    again = process_telemetry(session, "student0", make_event("idle", idle_seconds=300))
    assert again["should_hint"] is True
    assert again["force_hint_level"] == 2


def test_idle_hint_just_inside_cooldown_is_suppressed(session, student):
    start_work(student)
    student.last_pause_hint_at = now_ts() - (PAUSE_HINT_COOLDOWN_SECONDS - 5)
    actions = process_telemetry(session, "student0", make_event("idle", idle_seconds=300))
    assert "should_hint" not in actions


def test_force_hint_level_caps_at_three(session, student):
    start_work(student)
    student.hint_level = 3
    actions = process_telemetry(session, "student0", make_event("idle", idle_seconds=999))
    assert actions["force_hint_level"] == 3


# ── _pause_interval_for_next_hint ─────────────────────────────────


@pytest.mark.parametrize("level,base", [("easy", 60), ("medium", 90), ("hard", 120)])
@pytest.mark.parametrize("hint_level,multiplier", [(0, 1.0), (1, 1.5), (2, 2.0), (3, 2.0), (7, 2.0)])
def test_pause_interval_multipliers(level, base, hint_level, multiplier):
    session = make_session(1, task_level=level)
    assert session.pause_threshold_seconds == base
    s = session.students["student0"]
    s.hint_level = hint_level
    assert _pause_interval_for_next_hint(session, s) == int(base * multiplier)


def test_pause_interval_has_30s_floor():
    session = make_session(1)
    session.pause_threshold_seconds = 5
    assert _pause_interval_for_next_hint(session, session.students["student0"]) == 30


def test_pause_interval_falls_back_to_idle_warning_without_attribute():
    class Bare:
        pass

    s = StudentState("x")
    assert _pause_interval_for_next_hint(Bare(), s) == IDLE_WARNING_SECONDS


def test_critical_idle_threshold_scales_with_hint_level(session, student):
    """After one hint (level 1) the next idle hint needs 1.5x the base pause (135s on medium)."""
    start_work(student)
    student.hint_level = 1
    actions = process_telemetry(session, "student0", make_event("idle", idle_seconds=134))
    assert "should_hint" not in actions
    actions = process_telemetry(session, "student0", make_event("idle", idle_seconds=135))
    assert actions["should_hint"] is True
    assert actions["force_hint_level"] == 2


# ── paste / plagiarism ────────────────────────────────────────────


def test_small_paste_is_recorded_without_alert(session, student):
    actions = process_telemetry(
        session, "student0", make_event("paste", length=PASTE_LENGTH_THRESHOLD - 1, content_preview="abc")
    )
    assert "plagiarism_alert" not in actions
    assert student.paste_events[0]["length"] == PASTE_LENGTH_THRESHOLD - 1
    assert student.paste_events[0]["preview"] == "abc"
    assert student.status == "green"


def test_large_paste_raises_alert_red_status_and_zero_frustration(session, student):
    start_work(student)
    student.frustration_score = 0.7
    actions = process_telemetry(
        session, "student0", make_event("paste", length=PASTE_LENGTH_THRESHOLD, content_preview="z" * 500)
    )
    alert = actions["plagiarism_alert"]
    assert alert["student_name"] == "student0"
    assert alert["paste_length"] == PASTE_LENGTH_THRESHOLD
    assert "student0" in alert["message"]
    assert student.frustration_score == 0
    assert student.status == "red"
    assert student.paste_events[0]["preview"] == "z" * 100
    assert student.understanding_score == 80.0


def test_large_paste_red_status_expires_after_three_more_pastes(session, student):
    process_telemetry(session, "student0", make_event("paste", length=500))
    assert student.status == "red"
    for _ in range(3):
        process_telemetry(session, "student0", make_event("paste", length=5))
    assert student.status == "green"


def test_large_paste_without_typing_penalises_understanding(session, student):
    process_telemetry(session, "student0", make_event("paste", length=500))
    assert student.status == "red"
    assert student.understanding_score == pytest.approx(80.0)


# ── help ──────────────────────────────────────────────────────────


def test_help_requests_hint_and_records_message(session, student):
    actions = process_telemetry(session, "student0", make_event("help", message="stuck on loops"))
    assert actions["should_hint"] is True
    assert actions["hint_reason"] == "help_request"
    assert actions["help_message"] == "stuck on loops"
    assert student.help_requests == ["stuck on loops"]
    assert student.frustration_score == pytest.approx(0.25)


def test_help_stores_current_answer_as_code(session, student):
    process_telemetry(session, "student0", make_event("help", message="?", current_answer="for i in x:"))
    assert student.current_code == "for i in x:"


def test_help_ignores_blank_current_answer(session, student):
    student.current_code = "keep"
    process_telemetry(session, "student0", make_event("help", message="?", current_answer="   "))
    assert student.current_code == "keep"


def test_help_frustration_caps_at_one(session, student):
    for _ in range(5):
        process_telemetry(session, "student0", make_event("help", message="?"))
    assert student.frustration_score == 1.0


# ── code_update ───────────────────────────────────────────────────


def test_code_update_sets_progress_from_line_count(session, student):
    process_telemetry(session, "student0", make_event("code_update", code="a\nb\nc"))
    assert student.current_code == "a\nb\nc"
    assert student.progress == 15.0


def test_code_update_progress_caps_at_100(session, student):
    process_telemetry(session, "student0", make_event("code_update", code="\n" * 40))
    assert student.progress == 100.0


def test_code_update_with_missing_code_clears_editor(session, student):
    student.current_code = "old"
    process_telemetry(session, "student0", make_event("code_update"))
    assert student.current_code == ""
    assert student.progress == 5.0


# ── pause_wait ────────────────────────────────────────────────────


def test_pause_wait_before_typing_is_ignored(session, student):
    actions = process_telemetry(session, "student0", make_event("pause_wait", idle_seconds=999))
    assert student.idle_seconds == 0
    assert "should_hint" not in actions
    assert student.frustration_score == 0.0


def test_pause_wait_at_threshold_hints_with_reason(session, student):
    start_work(student)
    actions = process_telemetry(session, "student0", make_event("pause_wait", idle_seconds=90))
    assert actions["should_hint"] is True
    assert actions["hint_reason"] == "pause_threshold_exceeded"
    assert actions["force_hint_level"] == 1
    assert student.frustration_score == pytest.approx(0.08)


def test_pause_wait_below_threshold_no_hint(session, student):
    start_work(student)
    actions = process_telemetry(session, "student0", make_event("pause_wait", idle_seconds=89))
    assert "should_hint" not in actions
    assert student.frustration_score == 0.0
    assert student.idle_seconds == 89


def test_pause_wait_respects_cooldown(session, student):
    start_work(student)
    process_telemetry(session, "student0", make_event("pause_wait", idle_seconds=300))
    second = process_telemetry(session, "student0", make_event("pause_wait", idle_seconds=300))
    assert "should_hint" not in second
    assert student.frustration_score == pytest.approx(0.16)


def test_pause_wait_stores_current_answer(session, student):
    process_telemetry(session, "student0", make_event("pause_wait", idle_seconds=0, current_answer="x=1"))
    assert student.current_code == "x=1"


def test_pause_wait_handles_none_idle_seconds(session, student):
    start_work(student)
    process_telemetry(session, "student0", make_event("pause_wait", idle_seconds=None))
    assert student.idle_seconds == pytest.approx(0, abs=1)


# ── _update_understanding_score ───────────────────────────────────


def test_understanding_ignores_idle_before_work_but_not_other_penalties():
    s = StudentState("x")
    s.idle_seconds = 300
    _update_understanding_score(s)
    assert s.understanding_score == 100.0
    s.hints_given = 1
    s.frustration_score = 1.0
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(100 - 18 - 20)


def test_compute_understanding_score_is_pure_and_matches_update():
    s = StudentState("x")
    start_work(s)
    s.hints_given = 2
    s.idle_seconds = 150
    s.frustration_score = 0.5
    s.paste_events = [{"length": PASTE_LENGTH_THRESHOLD}]
    expected = 100 - 2 * 18 - 12.5 - 10 - 20
    assert compute_understanding_score(s) == pytest.approx(expected)
    assert s.understanding_score == 100.0  # untouched
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(expected)


def test_understanding_hint_penalty_is_18_per_hint():
    s = StudentState("x")
    start_work(s)
    s.hints_given = 2
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(64.0)


def test_understanding_idle_penalty_scales_to_25_at_300s():
    s = StudentState("x")
    start_work(s)
    s.idle_seconds = 150
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(87.5)
    s.idle_seconds = 1000  # capped at 300s -> 25
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(75.0)


def test_understanding_frustration_penalty_is_20_at_max():
    s = StudentState("x")
    start_work(s)
    s.frustration_score = 0.5
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(90.0)
    s.frustration_score = 3.0  # capped at 1.0
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(80.0)


def test_understanding_paste_penalty_only_considers_last_three_pastes():
    s = StudentState("x")
    start_work(s)
    s.paste_events = [{"length": 500}]
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(80.0)
    s.paste_events += [{"length": 1}, {"length": 1}, {"length": 1}]
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(100.0)


def test_understanding_combined_penalties_sum():
    s = StudentState("x")
    start_work(s)
    s.hints_given = 1
    s.idle_seconds = 300
    s.frustration_score = 1.0
    s.paste_events = [{"length": 200}]
    _update_understanding_score(s)
    assert s.understanding_score == pytest.approx(100 - 18 - 25 - 20 - 20)


def test_understanding_clamps_to_zero():
    s = StudentState("x")
    start_work(s)
    s.hints_given = 6  # 108 penalty
    _update_understanding_score(s)
    assert s.understanding_score == 0.0


def test_understanding_never_exceeds_100():
    s = StudentState("x")
    start_work(s)
    s.frustration_score = -5.0
    _update_understanding_score(s)
    assert s.understanding_score == 100.0


# ── _update_status ────────────────────────────────────────────────


def _status(hints=0, idle=0.0, frustration=0.0, pastes=()):
    s = StudentState("x")
    s.hints_given = hints
    s.idle_seconds = idle
    s.frustration_score = frustration
    s.paste_events = [{"length": p} for p in pastes]
    _update_status(s)
    return s.status


def test_status_green_by_default():
    assert _status() == "green"


def test_status_red_on_large_paste_regardless_of_other_state():
    assert _status(pastes=(PASTE_LENGTH_THRESHOLD,)) == "red"
    assert _status(pastes=(PASTE_LENGTH_THRESHOLD - 1,)) == "green"


def test_status_large_paste_only_checks_last_three():
    assert _status(pastes=(500, 1, 1, 1)) == "green"
    assert _status(pastes=(1, 500, 1, 1)) == "red"


@pytest.mark.parametrize("hints,expected", [(2, "yellow"), (3, "red"), (4, "red")])
def test_status_hint_count_boundary_at_three(hints, expected):
    assert _status(hints=hints) == expected


def test_status_idle_boundary_at_double_critical():
    assert _status(idle=IDLE_CRITICAL_SECONDS * 2 - 1) == "yellow"
    assert _status(idle=IDLE_CRITICAL_SECONDS * 2) == "red"


def test_status_two_hints_plus_idle_warning_is_red():
    assert _status(hints=2, idle=IDLE_WARNING_SECONDS - 1) == "yellow"
    assert _status(hints=2, idle=IDLE_WARNING_SECONDS) == "red"


def test_status_one_hint_is_yellow_even_with_idle_warning():
    assert _status(hints=1, idle=IDLE_WARNING_SECONDS) == "yellow"


def test_status_idle_critical_boundary_is_yellow():
    assert _status(idle=IDLE_CRITICAL_SECONDS - 1) == "green"
    assert _status(idle=IDLE_CRITICAL_SECONDS) == "yellow"


def test_status_high_frustration_alone_stays_green():
    """The frustration>=0.95 branch is unreachable: it also requires idle>=120,
    which the preceding branch already turns yellow. Pin current behaviour."""
    assert _status(frustration=1.0) == "green"
    assert _status(frustration=1.0, idle=IDLE_CRITICAL_SECONDS - 1) == "green"
    assert _status(frustration=1.0, idle=IDLE_CRITICAL_SECONDS) == "yellow"


def test_status_uses_global_idle_constants_not_session_threshold():
    """A hard-level session (pause 120s) and an easy one (60s) share the same status cut-offs."""
    for level in ("easy", "medium", "hard"):
        session = make_session(1, task_level=level)
        s = session.students["student0"]
        start_work(s)
        process_telemetry(session, "student0", make_event("idle", idle_seconds=119))
        assert s.status == "green", level


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
    session.students["student0"].status = "red"
    session.students["student1"].status = "yellow"
    session.students["student2"].status = "green"
    assert detect_confusion_spike(session) is None
    session.students["student2"].status = "red"
    assert detect_confusion_spike(session)["struggling_count"] == 3


def test_process_telemetry_attaches_confusion_spike(session_factory):
    session = session_factory(3)
    for name in ("student0", "student1"):
        session.students[name].status = "yellow"
    actions = process_telemetry(session, "student2", make_event("help", message="lost"))
    # help -> frustration only, no hints_given yet, so student2 stays green -> no spike
    assert "confusion_spike" not in actions
    session.students["student2"].hints_given = 1
    actions = process_telemetry(session, "student2", make_event("keystroke"))
    assert actions["confusion_spike"]["struggling_count"] == 3


# ── confusion spike dedup window ─────────────────────────────────────


def test_confusion_spike_carries_timestamp():
    session = make_session(4)
    _mark_struggling(session, 3)
    spike = detect_confusion_spike(session)
    assert spike["timestamp"] == pytest.approx(now_ts(), abs=2)


def test_confusion_spike_dedup_uses_30_second_window():
    session = make_session(4)
    t0 = 1_000_000.0
    assert is_duplicate_confusion_spike(session, now=t0) is False
    session.alerts.append({"type": "plagiarism", "timestamp": t0})
    assert is_duplicate_confusion_spike(session, now=t0) is False
    session.alerts.append({"type": "confusion_spike", "timestamp": t0})
    assert is_duplicate_confusion_spike(session, now=t0 + 29.9) is True
    assert is_duplicate_confusion_spike(session, now=t0 + 30) is False
    # A later non-spike alert does not shadow the last spike's timestamp.
    session.alerts.append({"type": "plagiarism", "timestamp": t0 + 29})
    assert is_duplicate_confusion_spike(session, now=t0 + 29) is True
    assert telemetry.CONFUSION_SPIKE_DEDUP_SECONDS == 30


def test_module_constants_are_as_documented():
    assert telemetry.IDLE_WARNING_SECONDS == 60
    assert telemetry.IDLE_CRITICAL_SECONDS == 120
    assert telemetry.PASTE_LENGTH_THRESHOLD == 200
    assert telemetry.BACKSPACE_RATE_THRESHOLD == 0.35
    assert telemetry.CONFUSION_SPIKE_MIN_STUDENTS == 3
    assert telemetry.CONFUSION_SPIKE_RATIO == 0.5
    assert telemetry.PAUSE_HINT_COOLDOWN_SECONDS == 45
