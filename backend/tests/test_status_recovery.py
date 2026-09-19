"""Attention status: derived from current idle + recent help, and recoverable."""

from __future__ import annotations

import json

from models import StudentState
from telemetry import process_telemetry

from conftest import make_event, now_ts, start_work, student_id


def test_help_marks_attention_until_student_types(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("help", message="stuck"))
    assert student.status == "yellow"
    assert student.attention_reason == "help"
    assert student.needs_attention_since is not None
    d = student.to_dict()
    assert d["seconds_stuck"] >= 0
    assert student.support_signals == 1
    assert student.frustration_score == 0.0

    for _ in range(19):
        process_telemetry(session, student_id(session, "student0"), make_event("keystroke"))
    assert student.status == "yellow"

    process_telemetry(session, student_id(session, "student0"), make_event("keystroke"))
    assert student.status == "green"
    assert student.attention_reason == ""
    assert student.needs_attention_since is None


def test_help_attention_recovers_on_code_change(session, student):
    start_work(student)
    student.current_code = "x = 1"
    process_telemetry(session, student_id(session, "student0"), make_event("help", message="stuck", current_answer="x = 1"))
    assert student.status == "yellow"

    # Small edit is not enough.
    process_telemetry(session, student_id(session, "student0"), make_event("code_update", code="x = 2"))
    assert student.status == "yellow"

    process_telemetry(
        session, student_id(session, "student0"),
        make_event("code_update", code="x = 1\nfor i in range(10):\n    print(i)"),
    )
    assert student.status == "green"


def test_idle_yellow_then_red_then_recovers_on_keystroke(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=90))
    assert student.status == "yellow"
    assert student.attention_reason == "idle"

    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=180))
    assert student.status == "red"

    process_telemetry(session, student_id(session, "student0"), make_event("keystroke"))
    assert student.status == "green"


def test_help_plus_idle_is_red(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("help", message="stuck"))
    process_telemetry(session, student_id(session, "student0"), make_event("idle", idle_seconds=90))
    assert student.status == "red"


def test_help_recent_but_not_idle_is_yellow(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("help", message="stuck"))
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke", count=3))
    assert student.status == "yellow"


def test_stale_help_request_does_not_hold_status(session, student):
    start_work(student)
    student.last_help_at = now_ts() - 181
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke"))
    assert student.status == "green"


def test_hints_never_affect_status(session, student):
    start_work(student)
    student.hints_given = 3
    student.hint_level = 3
    process_telemetry(session, student_id(session, "student0"), make_event("keystroke"))
    assert student.status == "green"


def test_large_paste_never_affects_status(session, student):
    start_work(student)
    process_telemetry(session, student_id(session, "student0"), make_event("paste", length=500))
    assert student.status == "green"


def test_to_dict_has_no_mastery_metrics():
    d = StudentState("x").to_dict()
    assert "frustration_score" not in d
    assert "understanding" not in json.dumps(d)
