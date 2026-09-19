"""Telemetry tests for recoverable attention and neutral observations."""

from __future__ import annotations

import telemetry
from conftest import make_event, make_session, start_work, student_id
from telemetry import process_telemetry


def test_unknown_student_returns_empty_dict(session):
    assert process_telemetry(session, "missing", make_event("keystroke")) == {}


def test_help_marks_attention_without_frustration(session, student):
    start_work(student)
    actions = process_telemetry(session, student_id(session), make_event("help", message="stuck"))
    assert actions["should_hint"] is True
    assert student.status == "yellow"
    assert student.attention_reason == "help"
    assert student.frustration_score == 0.0
    assert "frustration_score" not in student.to_dict()


def test_idle_status_uses_session_threshold(session, student):
    start_work(student)
    process_telemetry(session, student_id(session), make_event("idle", idle_seconds=89))
    assert student.status == "green"
    process_telemetry(session, student_id(session), make_event("idle", idle_seconds=90))
    assert student.status == "yellow"
    process_telemetry(session, student_id(session), make_event("idle", idle_seconds=180))
    assert student.status == "red"
    assert student.attention_reason == "idle"


def test_auto_hint_is_one_shot_level_one(session, student):
    start_work(student)
    first = process_telemetry(session, student_id(session), make_event("idle", idle_seconds=90))
    second = process_telemetry(session, student_id(session), make_event("idle", idle_seconds=120))
    assert first["force_hint_level"] == 1
    assert "should_hint" not in second
    assert student.auto_hints_given == 1


def test_keystrokes_recover_help_attention(session, student):
    start_work(student)
    process_telemetry(session, student_id(session), make_event("help", message="stuck"))
    for _ in range(20):
        process_telemetry(session, student_id(session), make_event("keystroke"))
    assert student.last_help_at == 0
    assert student.status == "green"
    assert student.needs_attention_since is None


def test_code_delta_recovers_help_attention(session, student):
    start_work(student)
    student.set_code("x" * 30)
    process_telemetry(session, student_id(session), make_event("help", message="stuck"))
    student.set_code("y" * 30)
    telemetry._update_status(student, session, now=student.last_activity + 1)
    assert student.status == "yellow"
    process_telemetry(session, student_id(session), make_event("code_update", code="z" * 30))
    assert student.last_help_at == 0


def test_paste_is_neutral(session, student):
    start_work(student)
    before = student.frustration_score
    actions = process_telemetry(session, student_id(session), make_event("paste", length=500))
    assert "large_paste_alert" in actions
    assert student.frustration_score == before
    assert student.status == "green"


def test_confusion_excludes_left_room_students():
    session = make_session(3)
    now = 10_000.0
    for student in session.students.values():
        student.status = "yellow"
        student.last_activity = now
    left = session.student_by_name("student0")
    left.last_activity = now - telemetry.LEFT_ROOM_AFTER_SECONDS
    assert telemetry.detect_confusion_spike(session, now=now) is None
    left.last_activity = now - 1
    assert telemetry.detect_confusion_spike(session, now=now)["struggling_count"] == 3


def test_confusion_dedup_scans_all_alerts():
    session = make_session(3)
    now = 1_000.0
    session.alerts.extend([
        {"type": "plagiarism", "timestamp": now},
        {"type": "confusion_spike", "timestamp": now},
    ])
    assert telemetry.is_duplicate_confusion_spike(session, now=now + 299)
    assert not telemetry.is_duplicate_confusion_spike(session, now=now + 300)


def test_status_payload_keys(session, student):
    payload = student.to_dict()
    assert "frustration_score" not in payload
    assert payload["attention_reason"] == ""
    assert payload["needs_attention_since"] is None
    assert payload["seconds_stuck"] == 0
