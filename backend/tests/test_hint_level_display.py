"""Hint levels are 1-based end to end: the first hint is Level 1 / Concept Hint."""

from __future__ import annotations

from pathlib import Path

from conftest import make_event, start_work, student_id
from telemetry import process_telemetry

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def test_first_auto_hint_is_level_one(session, student):
    start_work(student)
    actions = process_telemetry(session, student_id(session), make_event("idle", idle_seconds=90))
    assert actions["force_hint_level"] == 1


def test_student_page_maps_level_one_to_the_concept_hint():
    student_html = (FRONTEND / "student.html").read_text()
    assert "'Level ' + (idx + 1)" in student_html
    assert "'Level ' + (level + 1)" not in student_html


def test_teacher_feed_does_not_shift_the_level():
    teacher_html = (FRONTEND / "teacher.html").read_text()
    assert "Level ${data.level || 1} hint" in teacher_html
    assert "(data.level || 0) + 1" not in teacher_html
