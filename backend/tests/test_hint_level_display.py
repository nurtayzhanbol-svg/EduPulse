"""Hint levels are 1-based end to end: the first hint is Level 1 / Concept Hint."""

from __future__ import annotations

from pathlib import Path

from telemetry import _next_hint_level
from models import StudentState

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def test_first_hint_is_level_one():
    assert _next_hint_level(StudentState(name="A")) == 1


def test_student_page_maps_level_one_to_the_concept_hint():
    student_html = (FRONTEND / "student.html").read_text()
    assert "'Level ' + (idx + 1)" in student_html
    assert "'Level ' + (level + 1)" not in student_html


def test_teacher_feed_does_not_shift_the_level():
    teacher_html = (FRONTEND / "teacher.html").read_text()
    assert "Level ${data.level || 1} hint" in teacher_html
    assert "(data.level || 0) + 1" not in teacher_html
