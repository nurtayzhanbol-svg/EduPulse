"""No mastery labels: frontend/README must not promise an understanding metric."""

from __future__ import annotations

from pathlib import Path

import pytest

from models import StudentState

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_FILES = [REPO_ROOT / "README.md"] + sorted((REPO_ROOT / "frontend").glob("*.html"))

BANNED = ("understanding", "mastery", "comprehension")


@pytest.mark.parametrize("path", TEXT_FILES, ids=lambda p: p.name)
def test_no_mastery_language_in_user_facing_text(path):
    text = path.read_text(encoding="utf-8").lower()
    for word in BANNED:
        assert word not in text, f"{word!r} found in {path.name}"


def test_student_payload_has_no_mastery_keys():
    d = StudentState("x").to_dict()
    assert "frustration_score" not in d
    assert not any("understanding" in key for key in d)
