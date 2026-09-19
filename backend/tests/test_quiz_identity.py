"""Quiz results are keyed by the stable student_id; display names are labels only."""

from __future__ import annotations

import pytest

import session_manager
import test_api
from test_api import LONG_PARAGRAPHS, make_pdf, sh, sid_of, stu, th

client = test_api.client  # re-export the httpx fixture (token bookkeeping lives in test_api)

pytestmark = pytest.mark.asyncio


async def quiz_session(client) -> str:
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    sid = r.json()["session_id"]
    assert (await client.post(f"/api/sessions/{sid}/generate-quiz", headers=th(sid))).status_code == 200
    return sid


async def submit(client, sid: str, name: str, answers=None):
    return await client.post(
        f"/api/sessions/{sid}/submit-quiz",
        json={"student_id": sid_of(sid, name), "answers": answers or {}},
        headers=sh(sid, name),
    )


async def test_duplicate_submission_is_a_readable_409_and_keeps_the_first_result(client):
    sid = await quiz_session(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann"})
    quiz = session_manager.get_session(sid).quiz
    all_correct = {str(i): q["correct"] for i, q in enumerate(quiz)}

    first = await submit(client, sid, "Ann", all_correct)
    assert first.status_code == 200 and first.json()["score"] == 100.0

    second = await submit(client, sid, "Ann", {})
    assert second.status_code == 409
    body = second.json()
    assert isinstance(body["detail"], str) and "already submitted" in body["detail"]

    stored = session_manager.get_session(sid).quiz_results[sid_of(sid, "Ann")]
    assert stored["score"] == 100.0  # not silently overwritten
    assert stu(sid, "Ann").quiz_score == 100.0


async def test_two_students_with_the_same_display_name_keep_separate_results(client):
    sid = await quiz_session(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alex"})
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alex (2)"})
    quiz = session_manager.get_session(sid).quiz
    all_correct = {str(i): q["correct"] for i, q in enumerate(quiz)}

    first_id, second_id = sid_of(sid, "Alex"), sid_of(sid, "Alex (2)")
    assert (await submit(client, sid, "Alex", all_correct)).status_code == 200
    assert (await submit(client, sid, "Alex (2)", {})).status_code == 200

    # Both students end up displayed as "Alex" (e.g. a rename) yet stay distinct records.
    session = session_manager.get_session(sid)
    session.students[second_id].name = "Alex"
    session.quiz_results[second_id]["student_name"] = "Alex"

    r = await client.get(f"/api/sessions/{sid}/report", headers=th(sid))
    report = r.json()
    rows = {row["student_id"]: row for row in report["quiz_results"]}
    assert set(rows) == {first_id, second_id}
    assert rows[first_id]["score"] == 100.0 and rows[second_id]["score"] == 0.0
    assert rows[first_id]["student_name"] == rows[second_id]["student_name"] == "Alex"
    assert report["analytics"]["quiz_submissions"] == 2
    assert report["analytics"]["quiz_accuracy"] == 50.0


async def test_rename_does_not_lose_quiz_results(client):
    sid = await quiz_session(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann"})
    ann_id = sid_of(sid, "Ann")
    assert (await submit(client, sid, "Ann", {})).status_code == 200

    session_manager.get_session(sid).students[ann_id].name = "Ann B."
    r = await client.get(f"/api/sessions/{sid}/report", headers=th(sid))
    rows = r.json()["quiz_results"]
    assert [row["student_id"] for row in rows] == [ann_id]
    assert r.json()["students"][0]["quiz_score"] == 0.0


async def test_legacy_name_keyed_results_are_migrated_not_crashed(client):
    sid = await quiz_session(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann"})
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Bob"})
    ann_id = sid_of(sid, "Ann")

    # Simulate a blob persisted before results were id-keyed: name keys, a student who
    # has since left, and a non-dict entry.
    session = session_manager.get_session(sid)
    session.quiz_results = {
        "Ann": {"score": 75.0, "correct": 3, "total": 4, "results": []},
        "Ghost": {"score": 10.0, "correct": 0, "total": 4, "results": []},
        "Weird": 42,
    }
    session_manager.persist_session(session)
    session_manager.reset_cache()

    r = await client.get(f"/api/sessions/{sid}/report", headers=th(sid))
    assert r.status_code == 200
    report = r.json()
    rows = {row["student_id"]: row for row in report["quiz_results"]}
    assert rows[ann_id]["student_name"] == "Ann" and rows[ann_id]["score"] == 75.0
    assert rows["Ghost"]["score"] == 10.0  # unmatched legacy data is kept, never dropped
    assert rows["Weird"]["student_name"] == "Weird"
    assert "Ann" not in rows
    assert stu(sid, "Ann").quiz_score == 75.0
    assert report["analytics"]["quiz_submissions"] == 1

    # Ann's legacy result counts as her one submission; Bob can still submit.
    assert (await submit(client, sid, "Ann", {})).status_code == 409
    assert (await submit(client, sid, "Bob", {})).status_code == 200
    assert set(session_manager.get_session(sid).quiz_results) == {ann_id, sid_of(sid, "Bob"), "Ghost", "Weird"}
