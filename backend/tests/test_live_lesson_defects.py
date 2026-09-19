"""Regressions from the post-merge live-lesson test: student reload keeps work, edited
tasks keep consistent steps, confusion alerts resolve, and the report's follow-up
count is evidence-based and names students."""

from __future__ import annotations

import pytest

import main
import session_manager
import telemetry
from ai_engine import task_steps_from_description
from test_api import client, create, th, sh, stu, make_pdf, LONG_PARAGRAPHS  # noqa: F401 (fixtures)
from test_socket_auth import sio_spy, setup_session, keystroke, sid_of  # noqa: F401 (fixtures)
from test_teacher_control import create_from_pdf, student_payload

pytestmark = pytest.mark.asyncio


# ── 1. Student reload restores saved work ─────────────────────────


async def test_rejoin_returns_saved_code_and_completed_steps(client):  # noqa: F811
    sid = await create(client)
    session = session_manager.get_session(sid)
    session.task_steps = ["Read input", "Sum it", "Print it"]

    first = (await client.post(f"/api/sessions/{sid}/join", json=student_payload("Ann"))).json()
    assert first["current_code"] == "" and first["completed_steps"] == []
    assert first["task_steps"] == ["Read input", "Sum it", "Print it"]

    student = stu(sid, "Ann")
    student.current_code = "print(sum(map(int, input().split())))"
    r = await client.post(f"/api/sessions/{sid}/steps/done", headers=sh(sid, "Ann"),
                          json={"student_id": student.student_id, "step": 1})
    assert r.status_code == 200

    again = await client.post(f"/api/sessions/{sid}/join", headers=sh(sid, "Ann"), json=student_payload("Ann"))
    assert again.status_code == 200, again.text
    body = again.json()
    assert body["student_id"] == first["student_id"]
    assert body["current_code"] == "print(sum(map(int, input().split())))"
    assert body["completed_steps"] == [1]
    assert body["task_steps"] == ["Read input", "Sum it", "Print it"]
    # Rejoining does not disturb the saved copy.
    assert student.current_code == "print(sum(map(int, input().split())))"
    assert student.completed_steps == [1]


async def test_join_without_token_is_a_new_student_with_empty_work(client):  # noqa: F811
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json=student_payload("Ann"))
    stu(sid, "Ann").current_code = "x = 1"
    r = await client.post(f"/api/sessions/{sid}/join", json=student_payload("Ann"))
    assert r.status_code == 409  # name taken; no token, no saved work leaks
    r = await client.post(f"/api/sessions/{sid}/join", json=student_payload("Ben"))
    assert r.json()["current_code"] == "" and r.json()["completed_steps"] == []


# ── 2. Edited task ⇒ consistent steps ─────────────────────────────


async def test_launch_with_edited_task_regenerates_steps(client):  # noqa: F811
    body = await create_from_pdf(client)
    sid = body["session_id"]
    old_steps = body["task_steps"]
    edited = ("Task: Count vowels in a string.\n"
              "Input: one line of text.\n"
              "Output: the number of vowels.\n"
              "Edge case: empty string returns 0.")
    r = await client.post(f"/api/sessions/{sid}/launch", headers=th(sid), json={"task_description": edited})
    assert r.status_code == 200, r.text
    steps = r.json()["task_steps"]
    assert steps == task_steps_from_description(edited)
    assert steps != old_steps
    assert 2 <= len(steps) <= 3
    assert any("vowel" in s.lower() for s in steps)
    state = (await client.get(f"/api/sessions/{sid}")).json()
    assert state["task_steps"] == steps
    session_manager.reset_cache()
    assert session_manager.get_session(sid).task_steps == steps


async def test_launch_with_unchanged_task_keeps_generated_steps(client):  # noqa: F811
    body = await create_from_pdf(client)
    sid = body["session_id"]
    r = await client.post(f"/api/sessions/{sid}/launch", headers=th(sid),
                          json={"task_description": body["task_description"]})
    assert r.json()["task_steps"] == body["task_steps"]


async def test_launch_with_teacher_written_steps_uses_them(client):  # noqa: F811
    body = await create_from_pdf(client)
    sid = body["session_id"]
    r = await client.post(f"/api/sessions/{sid}/launch", headers=th(sid), json={
        "task_description": "Task: something else entirely",
        "task_steps": [" Write the function ", "", "Test it", "Extra", "Too many"],
    })
    assert r.status_code == 200, r.text
    assert r.json()["task_steps"] == ["Write the function", "Test it", "Extra"]
    assert session_manager.get_session(sid).task_steps == ["Write the function", "Test it", "Extra"]

    sid2 = (await create_from_pdf(client))["session_id"]
    r = await client.post(f"/api/sessions/{sid2}/launch", headers=th(sid2), json={"task_steps": ["  "]})
    assert r.status_code == 400


# ── 3. Confusion alert resolves and updates ───────────────────────


def _events(emitted, name):
    return [d for e, d, _ in emitted if e == name]


async def test_confusion_alert_updates_count_then_resolves(sio_spy):  # noqa: F811
    emitted, _ = sio_spy
    session, _, alice_token, _ = setup_session()
    for name in ("Carol", "Dave", "Erin"):
        session_manager.join_session(session.session_id, name)
    for name in ("Bob", "Carol", "Dave"):
        session.student_by_name(name).status = "yellow"
    await main.join_room("sid-a", {
        "session_id": session.session_id, "role": "student",
        "student_id": sid_of(session, "Alice"), "student_token": alice_token,
    })

    async def send():
        await main.telemetry("sid-a", keystroke(session.session_id, sid_of(session, "Alice"), alice_token, count=1))

    await send()
    alerts = [d for d in _events(emitted, "alert") if d["type"] == "confusion_spike"]
    assert len(alerts) == 1 and alerts[0]["struggling_count"] == 3
    assert _events(emitted, "confusion_update") == []

    # One more student gets stuck: the same episode, live count goes up, no new alert.
    session.student_by_name("Erin").status = "red"
    await send()
    updates = _events(emitted, "confusion_update")
    assert len(updates) == 1
    assert updates[0]["active"] is True and updates[0]["struggling_count"] == 4
    assert "4/5" in updates[0]["message"]
    assert len([d for d in _events(emitted, "alert") if d["type"] == "confusion_spike"]) == 1

    # Class recovers below threshold: the teacher is told the alert is resolved, once.
    for name in ("Bob", "Carol", "Erin"):
        session.student_by_name(name).status = "green"
    await send()
    await send()
    updates = _events(emitted, "confusion_update")
    assert len(updates) == 2
    resolved = updates[-1]
    assert resolved["active"] is False
    assert resolved["struggling_count"] == 1 and resolved["students"] == ["Dave"]
    assert session.confusion_episode_active is False
    assert session.to_dict()["confusion_episode_active"] is False


def test_process_telemetry_reports_confusion_lifecycle():
    session, _ = session_manager.create_session("Sum a list", "easy")
    ids = {}
    for name in ("A", "B", "C", "D"):
        s, _ = session_manager.join_session(session.session_id, name)
        ids[name] = s.student_id
    for name in ("B", "C", "D"):
        session.students[ids[name]].status = "yellow"
    ev = {"event_type": "keystroke", "timestamp": 0.0, "payload": {"count": 1}}
    from models import TelemetryEvent

    actions = telemetry.process_telemetry(session, ids["A"], TelemetryEvent(**ev))
    assert "confusion_spike" in actions and "confusion_update" not in actions
    actions = telemetry.process_telemetry(session, ids["A"], TelemetryEvent(**ev))
    assert "confusion_spike" not in actions and actions["confusion_update"]["struggling_count"] == 3
    session.students[ids["B"]].status = "green"
    actions = telemetry.process_telemetry(session, ids["A"], TelemetryEvent(**ev))
    assert actions["confusion_resolved"]["active"] is False
    assert actions["confusion_resolved"]["struggling_count"] == 2
    actions = telemetry.process_telemetry(session, ids["A"], TelemetryEvent(**ev))
    assert not any(k.startswith("confusion") for k in actions)


# ── 4. Follow-up is evidence-based and names students ─────────────


async def test_report_follow_up_counts_low_quiz_and_help_by_name(client):  # noqa: F811
    sid = await create(client)
    for name in ("Ann", "Ben", "Cid", "Dee"):
        await client.post(f"/api/sessions/{sid}/join", json=student_payload(name))
    session = session_manager.get_session(sid)
    for s in session.students.values():
        s.status = "green"  # nobody looks stuck behaviourally
    ann, ben, cid = stu(sid, "Ann"), stu(sid, "Ben"), stu(sid, "Cid")
    ann.quiz_score, ann.quiz_correct, ann.quiz_total = 0.0, 0, 3
    ben.help_requests.append("I don't get step 2")
    cid.quiz_score, cid.quiz_correct, cid.quiz_total = 100.0, 3, 3

    r = await client.post(f"/api/sessions/{sid}/end", headers=th(sid))
    assert r.status_code == 200, r.text
    report = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()
    fu = report["follow_up"]
    assert fu["count"] == 2
    assert fu["quiz_threshold_pct"] == 50
    assert "50%" in fu["definition"] and "help" in fu["definition"]
    assert [r["name"] for r in fu["students"]] == ["Ann", "Ben"]
    assert fu["students"][0]["reasons"] == ["quiz 0% (0/3)"]
    assert fu["students"][1]["reasons"] == ["asked for help 1 time"]
    for text in (fu["definition"], *(x for r in fu["students"] for x in r["reasons"])):
        assert "understanding" not in text.lower() and "mastery" not in text.lower()


def test_follow_up_threshold_and_status_reason():
    from models import StudentState

    ok = StudentState("Ok")
    ok.quiz_score, ok.quiz_correct, ok.quiz_total = 50.0, 1, 2
    stuck = StudentState("Stuck")
    stuck.status = "red"
    both = StudentState("Both")
    both.quiz_score, both.quiz_correct, both.quiz_total = 25.0, 1, 4
    both.help_requests.extend(["a", "b"])
    fu = main._build_follow_up([ok, stuck, both])
    assert fu["count"] == 2
    assert {r["name"]: r["reasons"] for r in fu["students"]} == {
        "Both": ["quiz 25% (1/4)", "asked for help 2 times"],
        "Stuck": ["red at session end"],
    }
