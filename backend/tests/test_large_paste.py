"""Council report §4 item 5: a large paste is a neutral, teacher-only, dismissable
observation — never an automatic cheating label, never red, never in the report."""

from __future__ import annotations

import pytest

import ai_engine
import main
import session_manager
from conftest import make_event, student_id
from telemetry import PASTE_LENGTH_THRESHOLD, process_telemetry
from test_socket_auth import sio_spy, setup_session, sid_of  # noqa: F401 (fixtures)


async def paste(session_id, student_id_, token, length, sock="sock-a"):
    await main.telemetry(sock, {
        "session_id": session_id, "student_id": student_id_, "student_token": token,
        "event": {"event_type": "paste", "payload": {"length": length}},
    })


@pytest.mark.asyncio
async def test_large_paste_alert_is_neutral_and_teacher_only(sio_spy):  # noqa: F811
    emitted, _rooms = sio_spy
    session, teacher_token, alice_token, bob_token = setup_session()
    sid_ = session.session_id
    alice_id = sid_of(session, "Alice")
    await main.join_room("sock-t", {"session_id": sid_, "role": "teacher", "teacher_token": teacher_token})
    await main.join_room("sock-a", {"session_id": sid_, "role": "student",
                                    "student_id": alice_id, "student_token": alice_token})
    emitted.clear()

    await paste(sid_, alice_id, alice_token, 5000)

    alerts = [(d, kw) for e, d, kw in emitted if e == "alert"]
    assert len(alerts) == 1
    alert, kw = alerts[0]
    assert kw == {"room": main.teacher_room(sid_)}
    assert alert["type"] == "large_paste"
    assert alert["student_id"] == alice_id
    assert alert["paste_length"] == 5000
    assert "plagiarism" not in alert["message"].lower()
    assert "risk" not in alert["message"].lower()
    assert session.alerts[-1]["type"] == "large_paste"

    # Nothing addressed to the shared room or a student room mentions the paste.
    for event, data, kw in emitted:
        target = kw.get("room") or kw.get("to")
        if target != main.teacher_room(sid_):
            assert event != "alert"
            assert "large_paste" not in str(data)

    student = session.students[alice_id]
    assert student.status == "green"
    dashboard = [d for e, d, _ in emitted if e == "dashboard_update"][-1]
    assert dashboard["students"][alice_id]["status"] == "green"


@pytest.mark.asyncio
async def test_large_paste_does_not_reset_frustration_or_mask_status(sio_spy):  # noqa: F811
    emitted, _rooms = sio_spy
    session, _teacher_token, alice_token, _ = setup_session()
    sid_ = session.session_id
    alice_id = sid_of(session, "Alice")
    await main.join_room("sock-a", {"session_id": sid_, "role": "student",
                                    "student_id": alice_id, "student_token": alice_token})
    student = session.students[alice_id]
    student.frustration_score = 0.6
    student.total_keystrokes = 10

    await paste(sid_, alice_id, alice_token, PASTE_LENGTH_THRESHOLD)
    assert student.frustration_score == 0.6
    assert student.status == "green"

    # Real signals still work after a paste.
    process_telemetry(session, alice_id, make_event("idle", idle_seconds=600))
    assert student.status != "green"


def test_plagiarism_action_no_longer_exists(session, student):
    actions = process_telemetry(session, student_id(session), make_event("paste", length=10_000))
    assert "plagiarism_alert" not in actions
    assert actions["large_paste_alert"]["paste_length"] == 10_000
    assert student.frustration_score == 0.0


@pytest.mark.asyncio
async def test_report_has_no_plagiarism_section():
    session, _ = session_manager.create_session("Sum a list", "easy")
    alice, _ = session_manager.join_session(session.session_id, "Alice")
    process_telemetry(session, alice.student_id, make_event("paste", length=5000))
    process_telemetry(session, alice.student_id, make_event("paste", length=5000))
    alice.quiz_score = 90

    summary = await ai_engine.generate_session_summary(session)
    low = summary.lower()
    assert "plagiarism" not in low
    assert "integrity" not in low
    assert "large paste" not in low
    assert "plagiarism" not in ai_engine.SUMMARY_SYSTEM_PROMPT.split("Do not")[0].lower()


# ── teacher confirmation gates the report ─────────────────────────────


@pytest.mark.asyncio
async def test_paste_counts_in_report_only_after_teacher_confirms():
    import httpx

    session, teacher_token = session_manager.create_session("Sum a list", "easy")
    alice, alice_token = session_manager.join_session(session.session_id, "Alice")
    process_telemetry(session, alice.student_id, make_event("paste", length=5000))
    process_telemetry(session, alice.student_id, make_event("paste", length=6000))
    sid_ = session.session_id
    headers = {"Authorization": f"Bearer {teacher_token}"}

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert main._build_session_analytics(session)["total_large_pastes"] == 0

        r = await client.post(f"/api/sessions/{sid_}/paste-signals/confirm", headers=headers,
                              json={"student_id": alice.student_id, "paste_length": 5000, "timestamp": 1.0})
        assert r.status_code == 200 and r.json()["confirmed_count"] == 1
        # Confirming the same observation twice is idempotent.
        r = await client.post(f"/api/sessions/{sid_}/paste-signals/confirm", headers=headers,
                              json={"student_id": alice.student_id, "paste_length": 5000, "timestamp": 1.0})
        assert r.json()["confirmed_count"] == 1
        assert main._build_session_analytics(session)["total_large_pastes"] == 1

        # Students cannot confirm, and unknown students are rejected.
        r = await client.post(f"/api/sessions/{sid_}/paste-signals/confirm",
                              headers={"Authorization": f"Bearer {alice_token}"},
                              json={"student_id": alice.student_id})
        assert r.status_code in (401, 403)
        r = await client.post(f"/api/sessions/{sid_}/paste-signals/confirm", headers=headers,
                              json={"student_id": "nobody"})
        assert r.status_code == 404

        r = await client.post(f"/api/sessions/{sid_}/end", headers=headers)
        assert r.status_code == 200
        assert r.json()["analytics"]["total_large_pastes"] == 1


def test_teacher_ui_paste_alert_is_dismissable_and_confirmable():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[2] / "frontend" / "teacher.html").read_text()
    assert "paste-signals/confirm" in html
    assert 'aria-label="Dismiss alert"' in html
    for banned in ("plagiarism", "cheating", "High risk"):
        assert banned.lower() not in html.lower(), banned


@pytest.mark.asyncio
async def test_public_session_endpoint_does_not_expose_paste_alerts():
    import httpx

    session, teacher_token = session_manager.create_session("Sum a list", "easy")
    alice, _ = session_manager.join_session(session.session_id, "Alice")
    session.alerts.append({"type": "large_paste", "student_name": "Alice", "paste_length": 500, "message": "x"})
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"/api/sessions/{session.session_id}")
        assert r.status_code == 200
        assert "alerts" not in r.json()
        assert "large_paste" not in r.text
        r = await client.get(f"/api/sessions/{session.session_id}",
                             headers={"Authorization": f"Bearer {teacher_token}"})
        assert r.json()["alerts"][0]["type"] == "large_paste"
