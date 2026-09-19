"""Integration tests for the FastAPI app via httpx.ASGITransport (no network, no AI key)."""

from __future__ import annotations

import fitz  # PyMuPDF
import httpx
import pytest
import pytest_asyncio

import ai_engine
import session_manager
from main import app

pytestmark = pytest.mark.asyncio


LONG_PARAGRAPHS = [
    "Python lists are ordered, mutable collections that can hold items of any type. "
    "They support indexing, slicing, appending and in-place sorting.",
    "A for loop iterates over any iterable object. Combined with range() it is the most "
    "common way to repeat an action a fixed number of times in Python programs.",
    "Functions are defined with the def keyword. They accept positional and keyword "
    "arguments, may return values, and help organise code into reusable pieces.",
]


def make_pdf(paragraphs: list[str]) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for para in paragraphs:
        page.insert_textbox(fitz.Rect(72, y, 540, y + 120), para, fontsize=11)
        y += 130
    data = doc.tobytes()
    doc.close()
    return data


# Raw tokens handed out by the API during a test, so helpers can authenticate later.
TEACHER_TOKENS: dict[str, str] = {}
STUDENT_TOKENS: dict[tuple[str, str], str] = {}


async def _remember_tokens(response: httpx.Response) -> None:
    if response.status_code != 200 or response.request.method != "POST":
        return
    await response.aread()
    path = response.request.url.path
    if path in ("/api/sessions", "/api/sessions/create-from-pdf"):
        body = response.json()
        TEACHER_TOKENS[body["session_id"]] = body["teacher_token"]
    elif path.endswith("/join"):
        body = response.json()
        sid = path.split("/")[3]
        STUDENT_TOKENS[(sid, body["student_name"])] = body["student_token"]


@pytest_asyncio.fixture
async def client():
    assert not ai_engine.is_ai_available(), "tests must run on the mock AI path"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", event_hooks={"response": [_remember_tokens]},
    ) as c:
        yield c


def th(sid: str) -> dict:
    """Teacher auth headers for a session created through ``client``."""
    return {"Authorization": f"Bearer {TEACHER_TOKENS[sid]}"}


def sh(sid: str, name: str) -> dict:
    """Student auth headers for a student who joined through ``client``."""
    return {"Authorization": f"Bearer {STUDENT_TOKENS[(sid, name)]}"}


def stu(sid: str, name: str):
    """The StudentState of ``name`` in session ``sid``."""
    return session_manager.get_session(sid).student_by_name(name)


def sid_of(sid: str, name: str) -> str:
    return stu(sid, name).student_id


async def create(client, task_description="Sum a list", task_level="medium") -> str:
    r = await client.post("/api/sessions", json={"task_description": task_description, "task_level": task_level})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


async def launch(client, sid: str) -> None:
    """PDF-created sessions stay closed to students until the teacher launches them."""
    r = await client.post(f"/api/sessions/{sid}/launch", headers=th(sid), json={})
    assert r.status_code == 200, r.text


# ── Happy path ────────────────────────────────────────────────────


async def test_full_happy_path(client):
    r = await client.post("/api/sessions", json={"task_description": "Sum a list", "task_level": "easy"})
    assert r.status_code == 200
    body = r.json()
    sid = body["session_id"]
    assert len(sid) == 16  # token_urlsafe(12)
    assert body["join_url"] == f"/student.html?session={sid}"
    assert len(body["teacher_token"]) >= 32

    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True})
    assert r.status_code == 200
    joined = r.json()
    assert joined["status"] == "joined" and joined["student_name"] == "Alice"
    assert len(joined["student_token"]) >= 32

    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Bob", "consent": True})
    assert r.status_code == 200

    r = await client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    state = r.json()
    assert state["active"] is True
    assert state["task_level"] == "easy"
    assert state["pause_threshold_seconds"] == 60
    assert state["student_count"] == 2
    assert "students" not in state  # unauthenticated view carries no student data
    assert state["has_material"] is False
    state = (await client.get(f"/api/sessions/{sid}", headers=th(sid))).json()
    assert set(state["students"]) == {sid_of(sid, "Alice"), sid_of(sid, "Bob")}
    assert {s["name"] for s in state["students"].values()} == {"Alice", "Bob"}

    r = await client.post(f"/api/sessions/{sid}/end", headers=th(sid))
    assert r.status_code == 200
    ended = r.json()
    assert "summary" in ended
    analytics = ended["analytics"]
    assert analytics["total_students"] == 2
    assert analytics["total_hints"] == 0
    assert analytics["help_request_count"] == 0
    assert analytics["quiz_avg_correct_pct"] is None
    assert analytics["quiz_submitted_count"] == 0
    assert "avg_understanding_score" not in analytics
    assert analytics["insights"][0] == (
        "No quiz evidence yet — run a quiz to see what the class actually got right."
    )
    assert [b["label"] for b in analytics["bars"]] == [
        "Quiz submitted", "Quiz avg correct", "Help requests", "Hints given",
    ]

    r = await client.get(f"/api/sessions/{sid}")
    assert r.json()["active"] is False

    r = await client.get(f"/api/sessions/{sid}/report", headers=th(sid))
    assert r.status_code == 200
    report = r.json()
    assert report["session_id"] == sid
    assert report["task_level"] == "easy"
    assert report["analytics"] == analytics
    assert "counts" not in report
    assert "percentages" not in report
    assert report["evidence_note"] == "No evidence yet"
    assert report["duration_minutes"] == 1  # floor of 60s
    assert report["timeline"]["labels"] == ["0"]
    assert report["timeline"]["data"] == [0]
    assert [s["name"] for s in report["students"]] == ["Alice", "Bob"]
    assert all(s["quiz"] is None for s in report["students"])
    assert all("understanding_score" not in s for s in report["students"])
    assert "hardest_topics" not in report
    assert report["missed_questions"] == []
    assert "none yet" in ended["summary"]
    assert "confused" not in ended["summary"].lower()
    assert "Nobody asked for help" in ended["summary"]


async def test_join_is_idempotent_per_name_with_token(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True})
    token = r.json()["student_token"]
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True}, headers=sh(sid, "Alice"))
    assert r.status_code == 200
    assert r.json()["student_token"] == token
    r = await client.get(f"/api/sessions/{sid}")
    assert r.json()["student_count"] == 1


async def test_join_taken_name_without_token_is_409(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True})
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True})
    assert r.status_code == 409
    r = await client.post(
        f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True}, headers={"Authorization": "Bearer nope"},
    )
    assert r.status_code == 409


async def test_join_blank_name_is_400(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "   ", "consent": True})
    assert r.status_code == 400


async def test_join_ended_session_is_404(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/end", headers=th(sid))
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Late", "consent": True})
    assert r.status_code == 404


async def test_report_before_end_builds_analytics_on_the_fly(client):
    sid = await create(client)
    r = await client.get(f"/api/sessions/{sid}/report", headers=th(sid))
    assert r.status_code == 200
    report = r.json()
    assert report["analytics"] == {
        "total_students": 0,
        "total_hints": 0,
        "help_request_count": 0,
        "quiz_submitted_count": 0,
        "quiz_avg_correct_pct": None,
        "total_large_pastes": 0,
        "teacher_intervention_count": 0,
        "bars": [],
        "insights": [],
    }
    assert report["summary"] == "Session completed."


async def test_report_reflects_student_hints_and_status(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Zed", "consent": True})
    session = session_manager.get_session(sid)
    session.student_by_name("Zed").hints_given = 3
    session.student_by_name("Zed").status = "red"
    r = await client.get(f"/api/sessions/{sid}/report", headers=th(sid))
    report = r.json()
    assert "counts" not in report
    assert "percentages" not in report
    assert report["evidence_note"] == "No evidence yet"
    assert report["students"][0] == {
        "student_id": session.student_by_name("Zed").student_id,
        "name": "Zed", "hints": 3, "status": "red", "idle_seconds": 0.0,
        "help_requests": 0, "teacher_nudges": 0, "quiz": None,
        "steps_done": 0, "steps_total": 0, "progress": 0.0,
    }
    assert report["analytics"]["quiz_avg_correct_pct"] is None
    assert report["analytics"]["help_request_count"] == 0
    assert session.student_by_name("Zed").to_dict()["quiz_score"] is None


async def test_support_signals_count_hints_and_explicit_help_requests(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Zed", "consent": True})
    session = session_manager.get_session(sid)
    zed = session.student_by_name("Zed")
    zed.hints_given = 2
    zed.help_requests = ["why does this loop stop?"]
    live = (await client.get(f"/api/sessions/{sid}", headers=th(sid))).json()["students"][zed.student_id]
    assert live["support_signals"] == 3
    assert live["help_requests_count"] == 1
    assert live["quiz_score"] is None
    assert "understanding_score" not in live
    report = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()
    assert "support_signals" not in report["students"][0]
    assert report["analytics"]["help_request_count"] == 1
    assert report["analytics"]["total_hints"] == 2


async def test_list_sessions_returns_only_the_callers_session(client):
    other = await create(client)
    sid = await create(client, task_description="x" * 100)
    r = await client.get("/api/sessions", headers=th(sid))
    assert r.status_code == 200
    assert [s["session_id"] for s in r.json()] == [sid]
    entry = r.json()[0]
    assert entry["active"] is True
    assert entry["student_count"] == 0
    assert len(entry["task_description"]) == 80
    assert other not in [s["session_id"] for s in r.json()]


# ── Auth: 401 without token, 403 with the wrong one ───────────────


BAD = {"Authorization": "Bearer definitely-not-the-token"}


async def test_end_without_token_is_401(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/end")
    assert r.status_code == 401
    assert (await client.get(f"/api/sessions/{sid}")).json()["active"] is True


async def test_end_with_wrong_token_is_403(client):
    sid = await create(client)
    other = await create(client)
    r = await client.post(f"/api/sessions/{sid}/end", headers=BAD)
    assert r.status_code == 403
    r = await client.post(f"/api/sessions/{sid}/end", headers=th(other))  # another teacher's token
    assert r.status_code == 403
    assert (await client.get(f"/api/sessions/{sid}")).json()["active"] is True


async def test_end_with_malformed_authorization_header_is_401(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/end", headers={"Authorization": "Basic abc"})
    assert r.status_code == 401
    r = await client.post(f"/api/sessions/{sid}/end", headers={"Authorization": "Bearer "})
    assert r.status_code == 401


async def test_report_requires_teacher_token(client):
    sid = await create(client)
    assert (await client.get(f"/api/sessions/{sid}/report")).status_code == 401
    assert (await client.get(f"/api/sessions/{sid}/report", headers=BAD)).status_code == 403


async def test_generate_quiz_requires_teacher_token(client):
    sid = await create(client)
    assert (await client.post(f"/api/sessions/{sid}/generate-quiz")).status_code == 401
    assert (await client.post(f"/api/sessions/{sid}/generate-quiz", headers=BAD)).status_code == 403


async def test_upload_pdf_requires_teacher_token(client):
    sid = await create(client)
    files = {"file": ("m.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")}
    assert (await client.post(f"/api/sessions/{sid}/upload-pdf", files=files)).status_code == 401
    assert (await client.post(f"/api/sessions/{sid}/upload-pdf", files=files, headers=BAD)).status_code == 403
    assert (await client.get(f"/api/sessions/{sid}")).json()["has_material"] is False


async def test_list_sessions_requires_teacher_token(client):
    await create(client)
    assert (await client.get("/api/sessions")).status_code == 401
    assert (await client.get("/api/sessions", headers=BAD)).status_code == 403


async def test_student_token_is_not_a_teacher_token(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True})
    assert (await client.post(f"/api/sessions/{sid}/end", headers=sh(sid, "Alice"))).status_code == 403
    assert (await client.get(f"/api/sessions/{sid}/report", headers=sh(sid, "Alice"))).status_code == 403


async def test_teacher_token_is_never_stored_raw(client):
    sid = await create(client)
    session = session_manager.get_session(sid)
    assert TEACHER_TOKENS[sid] not in vars(session).values()
    assert len(session.teacher_token_hash) == 64


async def test_submit_quiz_requires_matching_student_token(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    sid = r.json()["session_id"]
    await client.post(f"/api/sessions/{sid}/generate-quiz", headers=th(sid))
    await launch(client, sid)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann", "consent": True})
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Bob", "consent": True})
    body = {"student_id": sid_of(sid, "Ann"), "answers": {}}
    assert (await client.post(f"/api/sessions/{sid}/submit-quiz", json=body)).status_code == 401
    assert (await client.post(f"/api/sessions/{sid}/submit-quiz", json=body, headers=BAD)).status_code == 403
    # Bob may not submit as Ann.
    r = await client.post(f"/api/sessions/{sid}/submit-quiz", json=body, headers=sh(sid, "Bob"))
    assert r.status_code == 403
    # A student who never joined cannot submit at all.
    r = await client.post(
        f"/api/sessions/{sid}/submit-quiz", json={"student_id": "ghost-id", "answers": {}}, headers=sh(sid, "Bob"),
    )
    assert r.status_code == 403
    assert session_manager.get_session(sid).quiz_results == {}
    r = await client.post(f"/api/sessions/{sid}/submit-quiz", json=body, headers=sh(sid, "Ann"))
    assert r.status_code == 200


# ── 404s ──────────────────────────────────────────────────────────


async def test_get_unknown_session_404(client):
    r = await client.get("/api/sessions/deadbeef")
    assert r.status_code == 404
    assert r.json()["detail"] == "Session not found"


async def test_join_unknown_session_404(client):
    r = await client.post("/api/sessions/deadbeef/join", json={"student_name": "A", "consent": True})
    assert r.status_code == 404


async def test_end_unknown_session_404(client):
    r = await client.post("/api/sessions/deadbeef/end")
    assert r.status_code == 404


async def test_report_unknown_session_404(client):
    r = await client.get("/api/sessions/deadbeef/report")
    assert r.status_code == 404


async def test_upload_pdf_unknown_session_404(client):
    r = await client.post(
        "/api/sessions/deadbeef/upload-pdf",
        files={"file": ("m.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    assert r.status_code == 404


async def test_generate_quiz_unknown_session_404(client):
    r = await client.post("/api/sessions/deadbeef/generate-quiz")
    assert r.status_code == 404


# ── 400s ──────────────────────────────────────────────────────────


async def test_upload_non_pdf_is_400(client):
    sid = await create(client)
    r = await client.post(
        f"/api/sessions/{sid}/upload-pdf",
        files={"file": ("notes.txt", b"hello " * 100, "text/plain")},
        headers=th(sid),
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "Only PDF files are supported"


async def test_create_from_non_pdf_is_400(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("notes.docx", b"hello " * 100, "application/octet-stream")},
    )
    assert r.status_code == 400


async def test_upload_pdf_under_20_words_is_400(client):
    sid = await create(client)
    short_pdf = make_pdf(["one two three four five six seven eight nine ten"])
    r = await client.post(
        f"/api/sessions/{sid}/upload-pdf",
        files={"file": ("short.pdf", short_pdf, "application/pdf")},
        headers=th(sid),
    )
    assert r.status_code == 400
    assert "enough text" in r.json()["detail"]


async def test_create_from_pdf_under_20_words_is_400(client):
    short_pdf = make_pdf(["just a few words here"])
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("short.pdf", short_pdf, "application/pdf")},
    )
    assert r.status_code == 400


async def test_upload_empty_pdf_is_400(client):
    sid = await create(client)
    r = await client.post(
        f"/api/sessions/{sid}/upload-pdf",
        files={"file": ("blank.pdf", make_pdf([]), "application/pdf")},
        headers=th(sid),
    )
    assert r.status_code == 400


async def test_generate_quiz_without_material_is_400(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/generate-quiz", headers=th(sid))
    assert r.status_code == 400
    assert r.json()["detail"] == "Upload class PDF material before generating quiz questions."


async def test_submit_quiz_without_quiz_is_400(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "A", "consent": True})
    r = await client.post(
        f"/api/sessions/{sid}/submit-quiz", json={"student_id": sid_of(sid, "A"), "answers": {}}, headers=sh(sid, "A"),
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "No quiz available"


# ── PDF upload / create-from-pdf (mock AI path) ───────────────────


async def test_upload_pdf_happy_path(client):
    sid = await create(client)
    r = await client.post(
        f"/api/sessions/{sid}/upload-pdf",
        files={"file": ("material.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
        headers=th(sid),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "material.pdf"
    assert body["pages"] == 1
    assert body["word_count"] >= 60
    assert isinstance(body["analysis"], str) and body["analysis"]

    r = await client.get(f"/api/sessions/{sid}")
    assert r.json()["has_material"] is True


async def test_create_from_pdf_returns_session_and_task(client):
    pdf = make_pdf(LONG_PARAGRAPHS)
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", pdf, "application/pdf")},
        data={"task_level": "hard", "mode": "practical", "quiz_difficulty": "easy"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    sid = body["session_id"]
    assert len(sid) == 16
    assert body["join_url"] == f"/student.html?session={sid}"
    assert len(body["teacher_token"]) >= 32
    assert body["task_description"].startswith("Task:")
    assert body["task_description"].count("\n") == 3  # Task / Input / Output / Edge Case
    assert 2 <= len(body["task_steps"]) <= 3
    assert all(isinstance(s, str) and s for s in body["task_steps"])
    assert body["filename"] == "lesson.pdf"
    assert body["pages"] == 1
    assert body["word_count"] >= 60
    assert body["mode"] == "practical"
    assert body["difficulty"] == "easy"
    assert isinstance(body["analysis"], str) and body["analysis"]

    r = await client.get(f"/api/sessions/{sid}")
    state = r.json()
    assert state["task_level"] == "hard"
    assert state["pause_threshold_seconds"] == 120
    assert state["has_material"] is True
    assert state["task_description"] == body["task_description"]
    assert state["task_steps"] == body["task_steps"]


async def test_create_from_pdf_theoretical_mode(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
        data={"mode": "theoretical"},
    )
    assert r.status_code == 200
    assert r.json()["mode"] == "theoretical"
    assert "Explain" in r.json()["task_description"]


async def test_create_from_pdf_then_generate_quiz_uses_mock(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    sid = r.json()["session_id"]
    r = await client.post(f"/api/sessions/{sid}/generate-quiz", params={"num_questions": 3}, headers=th(sid))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == len(body["questions"]) > 0
    for q in body["questions"]:
        assert set(q) >= {"question", "options", "correct"}
        assert q["correct"] in q["options"]


async def test_submit_quiz_grades_answers(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    sid = r.json()["session_id"]
    quiz = (await client.post(f"/api/sessions/{sid}/generate-quiz", headers=th(sid))).json()["questions"]
    answers = {str(i): q["correct"] for i, q in enumerate(quiz)}
    answers["0"] = "ZZZ"  # miss the first one
    await launch(client, sid)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann", "consent": True})
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Bob", "consent": True})
    r = await client.post(
        f"/api/sessions/{sid}/submit-quiz",
        json={"student_id": sid_of(sid, "Ann"), "answers": answers},
        headers=sh(sid, "Ann"),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == len(quiz)
    assert body["correct"] == len(quiz) - 1
    assert body["score"] == round((len(quiz) - 1) / len(quiz) * 100)
    assert body["results"][0]["is_correct"] is False

    student = stu(sid, "Ann")
    assert student.to_dict()["quiz_score"] == body["score"]
    assert student.to_dict()["quiz_correct"] == body["correct"]
    analytics = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()["analytics"]
    assert analytics["quiz_submitted_count"] == 1
    assert analytics["quiz_avg_correct_pct"] == body["score"]
    report = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()
    assert report["evidence_note"] is None
    assert report["students"][0]["quiz"] == {
        "correct": body["correct"],
        "total": body["total"],
        "score": body["score"],
    }
    assert report["students"][1]["quiz"] is None
    assert report["missed_questions"] == [
        {"index": 1, "question": quiz[0]["question"], "missed": 1, "submitted": 1},
    ]


# ── Task steps & progress ─────────────────────────────────────────


async def pdf_session_with_students(client, *names: str) -> str:
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    sid = r.json()["session_id"]
    await launch(client, sid)
    for name in names:
        await client.post(f"/api/sessions/{sid}/join", json={"student_name": name, "consent": True})
    return sid


async def test_marking_steps_done_drives_progress(client):
    sid = await pdf_session_with_students(client, "Ann")
    steps = session_manager.get_session(sid).task_steps
    n = len(steps)
    ann = sid_of(sid, "Ann")

    r = await client.post(f"/api/sessions/{sid}/steps/done", json={"student_id": ann, "step": 0}, headers=sh(sid, "Ann"))
    assert r.status_code == 200, r.text
    assert r.json() == {"completed_steps": [0], "total_steps": n, "progress": round(100 / n, 1)}

    # idempotent
    r = await client.post(f"/api/sessions/{sid}/steps/done", json={"student_id": ann, "step": 0}, headers=sh(sid, "Ann"))
    assert r.json()["completed_steps"] == [0]

    for i in range(1, n):
        await client.post(f"/api/sessions/{sid}/steps/done", json={"student_id": ann, "step": i}, headers=sh(sid, "Ann"))
    state = (await client.get(f"/api/sessions/{sid}")).json()["students"][ann]
    assert state["progress"] == 100.0
    assert state["completed_steps"] == list(range(n))


async def test_marking_an_unknown_step_is_400(client):
    sid = await pdf_session_with_students(client, "Ann")
    n = len(session_manager.get_session(sid).task_steps)
    for bad in (n, -1):
        r = await client.post(
            f"/api/sessions/{sid}/steps/done", json={"student_id": sid_of(sid, "Ann"), "step": bad}, headers=sh(sid, "Ann"),
        )
        assert r.status_code == 400
    assert stu(sid, "Ann").progress == 0.0


async def test_marking_a_step_requires_the_students_own_token(client):
    sid = await pdf_session_with_students(client, "Ann", "Bob")
    r = await client.post(f"/api/sessions/{sid}/steps/done", json={"student_id": sid_of(sid, "Ann"), "step": 0})
    assert r.status_code == 401
    r = await client.post(
        f"/api/sessions/{sid}/steps/done", json={"student_id": sid_of(sid, "Ann"), "step": 0}, headers=sh(sid, "Bob"),
    )
    assert r.status_code == 403
    assert stu(sid, "Ann").progress == 0.0


async def test_marking_a_step_after_the_session_ended_is_409(client):
    sid = await pdf_session_with_students(client, "Ann")
    await client.post(f"/api/sessions/{sid}/end", headers=th(sid))
    r = await client.post(f"/api/sessions/{sid}/steps/done", json={"student_id": sid_of(sid, "Ann"), "step": 0}, headers=sh(sid, "Ann"))
    assert r.status_code == 409


async def test_session_without_steps_has_no_progress_to_mark(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann", "consent": True})
    r = await client.post(f"/api/sessions/{sid}/steps/done", json={"student_id": sid_of(sid, "Ann"), "step": 0}, headers=sh(sid, "Ann"))
    assert r.status_code == 400


async def test_completed_steps_survive_a_restart(client):
    sid = await pdf_session_with_students(client, "Ann")
    await client.post(f"/api/sessions/{sid}/steps/done", json={"student_id": sid_of(sid, "Ann"), "step": 1}, headers=sh(sid, "Ann"))
    steps = session_manager.get_session(sid).task_steps
    session_manager.reset_cache()
    session = session_manager.get_session(sid)
    assert session.task_steps == steps
    ann = session.student_by_name("Ann")
    assert ann.completed_steps == [1]
    assert ann.progress == 100 / len(steps)


async def test_editing_the_steps_resets_student_progress(client):
    sid = await pdf_session_with_students(client, "Ann")
    await client.post(f"/api/sessions/{sid}/steps/done", json={"student_id": sid_of(sid, "Ann"), "step": 0}, headers=sh(sid, "Ann"))
    r = await client.patch(
        f"/api/sessions/{sid}/task",
        json={"task_description": "Task: new", "task_steps": ["Do A", "  Do B ", "", "Do C", "Do D"]},
        headers=th(sid),
    )
    assert r.status_code == 200
    assert r.json()["task_steps"] == ["Do A", "Do B", "Do C"]
    assert stu(sid, "Ann").completed_steps == []
    assert stu(sid, "Ann").progress == 0.0


async def test_report_progress_comes_from_steps(client):
    sid = await pdf_session_with_students(client, "Ann")
    n = len(session_manager.get_session(sid).task_steps)
    await client.post(f"/api/sessions/{sid}/steps/done", json={"student_id": sid_of(sid, "Ann"), "step": 0}, headers=sh(sid, "Ann"))
    report = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()
    row = report["students"][0]
    assert (row["steps_done"], row["steps_total"]) == (1, n)
    assert row["progress"] == round(100 / n, 1)


# ── Evidence-based summary ────────────────────────────────────────


async def submit(client, sid: str, name: str, answers: dict) -> None:
    r = await client.post(
        f"/api/sessions/{sid}/submit-quiz", json={"student_id": sid_of(sid, name), "answers": answers}, headers=sh(sid, name),
    )
    assert r.status_code == 200, r.text


async def test_summary_names_quiz_scores_help_askers_and_missed_questions(client):
    sid = await pdf_session_with_students(client, "Ann", "Bob", "Cid")
    quiz = (await client.post(f"/api/sessions/{sid}/generate-quiz", headers=th(sid))).json()["questions"]
    right = {str(i): q["correct"] for i, q in enumerate(quiz)}
    await submit(client, sid, "Ann", right)
    await submit(client, sid, "Bob", {**right, "1": "ZZZ"})
    await submit(client, sid, "Cid", {**right, "1": "ZZZ", "0": "ZZZ"})
    session = session_manager.get_session(sid)
    session.student_by_name("Bob").help_requests.append("how do I start?")
    session.student_by_name("Cid").hints_given = 3  # hints alone must not label anyone
    session.student_by_name("Cid").idle_seconds = 400

    summary = (await client.post(f"/api/sessions/{sid}/end", headers=th(sid))).json()["summary"]
    assert "3 of 3 students submitted" in summary
    assert f"- Ann: {len(quiz)}/{len(quiz)} correct (100%)" in summary
    assert "Q2 missed by 2 of 3 — re-teach: " in summary
    assert "Q1 missed by 1 of 3 — re-teach: " in summary
    assert summary.index("Q2 missed") < summary.index("Q1 missed")
    assert "Asked for help: Bob (1)." in summary
    assert "Cid" not in summary.split("Asked for help")[1]
    for banned in ("confused", "struggl", "understanding", "mastery"):
        assert banned not in summary.lower()

    report = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()
    assert report["missed_questions"][0] == {"index": 2, "question": quiz[1]["question"], "missed": 2, "submitted": 3}
    assert report["missed_questions"][1]["index"] == 1
    assert report["summary"] == summary


async def test_summary_marks_students_without_a_quiz_as_no_evidence(client):
    sid = await pdf_session_with_students(client, "Ann", "Bob")
    quiz = (await client.post(f"/api/sessions/{sid}/generate-quiz", headers=th(sid))).json()["questions"]
    await submit(client, sid, "Ann", {str(i): q["correct"] for i, q in enumerate(quiz)})
    summary = (await client.post(f"/api/sessions/{sid}/end", headers=th(sid))).json()["summary"]
    assert "1 of 2 students submitted" in summary
    assert "- Bob: no quiz submitted (no evidence yet)" in summary
    assert "Most-missed" not in summary


async def test_summary_with_no_students(client):
    sid = await create(client)
    summary = (await client.post(f"/api/sessions/{sid}/end", headers=th(sid))).json()["summary"]
    assert "no evidence" in summary.lower()


# ── Input normalisation ───────────────────────────────────────────


async def test_create_session_invalid_level_falls_back_to_medium(client):
    sid = await create(client, task_level="INVALID")
    r = await client.get(f"/api/sessions/{sid}")
    assert r.json()["task_level"] == "medium"
    assert r.json()["pause_threshold_seconds"] == 90


async def test_create_session_level_is_case_insensitive(client):
    sid = await create(client, task_level="HARD")
    r = await client.get(f"/api/sessions/{sid}")
    assert r.json()["task_level"] == "hard"
    assert r.json()["pause_threshold_seconds"] == 120


async def test_create_session_defaults(client):
    r = await client.post("/api/sessions", json={})
    assert r.status_code == 200
    r = await client.get(f"/api/sessions/{r.json()['session_id']}")
    assert r.json()["task_level"] == "medium"
    assert r.json()["task_description"] == ""


async def test_create_from_pdf_normalises_invalid_inputs(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
        data={"task_level": "INVALID", "mode": "INVALID", "quiz_difficulty": "INVALID"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "practical"
    assert body["difficulty"] == "medium"  # falls back to normalized task level
    r = await client.get(f"/api/sessions/{body['session_id']}")
    assert r.json()["task_level"] == "medium"


async def test_create_from_pdf_invalid_quiz_difficulty_follows_task_level(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
        data={"task_level": "easy", "quiz_difficulty": "extreme"},
    )
    assert r.status_code == 200
    assert r.json()["difficulty"] == "easy"


async def test_create_from_pdf_uppercase_inputs_are_lowercased(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
        data={"task_level": "Hard", "mode": "THEORETICAL", "quiz_difficulty": "Easy"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "theoretical"
    assert body["difficulty"] == "easy"
    r = await client.get(f"/api/sessions/{body['session_id']}")
    assert r.json()["task_level"] == "hard"


async def test_create_from_pdf_filename_extension_check_is_case_insensitive(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("LESSON.PDF", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    assert r.status_code == 200


async def test_generate_quiz_normalises_difficulty_and_mode(client, monkeypatch):
    captured = {}

    async def fake_generate_quiz(**kwargs):
        captured.update(kwargs)
        return [{"question": "q", "options": {"A": "a"}, "correct": "A"}]

    import main as main_module
    monkeypatch.setattr(main_module, "generate_quiz", fake_generate_quiz)

    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    sid = r.json()["session_id"]
    r = await client.post(
        f"/api/sessions/{sid}/generate-quiz",
        params={"difficulty": "IMPOSSIBLE", "mode": "WEIRD", "num_questions": 2},
        headers=th(sid),
    )
    assert r.status_code == 200
    assert captured["difficulty"] == "medium"
    assert captured["mode"] == "practical"
    assert captured["num_questions"] == 2


async def test_generate_quiz_returns_502_when_generation_yields_nothing(client, monkeypatch):
    async def empty_quiz(**kwargs):
        return []

    import main as main_module
    monkeypatch.setattr(main_module, "generate_quiz", empty_quiz)
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    sid = r.json()["session_id"]
    r = await client.post(f"/api/sessions/{sid}/generate-quiz", headers=th(sid))
    assert r.status_code == 502


# ── Static routes ─────────────────────────────────────────────────


async def test_root_serves_index_html(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


async def test_unknown_html_page_is_404(client):
    r = await client.get("/does-not-exist.html")
    assert r.status_code == 404


# ── Student identity ──────────────────────────────────────────────


async def test_join_returns_a_server_generated_id_that_is_not_the_name(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True})
    body = r.json()
    student_id = body["student_id"]
    assert student_id and student_id != "Alice"
    assert list(session_manager.get_session(sid).students) == [student_id]


async def test_renaming_is_impossible_because_state_is_keyed_by_id(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True})
    student_id, token = r.json()["student_id"], r.json()["student_token"]
    stu(sid, "Alice").name = "Alice B."
    r = await client.post(
        f"/api/sessions/{sid}/join", json={"student_name": "Alice B.", "consent": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.json()["student_id"] == student_id
    assert len(session_manager.get_session(sid).students) == 1


async def test_student_id_survives_a_restart(client):
    sid = await create(client)
    student_id = (await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True})).json()["student_id"]
    session_manager.reset_cache()
    assert list(session_manager.get_session(sid).students) == [student_id]


# ── One quiz submission per student ───────────────────────────────


async def quiz_session(client) -> str:
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    sid = r.json()["session_id"]
    await client.post(f"/api/sessions/{sid}/generate-quiz", headers=th(sid))
    await launch(client, sid)
    return sid


async def test_second_quiz_submission_is_rejected(client):
    sid = await quiz_session(client)
    await launch(client, sid)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann", "consent": True})
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Bob", "consent": True})
    body = {"student_id": sid_of(sid, "Ann"), "answers": {}}

    assert (await client.post(f"/api/sessions/{sid}/submit-quiz", json=body, headers=sh(sid, "Ann"))).status_code == 200
    r = await client.post(f"/api/sessions/{sid}/submit-quiz", json=body, headers=sh(sid, "Ann"))
    assert r.status_code == 409
    assert r.json()["detail"] == "You have already submitted this quiz."

    # Another student is unaffected.
    r = await client.post(
        f"/api/sessions/{sid}/submit-quiz",
        json={"student_id": sid_of(sid, "Bob"), "answers": {}}, headers=sh(sid, "Bob"),
    )
    assert r.status_code == 200
    assert set(session_manager.get_session(sid).quiz_results) == {sid_of(sid, "Ann"), sid_of(sid, "Bob")}


async def test_quiz_results_are_keyed_by_id_and_keep_the_name(client):
    sid = await quiz_session(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann", "consent": True})
    ann_id = sid_of(sid, "Ann")
    await client.post(
        f"/api/sessions/{sid}/submit-quiz", json={"student_id": ann_id, "answers": {}}, headers=sh(sid, "Ann"),
    )
    session_manager.reset_cache()
    stored = session_manager.get_session(sid).quiz_results[ann_id]
    assert stored["student_name"] == "Ann"
    assert isinstance(stored["submitted_at"], float)


# ── Teacher review & edit of the generated task ───────────────────


async def test_teacher_can_edit_the_task_before_launch(client):
    sid = await create(client, task_description="Draft generated by the model")
    r = await client.patch(
        f"/api/sessions/{sid}/task", json={"task_description": "  Reviewed task  "}, headers=th(sid),
    )
    assert r.status_code == 200
    assert r.json()["task_description"] == "Reviewed task"
    session_manager.reset_cache()
    assert session_manager.get_session(sid).task_description == "Reviewed task"
    assert (await client.get(f"/api/sessions/{sid}")).json()["task_description"] == "Reviewed task"


async def test_task_edit_requires_the_teacher_token(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice", "consent": True})
    body = {"task_description": "Mine now"}
    assert (await client.patch(f"/api/sessions/{sid}/task", json=body)).status_code == 401
    assert (await client.patch(f"/api/sessions/{sid}/task", json=body, headers=sh(sid, "Alice"))).status_code == 403
    assert session_manager.get_session(sid).task_description != "Mine now"


async def test_task_edit_rejects_an_empty_description(client):
    sid = await create(client, task_description="Original")
    r = await client.patch(f"/api/sessions/{sid}/task", json={"task_description": "   "}, headers=th(sid))
    assert r.status_code == 400
    assert session_manager.get_session(sid).task_description == "Original"


async def test_task_edit_on_unknown_session_is_404(client):
    r = await client.patch("/api/sessions/deadbeef/task", json={"task_description": "x"}, headers=BAD)
    assert r.status_code == 404
