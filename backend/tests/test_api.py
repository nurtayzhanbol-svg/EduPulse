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
STUDENT_IDS: dict[tuple[str, str], str] = {}


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
        STUDENT_IDS[(sid, body["student_name"])] = body["student_id"]


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


def sid_of(sid: str, name: str) -> str:
    return STUDENT_IDS[(sid, name)]


async def create(client, task_description="Sum a list", task_level="medium") -> str:
    r = await client.post("/api/sessions", json={"task_description": task_description, "task_level": task_level})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


# ── Happy path ────────────────────────────────────────────────────


async def test_full_happy_path(client):
    r = await client.post("/api/sessions", json={"task_description": "Sum a list", "task_level": "easy"})
    assert r.status_code == 200
    body = r.json()
    sid = body["session_id"]
    assert len(sid) == 16  # token_urlsafe(12)
    assert body["join_url"] == f"/student.html?session={sid}"
    assert len(body["teacher_token"]) >= 32

    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    assert r.status_code == 200
    joined = r.json()
    assert joined["status"] == "joined" and joined["student_name"] == "Alice"
    assert len(joined["student_token"]) >= 32

    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Bob"})
    assert r.status_code == 200

    r = await client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    state = r.json()
    assert state["active"] is True
    assert state["task_level"] == "easy"
    assert state["pause_threshold_seconds"] == 60
    assert state["student_count"] == 2
    assert {s["name"] for s in state["students"].values()} == {"Alice", "Bob"}
    assert state["has_material"] is False

    r = await client.post(f"/api/sessions/{sid}/end", headers=th(sid))
    assert r.status_code == 200
    ended = r.json()
    assert "summary" in ended
    analytics = ended["analytics"]
    assert analytics["total_students"] == 2
    assert analytics["total_hints"] == 0
    assert analytics["quiz_avg_correct_pct"] is None
    assert analytics["quiz_submitted_count"] == 0
    assert analytics["help_request_count"] == 0
    assert "avg_understanding_score" not in analytics
    assert analytics["insights"][0] == "No quiz evidence yet — run a quiz to see what the class actually got right."
    assert [b["label"] for b in analytics["bars"]][:2] == ["Quiz submitted", "Quiz avg correct"]

    r = await client.get(f"/api/sessions/{sid}")
    assert r.json()["active"] is False

    r = await client.get(f"/api/sessions/{sid}/report", headers=th(sid))
    assert r.status_code == 200
    report = r.json()
    assert report["session_id"] == sid
    assert report["task_level"] == "easy"
    assert report["analytics"] == analytics
    assert "counts" not in report and "percentages" not in report
    assert report["evidence_note"] == "No evidence yet"
    assert report["duration_minutes"] == 1  # floor of 60s
    assert report["timeline"]["labels"] == ["0"]
    assert report["timeline"]["data"] == [0]
    assert [s["name"] for s in report["students"]] == ["Alice", "Bob"]
    assert report["hardest_topics"][0]["name"] in {"Sum", "List"}


async def test_join_is_idempotent_per_name_with_token(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    token = r.json()["student_token"]
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"}, headers=sh(sid, "Alice"))
    assert r.status_code == 200
    assert r.json()["student_token"] == token
    r = await client.get(f"/api/sessions/{sid}")
    assert r.json()["student_count"] == 1


async def test_duplicate_name_without_token_creates_fresh_student(client):
    sid = await create(client)
    first = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    second = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    assert first.status_code == second.status_code == 200
    assert first.json()["student_id"] != second.json()["student_id"]
    assert (await client.get(f"/api/sessions/{sid}")).json()["student_count"] == 2


async def test_join_token_reuses_student_and_bogus_token_creates_fresh_student(client):
    sid = await create(client)
    first = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alex"})
    first_body = first.json()
    reused = await client.post(
        f"/api/sessions/{sid}/join",
        json={"student_name": "Different display name"},
        headers={"Authorization": f"Bearer {first_body['student_token']}"},
    )
    assert reused.status_code == 200
    assert reused.json()["student_id"] == first_body["student_id"]
    bogus = await client.post(
        f"/api/sessions/{sid}/join",
        json={"student_name": "Alex"},
        headers={"Authorization": "Bearer bogus-token"},
    )
    assert bogus.status_code == 200
    assert bogus.json()["student_id"] != first_body["student_id"]


async def test_join_blank_name_is_400(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "   "})
    assert r.status_code == 400


async def test_join_ended_session_is_404(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/end", headers=th(sid))
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Late"})
    assert r.status_code == 404


async def test_report_before_end_builds_analytics_on_the_fly(client):
    sid = await create(client)
    r = await client.get(f"/api/sessions/{sid}/report", headers=th(sid))
    assert r.status_code == 200
    assert r.json()["analytics"]["total_students"] == 0
    assert r.json()["summary"] == "Session completed."


async def test_report_reflects_student_hints_and_status(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Zed"})
    session = session_manager.get_session(sid)
    next(s for s in session.students.values() if s.name == "Zed").hints_given = 3
    next(s for s in session.students.values() if s.name == "Zed").status = "red"
    r = await client.get(f"/api/sessions/{sid}/report", headers=th(sid))
    report = r.json()
    # Hints alone are never evidence of what the student understood.
    assert "counts" not in report and "percentages" not in report
    assert report["students"][0] == {
        "student_id": next(s for s in session.students.values() if s.name == "Zed").student_id,
        "name": "Zed", "hints": 3, "status": "red", "idle_seconds": 0.0,
        "help_requests": 0, "quiz": None,
    }
    assert report["analytics"]["quiz_avg_correct_pct"] is None
    assert next(s for s in session.students.values() if s.name == "Zed").to_dict()["quiz_score"] is None


async def test_help_requests_bar_uses_explicit_requests(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Flagged"})
    session = session_manager.get_session(sid)
    next(s for s in session.students.values() if s.name == "Flagged").hints_given = 1
    next(s for s in session.students.values() if s.name == "Flagged").status = "red"
    analytics = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()["analytics"]
    assert analytics["help_request_count"] == 0
    assert [b["label"] for b in analytics["bars"]] == ["Quiz submitted", "Quiz avg correct", "Help requests", "Hints given"]


async def test_support_signals_count_hints_and_explicit_help_requests(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Zed"})
    session = session_manager.get_session(sid)
    zed = next(s for s in session.students.values() if s.name == "Zed")
    zed.hints_given = 2
    zed.help_requests = ["why does this loop stop?"]
    live = (await client.get(f"/api/sessions/{sid}")).json()["students"][zed.student_id]
    assert live["support_signals"] == 3
    assert live["help_requests_count"] == 1
    assert live["quiz_score"] is None
    assert "understanding_score" not in live
    report = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()
    assert report["students"][0]["hints"] == 2
    assert report["students"][0]["help_requests"] == 1
    assert report["analytics"]["help_request_count"] == 1


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
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
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
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann"})
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Bob"})
    body = {"student_id": sid_of(sid, "Ann"), "student_name": "Ann", "answers": {}}
    assert (await client.post(f"/api/sessions/{sid}/submit-quiz", json=body)).status_code == 401
    assert (await client.post(f"/api/sessions/{sid}/submit-quiz", json=body, headers=BAD)).status_code == 403
    # Bob may not submit as Ann.
    r = await client.post(f"/api/sessions/{sid}/submit-quiz", json=body, headers=sh(sid, "Bob"))
    assert r.status_code == 403
    # A student who never joined cannot submit at all.
    r = await client.post(
        f"/api/sessions/{sid}/submit-quiz", json={"student_id": "ghost", "answers": {}}, headers=sh(sid, "Bob"),
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
    r = await client.post("/api/sessions/deadbeef/join", json={"student_name": "A"})
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
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "A"})
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
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Ann"})
    r = await client.post(
        f"/api/sessions/{sid}/submit-quiz",
        json={"student_id": sid_of(sid, "Ann"), "student_name": "Ann", "answers": answers},
        headers=sh(sid, "Ann"),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == len(quiz)
    assert body["correct"] == len(quiz) - 1
    assert body["score"] == round((len(quiz) - 1) / len(quiz) * 100)
    assert body["results"][0]["is_correct"] is False

    student = next(s for s in session_manager.get_session(sid).students.values() if s.name == "Ann")
    assert student.to_dict()["quiz_score"] == body["score"]
    assert student.to_dict()["quiz_correct"] == body["correct"]
    analytics = (await client.get(f"/api/sessions/{sid}/report", headers=th(sid))).json()["analytics"]
    assert analytics["quiz_submitted_count"] == 1
    assert analytics["quiz_avg_correct_pct"] == body["score"]


async def test_quiz_submission_is_once_per_student_but_duplicate_names_can_submit(client):
    r = await client.post(
        "/api/sessions/create-from-pdf",
        files={"file": ("lesson.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
    )
    sid = r.json()["session_id"]
    await client.post(f"/api/sessions/{sid}/generate-quiz", headers=th(sid))
    first = (await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alex"})).json()
    second = (await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alex"})).json()
    first_submit = await client.post(
        f"/api/sessions/{sid}/submit-quiz",
        json={"student_id": first["student_id"], "answers": {}},
        headers={"Authorization": f"Bearer {first['student_token']}"},
    )
    assert first_submit.status_code == 200
    duplicate = await client.post(
        f"/api/sessions/{sid}/submit-quiz",
        json={"student_id": first["student_id"], "answers": {}},
        headers={"Authorization": f"Bearer {first['student_token']}"},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "Quiz already submitted"
    second_submit = await client.post(
        f"/api/sessions/{sid}/submit-quiz",
        json={"student_id": second["student_id"], "answers": {}},
        headers={"Authorization": f"Bearer {second['student_token']}"},
    )
    assert second_submit.status_code == 200


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
