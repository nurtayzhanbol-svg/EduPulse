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


@pytest_asyncio.fixture
async def client():
    assert not ai_engine.is_ai_available(), "tests must run on the mock AI path"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


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
    assert len(sid) == 8
    assert body["join_url"] == f"/student.html?session={sid}"

    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    assert r.status_code == 200
    assert r.json() == {"status": "joined", "student_name": "Alice"}

    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Bob"})
    assert r.status_code == 200

    r = await client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    state = r.json()
    assert state["active"] is True
    assert state["task_level"] == "easy"
    assert state["pause_threshold_seconds"] == 60
    assert state["student_count"] == 2
    assert set(state["students"]) == {"Alice", "Bob"}
    assert state["has_material"] is False

    r = await client.post(f"/api/sessions/{sid}/end")
    assert r.status_code == 200
    ended = r.json()
    assert "summary" in ended
    analytics = ended["analytics"]
    assert analytics["total_students"] == 2
    assert analytics["total_hints"] == 0
    assert analytics["on_track_students"] == 2
    assert analytics["avg_understanding_score"] == 100.0
    assert analytics["insights"][0] == "No students required hints in this session."
    assert [b["label"] for b in analytics["bars"]][:2] == ["On-Track Students", "Struggling (>=1 hint)"]

    r = await client.get(f"/api/sessions/{sid}")
    assert r.json()["active"] is False

    r = await client.get(f"/api/sessions/{sid}/report")
    assert r.status_code == 200
    report = r.json()
    assert report["session_id"] == sid
    assert report["task_level"] == "easy"
    assert report["analytics"] == analytics
    assert report["counts"] == {"mastered": 2, "partial": 0, "struggling": 0, "incomplete": 0}
    assert report["percentages"]["mastered"] == 100
    assert report["duration_minutes"] == 1  # floor of 60s
    assert report["timeline"]["labels"] == ["0"]
    assert report["timeline"]["data"] == [0]
    assert [s["name"] for s in report["students"]] == ["Bob", "Alice"]
    assert report["hardest_topics"][0]["name"] in {"Sum", "List"}


async def test_join_is_idempotent_per_name(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Alice"})
    r = await client.get(f"/api/sessions/{sid}")
    assert r.json()["student_count"] == 1


async def test_join_ended_session_is_404(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/end")
    r = await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Late"})
    assert r.status_code == 404


async def test_report_before_end_builds_analytics_on_the_fly(client):
    sid = await create(client)
    r = await client.get(f"/api/sessions/{sid}/report")
    assert r.status_code == 200
    assert r.json()["analytics"]["total_students"] == 0
    assert r.json()["summary"] == "Session completed."


async def test_report_reflects_student_hints_and_status(client):
    sid = await create(client)
    await client.post(f"/api/sessions/{sid}/join", json={"student_name": "Zed"})
    session = session_manager.get_session(sid)
    session.students["Zed"].hints_given = 3
    session.students["Zed"].status = "red"
    r = await client.get(f"/api/sessions/{sid}/report")
    report = r.json()
    assert report["counts"] == {"mastered": 0, "partial": 1, "struggling": 0, "incomplete": 0}
    assert report["students"][0] == {"name": "Zed", "hints": 3, "status": "red", "idle_seconds": 0.0}
    assert report["analytics"]["critical_students"] == 1
    assert report["analytics"]["avg_understanding_score"] == 25.0


async def test_list_sessions_includes_created_session(client):
    sid = await create(client, task_description="x" * 100)
    r = await client.get("/api/sessions")
    assert r.status_code == 200
    entry = next(s for s in r.json() if s["session_id"] == sid)
    assert entry["active"] is True
    assert entry["student_count"] == 0
    assert len(entry["task_description"]) == 80


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
    )
    assert r.status_code == 400


async def test_generate_quiz_without_material_is_400(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/generate-quiz")
    assert r.status_code == 400
    assert r.json()["detail"] == "Upload class PDF material before generating quiz questions."


async def test_submit_quiz_without_quiz_is_400(client):
    sid = await create(client)
    r = await client.post(f"/api/sessions/{sid}/submit-quiz", json={"student_name": "A", "answers": {}})
    assert r.status_code == 400
    assert r.json()["detail"] == "No quiz available"


# ── PDF upload / create-from-pdf (mock AI path) ───────────────────


async def test_upload_pdf_happy_path(client):
    sid = await create(client)
    r = await client.post(
        f"/api/sessions/{sid}/upload-pdf",
        files={"file": ("material.pdf", make_pdf(LONG_PARAGRAPHS), "application/pdf")},
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
    assert len(sid) == 8
    assert body["join_url"] == f"/student.html?session={sid}"
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
    r = await client.post(f"/api/sessions/{sid}/generate-quiz", params={"num_questions": 3})
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
    quiz = (await client.post(f"/api/sessions/{sid}/generate-quiz")).json()["questions"]
    answers = {str(i): q["correct"] for i, q in enumerate(quiz)}
    answers["0"] = "ZZZ"  # miss the first one
    r = await client.post(f"/api/sessions/{sid}/submit-quiz", json={"student_name": "Ann", "answers": answers})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == len(quiz)
    assert body["correct"] == len(quiz) - 1
    assert body["score"] == round((len(quiz) - 1) / len(quiz) * 100)
    assert body["results"][0]["is_correct"] is False


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
    r = await client.post(f"/api/sessions/{sid}/generate-quiz")
    assert r.status_code == 502


# ── Static routes ─────────────────────────────────────────────────


async def test_root_serves_index_html(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


async def test_unknown_html_page_is_404(client):
    r = await client.get("/does-not-exist.html")
    assert r.status_code == 404
