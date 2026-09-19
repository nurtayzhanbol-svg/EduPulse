"""EduPulse — Main server: FastAPI + Socket.IO."""

from __future__ import annotations
import asyncio
import math
from contextlib import asynccontextmanager
import os
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

import socketio
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, Response

from models import (
    CreateSessionRequest, CreateSessionResponse,
    JoinSessionRequest, EndSessionResponse, TelemetryEvent,
    UpdateTaskRequest, LaunchSessionRequest, NudgeRequest,
)
import session_manager
from auth import bearer_token, require_student, require_teacher, token_matches
from telemetry import is_duplicate_confusion_spike, process_telemetry, refresh_status
from ai_engine import (
    gate_hint,
    generate_hint,
    generate_quiz,
    analyze_pdf_content,
    generate_task_description_from_pdf,
)
from pdf_engine import extract_text_from_pdf, max_upload_bytes, save_upload_to_tempfile

# ── App setup ──────────────────────────────────────────────────────

RETENTION_SWEEP_SECONDS = 15 * 60


async def _retention_sweeper() -> None:
    """Purge ended sessions past the retention TTL while the process runs."""
    while True:
        await asyncio.sleep(RETENTION_SWEEP_SECONDS)
        try:
            session_manager.cleanup_expired_sessions()
        except Exception as exc:  # keep sweeping; a failed pass is retried next tick
            print(f"[Retention] cleanup failed: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    session_manager.cleanup_expired_sessions()
    sweeper = asyncio.create_task(_retention_sweeper())
    try:
        yield
    finally:
        sweeper.cancel()


def ai_provider_name() -> str:
    """Which third party processes student code for hints, derived from env (mirrors ai_engine)."""
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "none (mock hints)"
    return "Azure OpenAI" if os.environ.get("AZURE_OPENAI_ENDPOINT") else "OpenAI"


app = FastAPI(title="EduPulse", version="1.0.0", lifespan=lifespan)

# Socket.IO server (ASGI mode). Browser origins allowed to open a socket are
# read from ALLOWED_ORIGINS (comma-separated); same-origin pages are always fine.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if o.strip()
]
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=ALLOWED_ORIGINS)
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Socket sid -> (session_id, student_id) for sockets that authenticated via join_room.
_student_sockets: dict[str, tuple[str, str]] = {}


def teacher_room(session_id: str) -> str:
    """Room holding only authenticated teacher sockets of a session."""
    return f"{session_id}:teachers"


def student_room(session_id: str, student_id: str) -> str:
    """Room holding the sockets of one student. The plain ``session_id`` room is
    reserved for events every student should receive (quiz_available, session_ended)."""
    return f"{session_id}:student:{student_id}"


async def broadcast_dashboard(session) -> None:
    """Send dashboard state, including live student code, to teachers only."""
    await sio.emit(
        "dashboard_update",
        session.to_dict(include_code=True),
        room=teacher_room(session.session_id),
    )


# Serve frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Multipart framing (boundaries + form fields) on top of the PDF itself.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    """Reject an oversized upload from its Content-Length before the body is read at all."""
    if request.method == "POST" and (
        request.url.path.endswith("/upload-pdf") or request.url.path.endswith("/create-from-pdf")
    ):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit():
            limit = max_upload_bytes()
            if int(declared) > limit + _MULTIPART_OVERHEAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"PDF is too large. Maximum upload size is {limit // (1024 * 1024)} MB."},
                )
    return await call_next(request)


async def _extract_uploaded_pdf(file: UploadFile) -> dict:
    """Stream the upload to disk (enforcing the size cap) and extract its text."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    tmp_path = await save_upload_to_tempfile(file)
    try:
        result = extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)
    if not result["text"].strip() or result["word_count"] < 20:
        raise HTTPException(
            400,
            "Could not extract enough text from this PDF. Please upload a text-based PDF (not scanned images only).",
        )
    return result


def _build_session_analytics(session) -> dict:
    _normalize_quiz_results(session)
    students = list(session.students.values())
    total_students = len(students)
    if total_students == 0:
        return {
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
    total_hints = sum(s.hints_given for s in students)
    help_request_count = sum(len(s.help_requests) for s in students)
    quiz_submitted = [s for s in students if s.quiz_score is not None]
    quiz_submitted_count = len(quiz_submitted)
    quiz_pcts = [
        100 * s.quiz_correct / s.quiz_total
        for s in quiz_submitted
        if s.quiz_total
    ]
    quiz_avg_correct_pct = round(sum(quiz_pcts) / len(quiz_pcts), 1) if quiz_pcts else None
    total_large_pastes = sum(
        1 for s in students for p in s.paste_events if p.get("length", 0) >= 200
    )
    students_with_help = sum(1 for s in students if s.help_requests)
    teacher_intervention_count = sum(len(s.teacher_nudges) for s in students)
    bars = [
        {"label": "Quiz submitted", "value": float(quiz_submitted_count), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Quiz avg correct", "value": quiz_avg_correct_pct if quiz_avg_correct_pct is not None else 0.0, "max": 100.0, "unit": "%" if quiz_avg_correct_pct is not None else " (no evidence yet)"},
        {"label": "Help requests", "value": float(help_request_count), "max": max(1.0, float(help_request_count)), "unit": ""},
        {"label": "Hints given", "value": float(total_hints), "max": max(1.0, float(total_hints)), "unit": ""},
    ]
    insights = [
        "No quiz evidence yet — run a quiz to see what the class actually got right."
        if quiz_avg_correct_pct is None
        else (
            f"Quiz: {quiz_avg_correct_pct}% correct on average across "
            f"{quiz_submitted_count}/{total_students} submissions."
            + (" Start next class with a focused recap and worked example."
               if quiz_avg_correct_pct <= 40 else "")
        ),
        f"{help_request_count} explicit help request(s) from {students_with_help} student(s)."
        if help_request_count > 0
        else "No explicit help requests in this session.",
    ]
    return {
        "total_students": total_students,
        "total_hints": total_hints,
        "help_request_count": help_request_count,
        "quiz_submitted_count": quiz_submitted_count,
        "quiz_avg_correct_pct": quiz_avg_correct_pct,
        "total_large_pastes": total_large_pastes,
        "teacher_intervention_count": teacher_intervention_count,
        "bars": bars,
        "insights": insights,
    }


def _build_report_payload(session) -> dict:
    quiz_results = _normalize_quiz_results(session)
    analytics = getattr(session, "analytics", None) or _build_session_analytics(session)
    students = list(session.students.values())

    start_ts = float(getattr(session, "created_at", datetime.now().timestamp()))
    end_ts = float(getattr(session, "ended_at", datetime.now().timestamp()) or datetime.now().timestamp())
    duration_seconds = max(60.0, end_ts - start_ts)
    bucket_seconds = 4 * 60
    bucket_count = max(1, int(math.ceil(duration_seconds / bucket_seconds)))
    timeline_data = [0 for _ in range(bucket_count)]
    for s in students:
        for ev in getattr(s, "events", []):
            et = ev.get("type")
            ts = float(ev.get("ts", 0) or 0)
            if ts <= 0:
                continue
            should_count = (et == "help") or (et == "pause_wait")
            if not should_count:
                continue
            idx = int(max(0, min(bucket_count - 1, (ts - start_ts) // bucket_seconds)))
            timeline_data[idx] += 1
    timeline_labels = [str(i * 4) for i in range(bucket_count)]

    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "what", "when", "where", "while", "into",
        "need", "help", "why", "how", "dont", "cant", "not", "does", "did", "are", "was", "were", "about",
        "task", "class", "material", "code", "line", "function", "python", "student",
    }
    topic_counts: dict[str, int] = {}
    for s in students:
        for msg in getattr(s, "help_requests", []):
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", (msg or "").lower()):
                if token in stopwords:
                    continue
                topic_counts[token] = topic_counts.get(token, 0) + 1
    if not topic_counts:
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", (session.task_description or "").lower()):
            if token in stopwords:
                continue
            topic_counts[token] = topic_counts.get(token, 0) + 1

    total_topic = max(1, sum(topic_counts.values()))
    hardest_topics = [
        {"name": k.replace("_", " ").title(), "pct": int(round((v / total_topic) * 100))}
        for k, v in sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ]
    if not hardest_topics:
        hardest_topics = [{"name": "Core Task Logic", "pct": 100}]

    students_table = [
        {
            "student_id": s.student_id,
            "name": s.name,
            "help_requests": len(s.help_requests),
            "hints": int(s.hints_given),
            "teacher_nudges": len(s.teacher_nudges),
            "quiz": (
                {
                    "correct": s.quiz_correct,
                    "total": s.quiz_total,
                    "score": s.quiz_score,
                }
                if s.quiz_score is not None else None
            ),
            "status": s.status,
            "idle_seconds": round(float(s.idle_seconds), 1),
        }
        for s in sorted(students, key=lambda s: (s.quiz_score is None, -(s.quiz_score or 0), s.name))
    ]
    quiz_rows = [
        {
            "student_id": student_id,
            "student_name": entry.get("student_name", student_id),
            "score": entry.get("score"),
            "correct": entry.get("correct"),
            "total": entry.get("total"),
        }
        for student_id, entry in quiz_results.items()
    ]

    return {
        "session_id": session.session_id,
        "task_description": session.task_description,
        "task_level": session.task_level,
        "created_at": start_ts,
        "ended_at": end_ts,
        "duration_minutes": int(round(duration_seconds / 60.0)),
        "summary": session.summary or "Session completed.",
        "analytics": analytics,
        "evidence_note": "No evidence yet" if analytics.get("quiz_avg_correct_pct") is None else None,
        "timeline": {
            "labels": timeline_labels,
            "data": timeline_data,
        },
        "hardest_topics": hardest_topics,
        "students": students_table,
        "quiz_results": quiz_rows,
    }


# ── REST API ───────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    """Public, non-secret runtime facts the consent notice needs to be truthful."""
    return {
        "ai_provider": ai_provider_name(),
        "session_retention_hours": session_manager.retention_hours(),
    }


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest):
    session, teacher_token = session_manager.create_session(req.task_description, req.task_level)
    return CreateSessionResponse(
        session_id=session.session_id,
        join_url=f"/student.html?session={session.session_id}",
        teacher_token=teacher_token,
    )


@app.post("/api/sessions/create-from-pdf")
async def create_session_from_pdf(
    file: UploadFile = File(...),
    task_level: str = Form("medium"),
    mode: str = Form("practical"),
    quiz_difficulty: str = Form("medium"),
):
    normalized_level = (task_level or "medium").lower()
    if normalized_level not in {"easy", "medium", "hard"}:
        normalized_level = "medium"
    normalized_mode = (mode or "practical").lower()
    if normalized_mode not in {"practical", "theoretical"}:
        normalized_mode = "practical"
    normalized_quiz_difficulty = (quiz_difficulty or normalized_level).lower()
    if normalized_quiz_difficulty not in {"easy", "medium", "hard"}:
        normalized_quiz_difficulty = normalized_level

    result = await _extract_uploaded_pdf(file)

    generated_task_description = await generate_task_description_from_pdf(
        pdf_text=result["text"],
        mode=normalized_mode,
        difficulty=normalized_quiz_difficulty,
    )

    session, teacher_token = session_manager.create_session(generated_task_description, normalized_level)
    session.launched = False  # the teacher reviews the generated task first
    session.quiz_mode_preference = normalized_mode
    session.quiz_difficulty_preference = normalized_quiz_difficulty
    session.pdf_text = result["text"]
    session.pdf_filename = file.filename
    session.pdf_analysis = await analyze_pdf_content(
        result["text"], generated_task_description, session_id=session.session_id,
    )
    session_manager.persist_session(session)

    return {
        "session_id": session.session_id,
        "join_url": f"/student.html?session={session.session_id}",
        "teacher_token": teacher_token,
        "task_description": session.task_description,
        "analysis": session.pdf_analysis,
        "filename": file.filename,
        "pages": result["pages"],
        "word_count": result["word_count"],
        "mode": session.quiz_mode_preference,
        "difficulty": session.quiz_difficulty_preference,
        "launched": session.launched,
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    """Public session state. A teacher presenting this session's token gets the full
    dashboard payload (with live code) so a reloaded dashboard can resume."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    token = bearer_token(request)
    if token is None:
        return session.to_dict()
    if not token_matches(token, session.teacher_token_hash):
        raise HTTPException(401, "Invalid teacher token")
    payload = session.to_dict(include_code=True)
    payload["role"] = "teacher"
    return payload


@app.post("/api/sessions/{session_id}/join")
async def join_session(session_id: str, req: JoinSessionRequest, request: Request):
    name = req.student_name.strip()
    if not name:
        raise HTTPException(400, "Student name is required")
    if req.consent is not True:
        raise HTTPException(400, "Consent to the data notice is required to join")
    pending = session_manager.get_session(session_id)
    if pending is not None and pending.active and not pending.launched:
        raise HTTPException(409, "Your teacher has not launched this session yet. Please wait and try again.")
    student, token = session_manager.join_session(session_id, name, bearer_token(request))
    if student is None:
        raise HTTPException(404, "Session not found or inactive")
    if token is None:
        raise HTTPException(409, "That name is already taken in this session. Pick another name.")
    if student.consented_at is None:
        student.consented_at = datetime.now().timestamp()
        session_manager.persist_session(session_manager.get_session(session_id))
    return {
        "status": "joined",
        "student_name": student.name,
        "student_id": student.student_id,
        "student_token": token,
    }


@app.patch("/api/sessions/{session_id}/task")
async def update_task(session_id: str, req: UpdateTaskRequest, request: Request):
    """Let the teacher review and edit the generated task before students see it."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    require_teacher(request, session)
    description = req.task_description.strip()
    if not description:
        raise HTTPException(400, "Task description cannot be empty")
    session.task_description = description
    session_manager.persist_session(session)
    await sio.emit("task_updated", {"task_description": description}, room=session_id)
    await broadcast_dashboard(session)
    return {"task_description": description}


@app.post("/api/sessions/{session_id}/launch")
async def launch_session(session_id: str, req: LaunchSessionRequest, request: Request):
    """Open the session to students once the teacher has reviewed (and maybe edited) the task."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    require_teacher(request, session)
    if not session.active:
        raise HTTPException(409, "Session has already ended")
    if req.task_description is not None:
        description = req.task_description.strip()
        if not description:
            raise HTTPException(400, "Task description cannot be empty")
        session.task_description = description
    session.launched = True
    session_manager.persist_session(session)
    await broadcast_dashboard(session)
    return {"launched": True, "task_description": session.task_description,
            "join_url": f"/student.html?session={session.session_id}"}


@app.post("/api/sessions/{session_id}/students/{student_id}/nudge")
async def nudge_student(session_id: str, student_id: str, req: NudgeRequest, request: Request):
    """Teacher sends a short message to one student. Delivered to that student's room only
    and logged as a teacher intervention."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    require_teacher(request, session)
    if not session.active:
        raise HTTPException(409, "Session has already ended")
    student = session.students.get(student_id)
    if student is None:
        raise HTTPException(404, "Student not found")
    message = " ".join(req.message.split())
    if not message:
        raise HTTPException(400, "Message cannot be empty")
    entry = student.log_teacher_nudge(message, datetime.now().timestamp())
    session_manager.persist_session(session)
    await sio.emit("teacher_message", {"message": message, "timestamp": entry["ts"]},
                   room=student_room(session_id, student_id))
    await broadcast_dashboard(session)
    return {"student_id": student_id, "message": message, "teacher_nudges": len(student.teacher_nudges)}


@app.post("/api/sessions/{session_id}/end", response_model=EndSessionResponse)
async def end_session(session_id: str, request: Request):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    require_teacher(request, session)
    session_manager.end_session(session_id)
    analytics = _build_session_analytics(session)
    summary = "Session ended. Review class metrics and chart for collective performance insights."
    session.summary = summary
    session.analytics = analytics
    session_manager.persist_session(session)
    # Students only learn that the session is over; class analytics stay with the teacher.
    await sio.emit("session_ended", {}, room=session_id)
    await sio.emit("session_ended", {"summary": summary, "analytics": analytics},
                   room=teacher_room(session_id))
    return EndSessionResponse(summary=summary, analytics=analytics)


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    """Erase a session and all of its student data (teacher only)."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    require_teacher(request, session)
    await sio.emit("session_ended", {}, room=session_id)
    await sio.emit("session_ended", {"summary": "Session data deleted by the teacher.", "analytics": {}},
                   room=teacher_room(session_id))
    session_manager.delete_session(session_id)
    return Response(status_code=204)


@app.get("/api/sessions/{session_id}/report")
async def session_report(session_id: str, request: Request):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    require_teacher(request, session)
    if not getattr(session, "analytics", None):
        session.analytics = _build_session_analytics(session)
        if not session.active:
            session_manager.persist_session(session)
    return _build_report_payload(session)


@app.get("/api/sessions")
async def list_sessions(request: Request):
    """List the sessions owned by the presented teacher token."""
    token = bearer_token(request)
    if token is None:
        raise HTTPException(401, "Teacher token required")
    owned = session_manager.list_sessions(token)
    if not owned:
        raise HTTPException(403, "Invalid teacher token")
    return owned


# ── PDF Upload & Analysis ─────────────────────────────────────────

@app.post("/api/sessions/{session_id}/upload-pdf")
async def upload_pdf(session_id: str, request: Request, file: UploadFile = File(...)):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    require_teacher(request, session)

    result = await _extract_uploaded_pdf(file)

    # Store extracted text in session
    session.pdf_text = result["text"]
    session.pdf_filename = file.filename

    # Generate AI analysis
    analysis = await analyze_pdf_content(result["text"], session.task_description, session_id=session_id)
    session.pdf_analysis = analysis
    session_manager.persist_session(session)

    return {
        "filename": file.filename,
        "pages": result["pages"],
        "word_count": result["word_count"],
        "analysis": analysis,
    }


# ── Quiz Generation & Submission ──────────────────────────────────

def _normalize_quiz_results(session) -> dict:
    """Make ``session.quiz_results`` a dict keyed by student_id.

    Sessions persisted before results were id-keyed may hold entries keyed by display
    name (and possibly no ``quiz_results`` at all). Those are re-keyed to the matching
    student's id when the name is still unique; anything unmatched is kept as-is under
    its original key so old data is never dropped or crashes the request.
    """
    results = getattr(session, "quiz_results", None)
    if not isinstance(results, dict):
        results = {}
    normalized: dict[str, dict] = {}
    for key, entry in results.items():
        entry = dict(entry) if isinstance(entry, dict) else {"score": entry}
        student = session.students.get(key)
        if student is None:
            same_name = [s for s in session.students.values() if s.name == key]
            if len(same_name) == 1 and same_name[0].student_id not in results:
                student = same_name[0]
        if student is None:
            entry.setdefault("student_name", str(key))
            normalized[str(key)] = entry
            continue
        entry.setdefault("student_name", student.name)
        normalized[student.student_id] = entry
        if student.quiz_score is None and isinstance(entry.get("score"), (int, float)):
            student.quiz_score = float(entry["score"])
            student.quiz_correct = int(entry.get("correct", 0) or 0)
            student.quiz_total = int(entry.get("total", 0) or 0)
    session.quiz_results = normalized
    return normalized


@app.post("/api/sessions/{session_id}/generate-quiz")
async def api_generate_quiz(
    session_id: str,
    request: Request,
    num_questions: int = 5,
    difficulty: Optional[str] = None,
    mode: Optional[str] = None,  # practical or theoretical
):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    require_teacher(request, session)

    pdf_text = getattr(session, "pdf_text", "") or ""
    pdf_analysis = getattr(session, "pdf_analysis", "") or ""
    material_context = ""
    if pdf_analysis:
        material_context += f"Material summary:\n{pdf_analysis[:2000]}\n\n"
    if pdf_text:
        material_context += f"Material raw text:\n{pdf_text[:4000]}"
    if not material_context.strip():
        raise HTTPException(400, "Upload class PDF material before generating quiz questions.")
    selected_difficulty = (difficulty or getattr(session, "quiz_difficulty_preference", "medium")).lower()
    if selected_difficulty not in {"easy", "medium", "hard"}:
        selected_difficulty = "medium"
    selected_mode = (mode or getattr(session, "quiz_mode_preference", "practical")).lower()
    if selected_mode not in {"practical", "theoretical"}:
        selected_mode = "practical"

    questions = await generate_quiz(
        task_description=session.task_description,
        pdf_text=material_context,
        num_questions=num_questions,
        difficulty=selected_difficulty,
        mode=selected_mode,
        session_id=session_id,
    )
    if not questions:
        raise HTTPException(
            502,
            "Quiz generation failed validation. Retry and ensure your model deployment is available.",
        )

    # Store quiz in session
    session.quiz = questions
    session.quiz_results = {}
    session_manager.persist_session(session)

    # Broadcast quiz to all students via WebSocket
    await sio.emit("quiz_available", {
        "questions": [
            {
                "question": q["question"],
                "options": q["options"],
                "task_description": q.get("task_description", ""),
            }
            for q in questions  # Don't send correct answers to students!
        ],
    }, room=session_id)

    return {"questions": questions, "count": len(questions)}


@app.post("/api/sessions/{session_id}/submit-quiz")
async def submit_quiz(session_id: str, submission: dict, request: Request):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    student_id = str(submission.get("student_id", ""))
    require_student(request, session, student_id)
    answers = submission.get("answers", {})
    quiz = getattr(session, "quiz", None)

    if not quiz:
        raise HTTPException(400, "No quiz available")
    _normalize_quiz_results(session)
    if student_id in session.quiz_results:
        raise HTTPException(409, "You have already submitted this quiz.")

    # Grade the quiz
    correct = 0
    total = len(quiz)
    results = []
    for i, q in enumerate(quiz):
        student_answer = answers.get(str(i), "")
        is_correct = student_answer == q["correct"]
        if is_correct:
            correct += 1
        results.append({
            "question": q["question"],
            "student_answer": student_answer,
            "correct_answer": q["correct"],
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
        })

    score = round((correct / total) * 100) if total > 0 else 0

    # Store results, keyed by the stable student_id; the name is only a label.
    student = session.students[student_id]
    session.quiz_results[student_id] = {
        "student_name": student.name,
        "score": score,
        "correct": correct,
        "total": total,
        "results": results,
        "submitted_at": datetime.now().timestamp(),
    }
    student.quiz_score = float(score)
    student.quiz_correct = correct
    student.quiz_total = total
    session_manager.persist_session(session)

    # Update teacher dashboard
    await sio.emit("quiz_result", {
        "student_id": student_id,
        "student_name": student.name,
        "score": score,
        "correct": correct,
        "total": total,
    }, room=teacher_room(session_id))
    await broadcast_dashboard(session)

    return {
        "score": score,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ── Static file serving ───────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{filename}.html")
async def serve_html(filename: str):
    filepath = FRONTEND_DIR / f"{filename}.html"
    if filepath.exists():
        return FileResponse(filepath)
    raise HTTPException(404, "Page not found")


# ── Socket.IO Events ──────────────────────────────────────────────

@sio.event
async def connect(sid, environ):
    print(f"[WS] Client connected: {sid}")


@sio.event
async def disconnect(sid):
    _student_sockets.pop(sid, None)
    print(f"[WS] Client disconnected: {sid}")


@sio.event
async def join_room(sid, data):
    """Student or teacher joins a session room."""
    session_id = data.get("session_id")
    role = data.get("role", "student")
    student_id = data.get("student_id", "")

    session = session_manager.get_session(session_id)
    if session is None:
        await sio.emit("error", {"message": "Session not found"}, to=sid)
        return

    if role == "teacher":
        if not token_matches(data.get("teacher_token"), session.teacher_token_hash):
            await sio.emit("error", {"message": "Invalid teacher token"}, to=sid)
            return
        await sio.enter_room(sid, teacher_room(session_id))
        print(f"[WS] Teacher joined session {session_id}")
        await sio.emit("dashboard_update", session.to_dict(include_code=True), to=sid)
        return

    if not session.launched:
        await sio.emit("error", {"message": "Your teacher has not launched this session yet."}, to=sid)
        return
    student = session_manager.authenticate_student(session_id, student_id, data.get("student_token"))
    if student is None:
        await sio.emit("error", {"message": "Invalid student token"}, to=sid)
        return

    _student_sockets[sid] = (session_id, student.student_id)
    student.sid = sid
    await sio.enter_room(sid, session_id)
    await sio.enter_room(sid, student_room(session_id, student.student_id))
    print(f"[WS] Student '{student.name}' joined session {session_id}")
    # If quiz already exists, send it to this newly-joined student
    if session.quiz:
        await sio.emit("quiz_available", {
            "questions": [
                {
                    "question": q["question"],
                    "options": q["options"],
                    "task_description": q.get("task_description", ""),
                }
                for q in session.quiz
            ],
        }, to=sid)
    # Notify teacher dashboard
    await broadcast_dashboard(session)


@sio.event
async def telemetry(sid, data):
    """Receive telemetry event from student."""
    session_id = data.get("session_id")
    student_id = data.get("student_id")
    event_data = data.get("event", {})

    # The socket must have authenticated via join_room, and may only report as itself.
    bound = _student_sockets.get(sid)
    if bound is None or bound != (session_id, student_id):
        await sio.emit("error", {"message": "Unauthorized telemetry"}, to=sid)
        return
    student = session_manager.authenticate_student(session_id, student_id, data.get("student_token"))
    if student is None:
        await sio.emit("error", {"message": "Invalid student token"}, to=sid)
        return

    session = session_manager.get_session(session_id)
    if session is None or not session.active:
        return

    event = TelemetryEvent(
        event_type=event_data.get("event_type", "unknown"),
        payload=event_data.get("payload", {}),
    )

    actions = process_telemetry(session, student_id, event)

    # Push dashboard update to all in the room
    if actions.get("dashboard_update"):
        await broadcast_dashboard(session)

    # Generate and send hint if needed
    if actions.get("should_hint"):
        student = session.students.get(student_id)
        hint_reason = actions.get("hint_reason", "idle")
        gate = gate_hint(student, hint_reason) if student else None
        if gate is not None and not gate.allowed:
            print(f"[Hints] {hint_reason} for '{student.name}' withheld: {gate.reason}")
            if gate.message:
                # Explicit request: answer without spending an LLM call so the student is not left waiting.
                await sio.emit("hint", {
                    "student_id": student_id,
                    "hint": gate.message,
                    "level": student.hint_level,
                    "withheld": gate.reason,
                }, to=student.sid or sid)
        elif student:
            if not student.sid:
                student.sid = sid

            material_context = ""
            if session.pdf_analysis:
                material_context += f"Material summary:\n{session.pdf_analysis[:1500]}\n\n"
            if session.pdf_text:
                material_context += f"Material raw text:\n{session.pdf_text[:2000]}"

            hint_text = await generate_hint(
                student=student,
                task_description=session.task_description,
                hint_reason=hint_reason,
                help_message=actions.get("help_message", ""),
                class_material=material_context,
                force_level=actions.get("force_hint_level"),
                session_id=session_id,
            )
            await sio.emit("hint", {
                "student_id": student_id,
                "hint": hint_text,
                "level": student.hint_level,
            }, room=student_room(session_id, student.student_id))
            # Also notify teacher
            await sio.emit("hint_given", {
                "student_id": student_id,
                "student_name": student.name,
                "hint": hint_text,
                "level": student.hint_level,
            }, room=teacher_room(session_id))
            refresh_status(student)
            session_manager.persist_session(session)
            await broadcast_dashboard(session)

    # Large paste: a neutral observation for the teacher only, never sent to students.
    if actions.get("large_paste_alert"):
        alert = actions["large_paste_alert"]
        alert["type"] = "large_paste"
        alert.setdefault("timestamp", datetime.now().timestamp())
        session.alerts.append(alert)
        await sio.emit("alert", alert, room=teacher_room(session_id))

    # Confusion spike alert
    if actions.get("confusion_spike"):
        spike = actions["confusion_spike"]
        # Avoid duplicate alerts within CONFUSION_SPIKE_DEDUP_SECONDS of the last one.
        if not is_duplicate_confusion_spike(session, now=spike["timestamp"]):
            session.alerts.append(spike)
            await sio.emit("alert", spike, room=teacher_room(session_id))


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 EduPulse server starting on http://localhost:{port}\n")
    uvicorn.run(socket_app, host="0.0.0.0", port=port, log_level="info")
