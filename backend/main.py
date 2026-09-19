"""EduPulse — Main server: FastAPI + Socket.IO."""

from __future__ import annotations
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
from fastapi.responses import FileResponse, JSONResponse

from models import (
    CreateSessionRequest, CreateSessionResponse,
    JoinSessionRequest, EndSessionResponse, TelemetryEvent,
)
import session_manager
from auth import bearer_token, require_student, require_teacher, token_matches
from telemetry import is_duplicate_confusion_spike, process_telemetry
from ai_engine import (
    generate_hint,
    generate_quiz,
    analyze_pdf_content,
    generate_task_description_from_pdf,
)
from pdf_engine import extract_text_from_pdf, max_upload_bytes, save_upload_to_tempfile

# ── App setup ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    session_manager.cleanup_expired_sessions()
    yield


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

# Socket sid -> (session_id, student_name) for sockets that authenticated via join_room.
_student_sockets: dict[str, tuple[str, str]] = {}


def teacher_room(session_id: str) -> str:
    """Room holding only authenticated teacher sockets of a session."""
    return f"{session_id}:teachers"


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
    students = list(session.students.values())
    total_students = len(students)
    if total_students == 0:
        return {
            "total_students": 0,
            "total_hints": 0,
            "hints_per_student_per_task": 0.0,
            "struggling_students": 0,
            "high_struggle_students": 0,
            "confused_students": 0,
            "on_track_students": 0,
            "critical_students": 0,
            "struggling_ratio": 0.0,
            "total_help_requests": 0,
            "total_support_signals": 0,
            "avg_support_signals": 0.0,
            "quiz_submissions": 0,
            "quiz_accuracy": None,
            "avg_frustration": 0.0,
            "avg_idle_seconds": 0.0,
            "avg_time_in_session_seconds": 0.0,
            "avg_keystrokes": 0.0,
            "avg_code_lines": 0.0,
            "total_large_pastes": 0,
            "bars": [],
        }

    total_hints = sum(s.hints_given for s in students)
    struggling_students = sum(1 for s in students if s.hints_given >= 1)
    high_struggle_students = sum(1 for s in students if s.hints_given >= 2)
    on_track_students = sum(1 for s in students if s.hints_given == 0 and s.status == "green")
    # "Needs follow-up": either the student leaned on hints, or telemetry flagged them red.
    critical_students = sum(1 for s in students if s.hints_given >= 3 or s.status == "red")
    confused_students = sum(
        1 for s in students
        if s.hints_given >= 1 or s.frustration_score >= 0.5
    )
    total_help_requests = sum(len(s.help_requests) for s in students)
    total_support_signals = sum(s.support_signals for s in students)
    # Quiz correctness is the only direct evidence of understanding; absent a
    # submission there is no evidence, which is different from a low score.
    quiz_scores = [float(s.quiz_score) for s in students if s.quiz_score is not None]
    quiz_accuracy = round(sum(quiz_scores) / len(quiz_scores), 1) if quiz_scores else None
    avg_frustration = sum(s.frustration_score for s in students) / total_students
    avg_idle = sum(s.idle_seconds for s in students) / total_students
    avg_time_in_session = sum(max(0.0, s.last_activity - s.joined_at) for s in students) / total_students
    avg_keystrokes = sum(s.total_keystrokes for s in students) / total_students
    avg_code_lines = sum((s.current_code.count("\n") + 1) if s.current_code else 0 for s in students) / total_students
    total_large_pastes = sum(
        1 for s in students for p in s.paste_events if p.get("length", 0) >= 200
    )
    long_pause_students = sum(
        1 for s in students if s.idle_seconds >= getattr(session, "pause_threshold_seconds", 60)
    )

    hints_per_student_per_task = round(total_hints / total_students, 2)
    struggling_ratio = round((struggling_students / total_students) * 100, 1)
    avg_support_signals = round(total_support_signals / total_students, 2)

    bars = [
        {"label": "Quiz Accuracy", "value": quiz_accuracy if quiz_accuracy is not None else 0.0, "max": 100.0, "unit": "%" if quiz_accuracy is not None else " (no evidence yet)"},
        {"label": "Quiz Submissions", "value": float(len(quiz_scores)), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Explicit Help Requests", "value": float(total_help_requests), "max": max(1.0, float(total_help_requests)), "unit": ""},
        {"label": "Support Signals per Student", "value": avg_support_signals, "max": max(3.0, avg_support_signals + 1.0), "unit": ""},
        {"label": "On-Track Students", "value": float(on_track_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Struggling (>=1 hint)", "value": float(struggling_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "High Struggle (>=2 hints)", "value": float(high_struggle_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Needs Follow-up (>=3 hints or flagged)", "value": float(critical_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Confused Students", "value": float(confused_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Avg Frustration", "value": round(avg_frustration, 2), "max": 1.0, "unit": ""},
        {"label": "Hints per Student/Task", "value": float(hints_per_student_per_task), "max": max(3.0, hints_per_student_per_task + 1.0), "unit": ""},
        {"label": "Avg Idle", "value": round(avg_idle, 1), "max": max(120.0, avg_idle + 30.0), "unit": "s"},
        {"label": "Long Pause Students", "value": float(long_pause_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Avg Time in Session", "value": round(avg_time_in_session / 60.0, 1), "max": max(10.0, round(avg_time_in_session / 60.0, 1) + 2.0), "unit": "min"},
        {"label": "Avg Keystrokes", "value": round(avg_keystrokes, 1), "max": max(10.0, avg_keystrokes + 10.0), "unit": ""},
        {"label": "Avg Code Lines", "value": round(avg_code_lines, 1), "max": max(5.0, avg_code_lines + 5.0), "unit": ""},
    ]

    insights: list[str] = []
    if struggling_students > 0:
        insights.append(
            f"{struggling_students}/{total_students} students needed hints; prioritize review of core task logic next class."
        )
    else:
        insights.append("No students required hints in this session.")
    if high_struggle_students > 0:
        insights.append(
            f"{high_struggle_students} students used 2+ hints (high struggle). Plan a guided practice segment next session."
        )
    if avg_idle >= 90:
        insights.append("High idle time detected. Add shorter milestones/checkpoints during the task.")
    if long_pause_students > 0:
        insights.append(
            f"{long_pause_students} students reached the pause-hint threshold ({getattr(session, 'pause_threshold_seconds', 60)}s)."
        )
    if quiz_accuracy is None:
        insights.append(
            "No quiz evidence yet — run a quiz to measure what the class actually understood."
        )
    elif quiz_accuracy >= 70:
        insights.append(
            f"Quiz accuracy {quiz_accuracy}% across {len(quiz_scores)}/{total_students} submissions; "
            "class readiness looks good for a harder follow-up task."
        )
    elif quiz_accuracy <= 40:
        insights.append(
            f"Quiz accuracy {quiz_accuracy}% across {len(quiz_scores)}/{total_students} submissions. "
            "Start next class with a focused recap and worked example."
        )

    return {
        "total_students": total_students,
        "total_hints": total_hints,
        "hints_per_student_per_task": hints_per_student_per_task,
        "struggling_students": struggling_students,
        "high_struggle_students": high_struggle_students,
        "confused_students": confused_students,
        "on_track_students": on_track_students,
        "critical_students": critical_students,
        "struggling_ratio": struggling_ratio,
        "total_help_requests": total_help_requests,
        "total_support_signals": total_support_signals,
        "avg_support_signals": avg_support_signals,
        "quiz_submissions": len(quiz_scores),
        "quiz_accuracy": quiz_accuracy,
        "avg_frustration": round(avg_frustration, 2),
        "avg_idle_seconds": round(avg_idle, 1),
        "avg_time_in_session_seconds": round(avg_time_in_session, 1),
        "avg_keystrokes": round(avg_keystrokes, 1),
        "avg_code_lines": round(avg_code_lines, 1),
        "total_large_pastes": total_large_pastes,
        "long_pause_students": long_pause_students,
        "bars": bars,
        "insights": insights,
    }


def _build_report_payload(session) -> dict:
    analytics = getattr(session, "analytics", None) or _build_session_analytics(session)
    students = list(session.students.values())
    total_students = len(students)

    # Buckets are quiz correctness, not hint usage: a student who asked for help and
    # then answered correctly understood the material.
    graded = [s for s in students if s.quiz_score is not None]
    strong = sum(1 for s in graded if s.quiz_score >= 80)
    mixed = sum(1 for s in graded if 50 <= s.quiz_score < 80)
    weak = sum(1 for s in graded if s.quiz_score < 50)
    no_evidence = max(0, total_students - len(graded))

    def pct(v: int) -> int:
        return int(round((v / total_students) * 100)) if total_students else 0

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
            "name": s.name,
            "hints": int(s.hints_given),
            "status": s.status,
            "idle_seconds": round(float(s.idle_seconds), 1),
            "help_requests": len(s.help_requests),
            "support_signals": s.support_signals,
            "quiz_score": s.quiz_score,
        }
        for s in sorted(students, key=lambda x: (x.support_signals, x.name), reverse=True)
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
        "counts": {
            "strong": strong,
            "mixed": mixed,
            "weak": weak,
            "no_evidence": no_evidence,
        },
        "percentages": {
            "strong": pct(strong),
            "mixed": pct(mixed),
            "weak": pct(weak),
            "no_evidence": pct(no_evidence),
        },
        "timeline": {
            "labels": timeline_labels,
            "data": timeline_data,
        },
        "hardest_topics": hardest_topics,
        "students": students_table,
    }


# ── REST API ───────────────────────────────────────────────────────

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
    session.quiz_mode_preference = normalized_mode
    session.quiz_difficulty_preference = normalized_quiz_difficulty
    session.pdf_text = result["text"]
    session.pdf_filename = file.filename
    session.pdf_analysis = await analyze_pdf_content(result["text"], generated_task_description)
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
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session.to_dict()


@app.post("/api/sessions/{session_id}/join")
async def join_session(session_id: str, req: JoinSessionRequest, request: Request):
    name = req.student_name.strip()
    if not name:
        raise HTTPException(400, "Student name is required")
    student, token = session_manager.join_session(session_id, name, bearer_token(request))
    if student is None:
        raise HTTPException(404, "Session not found or inactive")
    if token is None:
        raise HTTPException(409, "That name is already taken in this session. Pick another name.")
    return {
        "status": "joined",
        "student_name": student.name,
        "student_id": student.student_id,
        "student_token": token,
    }


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
    # Broadcast session ended
    await sio.emit("session_ended", {"summary": summary, "analytics": analytics}, room=session_id)
    return EndSessionResponse(summary=summary, analytics=analytics)


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
    analysis = await analyze_pdf_content(result["text"], session.task_description)
    session.pdf_analysis = analysis
    session_manager.persist_session(session)

    return {
        "filename": file.filename,
        "pages": result["pages"],
        "word_count": result["word_count"],
        "analysis": analysis,
    }


# ── Quiz Generation & Submission ──────────────────────────────────

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

    student_name = str(submission.get("student_name", ""))
    require_student(request, session, student_name)
    answers = submission.get("answers", {})
    quiz = getattr(session, "quiz", None)

    if not quiz:
        raise HTTPException(400, "No quiz available")

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

    # Store results
    if not hasattr(session, "quiz_results"):
        session.quiz_results = {}
    session.quiz_results[student_name] = {
        "score": score,
        "correct": correct,
        "total": total,
        "results": results,
    }
    student = session.students.get(student_name)
    if student is not None:
        student.quiz_score = float(score)
        student.quiz_correct = correct
        student.quiz_total = total
    session_manager.persist_session(session)

    # Update teacher dashboard
    await sio.emit("quiz_result", {
        "student_name": student_name,
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
    student_name = data.get("student_name", "")

    session = session_manager.get_session(session_id)
    if session is None:
        await sio.emit("error", {"message": "Session not found"}, to=sid)
        return

    if role == "teacher":
        if not token_matches(data.get("teacher_token"), session.teacher_token_hash):
            await sio.emit("error", {"message": "Invalid teacher token"}, to=sid)
            return
        await sio.enter_room(sid, session_id)
        await sio.enter_room(sid, teacher_room(session_id))
        print(f"[WS] Teacher joined session {session_id}")
        await sio.emit("dashboard_update", session.to_dict(include_code=True), to=sid)
        return

    student = session_manager.authenticate_student(session_id, student_name, data.get("student_token"))
    if student is None:
        await sio.emit("error", {"message": "Invalid student token"}, to=sid)
        return

    _student_sockets[sid] = (session_id, student.name)
    student.sid = sid
    await sio.enter_room(sid, session_id)
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
    student_name = data.get("student_name")
    event_data = data.get("event", {})

    # The socket must have authenticated via join_room, and may only report as itself.
    bound = _student_sockets.get(sid)
    if bound is None or bound != (session_id, student_name):
        await sio.emit("error", {"message": "Unauthorized telemetry"}, to=sid)
        return
    student = session_manager.authenticate_student(session_id, student_name, data.get("student_token"))
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

    actions = process_telemetry(session, student_name, event)

    # Push dashboard update to all in the room
    if actions.get("dashboard_update"):
        await broadcast_dashboard(session)

    # Generate and send hint if needed
    if actions.get("should_hint"):
        student = session.students.get(student_name)
        if student:
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
                hint_reason=actions.get("hint_reason", "idle"),
                help_message=actions.get("help_message", ""),
                class_material=material_context,
                force_level=actions.get("force_hint_level"),
            )
            target_sid = student.sid or sid
            await sio.emit("hint", {
                "student_name": student_name,
                "hint": hint_text,
                "level": student.hint_level,
            }, to=target_sid)
            # Also notify teacher
            await sio.emit("hint_given", {
                "student_name": student_name,
                "hint": hint_text,
                "level": student.hint_level,
            }, room=session_id)
            session_manager.persist_session(session)

    # Plagiarism alert
    if actions.get("plagiarism_alert"):
        alert = actions["plagiarism_alert"]
        alert["type"] = "plagiarism"
        alert.setdefault("timestamp", datetime.now().timestamp())
        session.alerts.append(alert)
        await sio.emit("alert", alert, room=session_id)

    # Confusion spike alert
    if actions.get("confusion_spike"):
        spike = actions["confusion_spike"]
        # Avoid duplicate alerts within CONFUSION_SPIKE_DEDUP_SECONDS of the last one.
        if not is_duplicate_confusion_spike(session, now=spike["timestamp"]):
            session.alerts.append(spike)
            await sio.emit("alert", spike, room=session_id)


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 EduPulse server starting on http://localhost:{port}\n")
    uvicorn.run(socket_app, host="0.0.0.0", port=port, log_level="info")
