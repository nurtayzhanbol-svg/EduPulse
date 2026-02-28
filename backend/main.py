"""EduPulse — Main server: FastAPI + Socket.IO."""

from __future__ import annotations
import asyncio
import math
import os
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

import socketio
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models import (
    CreateSessionRequest, CreateSessionResponse,
    JoinSessionRequest, EndSessionResponse, TelemetryEvent,
)
import session_manager
from telemetry import process_telemetry
from ai_engine import (
    generate_hint,
    generate_quiz,
    analyze_pdf_content,
    generate_task_description_from_pdf,
    is_ai_available,
)
from pdf_engine import extract_text_from_pdf

# ── App setup ──────────────────────────────────────────────────────
app = FastAPI(title="EduPulse", version="1.0.0")

# Socket.IO server (ASGI mode)
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


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
            "avg_understanding_score": 0.0,
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
    critical_students = sum(1 for s in students if s.hints_given >= 3 or s.status == "red")
    confused_students = sum(
        1 for s in students
        if s.hints_given >= 1 or s.frustration_score >= 0.5
    )
    # End-of-session understanding is based on hints taken (teacher request).
    hint_based_scores = [max(0.0, 100.0 - (min(4, s.hints_given) * 25.0)) for s in students]
    avg_score = sum(hint_based_scores) / total_students
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

    bars = [
        {"label": "On-Track Students", "value": float(on_track_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Struggling (>=1 hint)", "value": float(struggling_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "High Struggle (>=2 hints)", "value": float(high_struggle_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Critical (>=3 hints)", "value": float(critical_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Confused Students", "value": float(confused_students), "max": float(total_students), "unit": f"/{total_students}"},
        {"label": "Avg Understanding (Hints-Based)", "value": round(avg_score, 1), "max": 100.0, "unit": "%"},
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
    if avg_score >= 70:
        insights.append("Class readiness looks good for a slightly harder follow-up task next week.")
    elif avg_score <= 40:
        insights.append("Class understanding is low. Start next class with a focused recap and worked example.")

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
        "avg_understanding_score": round(avg_score, 1),
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

    mastered = sum(1 for s in students if s.hints_given <= 1)
    partial = sum(1 for s in students if 2 <= s.hints_given <= 4)
    struggling = sum(1 for s in students if s.hints_given >= 5)
    incomplete = max(0, total_students - mastered - partial - struggling)

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
        }
        for s in sorted(students, key=lambda x: (x.hints_given, x.name), reverse=True)
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
            "mastered": mastered,
            "partial": partial,
            "struggling": struggling,
            "incomplete": incomplete,
        },
        "percentages": {
            "mastered": pct(mastered),
            "partial": pct(partial),
            "struggling": pct(struggling),
            "incomplete": pct(incomplete),
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
    session = session_manager.create_session(req.task_description, req.task_level)
    return CreateSessionResponse(
        session_id=session.session_id,
        join_url=f"/student.html?session={session.session_id}",
    )


@app.post("/api/sessions/create-from-pdf")
async def create_session_from_pdf(
    file: UploadFile = File(...),
    task_level: str = Form("medium"),
    mode: str = Form("practical"),
    quiz_difficulty: str = Form("medium"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    normalized_level = (task_level or "medium").lower()
    if normalized_level not in {"easy", "medium", "hard"}:
        normalized_level = "medium"
    normalized_mode = (mode or "practical").lower()
    if normalized_mode not in {"practical", "theoretical"}:
        normalized_mode = "practical"
    normalized_quiz_difficulty = (quiz_difficulty or normalized_level).lower()
    if normalized_quiz_difficulty not in {"easy", "medium", "hard"}:
        normalized_quiz_difficulty = normalized_level

    pdf_bytes = await file.read()
    result = extract_text_from_pdf(pdf_bytes)
    if not result["text"].strip() or result["word_count"] < 20:
        raise HTTPException(
            400,
            "Could not extract enough text from this PDF. Please upload a text-based PDF (not scanned images only).",
        )

    generated_task_description = await generate_task_description_from_pdf(
        pdf_text=result["text"],
        mode=normalized_mode,
        difficulty=normalized_quiz_difficulty,
    )

    session = session_manager.create_session(generated_task_description, normalized_level)
    session.quiz_mode_preference = normalized_mode
    session.quiz_difficulty_preference = normalized_quiz_difficulty
    session.pdf_text = result["text"]
    session.pdf_filename = file.filename
    session.pdf_analysis = await analyze_pdf_content(result["text"], generated_task_description)

    return {
        "session_id": session.session_id,
        "join_url": f"/student.html?session={session.session_id}",
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
async def join_session(session_id: str, req: JoinSessionRequest):
    student = session_manager.join_session(session_id, req.student_name)
    if student is None:
        raise HTTPException(404, "Session not found or inactive")
    return {"status": "joined", "student_name": student.name}


@app.post("/api/sessions/{session_id}/end", response_model=EndSessionResponse)
async def end_session(session_id: str):
    session = session_manager.end_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    analytics = _build_session_analytics(session)
    summary = "Session ended. Review class metrics and chart for collective performance insights."
    session.summary = summary
    session.analytics = analytics
    session.ended_at = datetime.now().timestamp()
    # Broadcast session ended
    await sio.emit("session_ended", {"summary": summary, "analytics": analytics}, room=session_id)
    return EndSessionResponse(summary=summary, analytics=analytics)


@app.get("/api/sessions/{session_id}/report")
async def session_report(session_id: str):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if not getattr(session, "analytics", None):
        session.analytics = _build_session_analytics(session)
    return _build_report_payload(session)


@app.get("/api/sessions")
async def list_sessions():
    return session_manager.list_sessions()


# ── PDF Upload & Analysis ─────────────────────────────────────────

@app.post("/api/sessions/{session_id}/upload-pdf")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    pdf_bytes = await file.read()
    result = extract_text_from_pdf(pdf_bytes)
    if not result["text"].strip() or result["word_count"] < 20:
        raise HTTPException(
            400,
            "Could not extract enough text from this PDF. Please upload a text-based PDF (not scanned images only).",
        )

    # Store extracted text in session
    session.pdf_text = result["text"]
    session.pdf_filename = file.filename

    # Generate AI analysis
    analysis = await analyze_pdf_content(result["text"], session.task_description)
    session.pdf_analysis = analysis

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
    num_questions: int = 5,
    difficulty: Optional[str] = None,
    mode: Optional[str] = None,  # practical or theoretical
):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

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
async def submit_quiz(session_id: str, submission: dict):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    student_name = submission.get("student_name", "Unknown")
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

    # Update teacher dashboard
    await sio.emit("quiz_result", {
        "student_name": student_name,
        "score": score,
        "correct": correct,
        "total": total,
    }, room=session_id)

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


# Mount static dirs
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


# ── Socket.IO Events ──────────────────────────────────────────────

@sio.event
async def connect(sid, environ):
    print(f"[WS] Client connected: {sid}")


@sio.event
async def disconnect(sid):
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

    await sio.enter_room(sid, session_id)

    if role == "student" and student_name:
        student = session_manager.join_session(session_id, student_name, sid=sid)
        print(f"[WS] Student '{student_name}' joined session {session_id}")
        # If quiz already exists, send it to this newly-joined student
        if hasattr(session, "quiz") and session.quiz:
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
        await sio.emit("dashboard_update", session.to_dict(), room=session_id)

    elif role == "teacher":
        print(f"[WS] Teacher joined session {session_id}")
        await sio.emit("dashboard_update", session.to_dict(), to=sid)


@sio.event
async def telemetry(sid, data):
    """Receive telemetry event from student."""
    session_id = data.get("session_id")
    student_name = data.get("student_name")
    event_data = data.get("event", {})

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
        await sio.emit("dashboard_update", session.to_dict(), room=session_id)

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

    # Plagiarism alert
    if actions.get("plagiarism_alert"):
        alert = actions["plagiarism_alert"]
        alert["type"] = "plagiarism"
        session.alerts.append(alert)
        await sio.emit("alert", alert, room=session_id)

    # Confusion spike alert
    if actions.get("confusion_spike"):
        spike = actions["confusion_spike"]
        # Avoid duplicate alerts within 30 seconds
        recent_spikes = [
            a for a in session.alerts
            if a.get("type") == "confusion_spike"
        ]
        if not recent_spikes or len(session.alerts) == 0:
            session.alerts.append(spike)
            await sio.emit("alert", spike, room=session_id)


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 EduPulse server starting on http://localhost:{port}\n")
    uvicorn.run(socket_app, host="0.0.0.0", port=port, log_level="info")
