"""AI hint generation and post-class summary via Azure OpenAI (with mock fallback)."""

from __future__ import annotations
import os
import json
import re
import time
from datetime import datetime
from models import StudentState, SessionState

# ── OpenAI client (lazy init) ─────────────────────────────────────
_client = None
_temperature_supported = True
_json_schema_supported = True

# ── Spend guardrails ──────────────────────────────────────────────
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BREAKER_FAILURES = 3
DEFAULT_BREAKER_COOLDOWN_SECONDS = 120.0

_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
_budget_exhausted = False
# Tokens spent per session_id, so one runaway session cannot drain the process budget.
_session_usage: dict[str, int] = {}

# Circuit breaker: consecutive provider failures open it for a cool-off window,
# after which calls are attempted again (instead of a permanent "AI disabled" latch).
_breaker_failures = 0
_breaker_open_until = 0.0


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _token_budget() -> int:
    """Total tokens this process may spend. 0 (the default) means unlimited."""
    return _int_env("AI_TOKEN_BUDGET", 0)


def _session_token_budget() -> int:
    """Tokens a single session may spend. 0 (the default) means unlimited."""
    return _int_env("AI_SESSION_TOKEN_BUDGET", 0)


def session_budget_exhausted(session_id: str | None) -> bool:
    budget = _session_token_budget()
    if not budget or not session_id:
        return False
    return _session_usage.get(session_id, 0) >= budget


def get_usage(session_id: str | None = None) -> dict:
    """Token usage accumulated since process start (plus one session's usage if given)."""
    total = _usage["prompt_tokens"] + _usage["completion_tokens"]
    usage = {
        **_usage,
        "total_tokens": total,
        "budget": _token_budget(),
        "budget_exhausted": _budget_exhausted,
        "breaker_open": breaker_is_open(),
    }
    if session_id is not None:
        usage["session_tokens"] = _session_usage.get(session_id, 0)
        usage["session_budget"] = _session_token_budget()
        usage["session_budget_exhausted"] = session_budget_exhausted(session_id)
    return usage


def _record_usage(response, operation: str, session_id: str | None = None):
    global _budget_exhausted
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    _usage["calls"] += 1
    _usage["prompt_tokens"] += prompt_tokens
    _usage["completion_tokens"] += completion_tokens
    total = _usage["prompt_tokens"] + _usage["completion_tokens"]
    print(
        f"[AI Engine] {operation}: {prompt_tokens} in / {completion_tokens} out | process total {total} tokens"
    )
    budget = _token_budget()
    if budget and total >= budget and not _budget_exhausted:
        _budget_exhausted = True
        print(f"[AI Engine] Token budget of {budget} reached. Falling back to mock output until restart.")
    if session_id:
        _session_usage[session_id] = _session_usage.get(session_id, 0) + prompt_tokens + completion_tokens
        session_budget = _session_token_budget()
        if session_budget and _session_usage[session_id] >= session_budget:
            print(f"[AI Engine] Session {session_id} reached its token budget of {session_budget}. Using mock output for it.")


def breaker_is_open(now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return now < _breaker_open_until


def _open_breaker(reason: str, now: float | None = None):
    global _breaker_open_until
    now = time.time() if now is None else now
    cooldown = _float_env("AI_BREAKER_COOLDOWN_SECONDS", DEFAULT_BREAKER_COOLDOWN_SECONDS)
    _breaker_open_until = now + cooldown
    print(f"[AI Engine] Circuit breaker open for {cooldown:.0f}s ({reason}). Using mock output meanwhile.")


def _record_success():
    global _breaker_failures
    _breaker_failures = 0


def _handle_ai_exception(error: Exception, operation: str):
    global _breaker_failures
    message = str(error)
    print(f"[AI Engine] {operation} error: {error}")
    if "DeploymentNotFound" in message:
        # Configuration problem: every call will fail, so trip the breaker right away.
        _breaker_failures = 0
        _open_breaker("Azure deployment not found; set OPENAI_MODEL to a valid deployment name")
        return
    _breaker_failures += 1
    if _breaker_failures >= max(1, _int_env("AI_BREAKER_FAILURES", DEFAULT_BREAKER_FAILURES)):
        _breaker_failures = 0
        _open_breaker(f"{_int_env('AI_BREAKER_FAILURES', DEFAULT_BREAKER_FAILURES)} consecutive provider errors")


def _get_client(session_id: str | None = None):
    """The provider client, or None when callers must use mock output
    (no key, budget exhausted, session budget exhausted, or breaker open)."""
    global _client
    if _budget_exhausted or breaker_is_open() or session_budget_exhausted(session_id):
        return None
    if _client is not None:
        return _client
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SECONDS
    try:
        max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    except ValueError:
        max_retries = DEFAULT_MAX_RETRIES

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    if endpoint:
        from openai import AsyncAzureOpenAI
        _client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=endpoint,
            timeout=timeout,
            max_retries=max_retries,
        )
    else:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            timeout=timeout,
            max_retries=max_retries,
        )
    return _client


def _get_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")


async def _chat(
    client,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    json_schema: dict | None = None,
    operation: str = "chat",
    session_id: str | None = None,
):
    """Chat completion that degrades gracefully on models rejecting custom temperature
    or structured outputs, and records token usage against the process and session budgets.

    The request timeout and retry count come from the client (OPENAI_TIMEOUT_SECONDS /
    OPENAI_MAX_RETRIES), so they apply to every provider call made here."""
    global _temperature_supported, _json_schema_supported
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {"max_completion_tokens": max_tokens}
    if _temperature_supported:
        kwargs["temperature"] = temperature
    if json_schema is not None and _json_schema_supported:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "structured_output", "strict": True, "schema": json_schema},
        }

    while True:
        try:
            response = await client.chat.completions.create(model=_get_model(), messages=messages, **kwargs)
            _record_usage(response, operation, session_id)
            _record_success()
            return response
        except Exception as e:
            message = str(e)
            if _temperature_supported and "temperature" in message:
                _temperature_supported = False
                kwargs.pop("temperature", None)
                continue
            if "response_format" in kwargs and ("response_format" in message or "json_schema" in message):
                _json_schema_supported = False
                kwargs.pop("response_format")
                continue
            raise


def is_ai_available(session_id: str | None = None) -> bool:
    """Return True when a usable AI client is configured (and not budget/breaker blocked)."""
    return _get_client(session_id) is not None


def _extract_material_anchors(class_material: str, limit: int = 4) -> list[str]:
    """Extract short material terms that hints should mention explicitly."""
    text = re.sub(r"\s+", " ", class_material or "").strip()
    if not text:
        return []

    candidates = []
    # Prefer explicit concept bullets from summary text if present.
    for match in re.findall(r"(?:Key Concepts|Material summary|Task:|Input:|Output:|Edge Case:)\s*[:\-]?\s*([^.\n]{3,80})", text, flags=re.IGNORECASE):
        cleaned = re.sub(r"[^A-Za-z0-9_+\- ]", "", match).strip()
        if len(cleaned) >= 3:
            candidates.append(cleaned)

    # Add domain-ish tokens/phrases.
    for match in re.findall(r"\b(?:loop|modulo|counter|odd|even|array|list|function|return|input|edge case|condition|iteration)\b", text, flags=re.IGNORECASE):
        candidates.append(match.lower())

    # Unique in order, prefer shorter anchor terms.
    seen = set()
    anchors = []
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        anchors.append(c)
        if len(anchors) >= limit:
            break
    return anchors


def _ensure_anchor_in_hint(hint: str, anchors: list[str]) -> str:
    if not hint:
        return hint
    if not anchors:
        return hint
    lower = hint.lower()
    for a in anchors:
        if a.lower() in lower:
            return hint
    return f"{hint} Focus on '{anchors[0]}' from the class material."


def _material_required_hint(student_name: str) -> str:
    return (
        f"{student_name}, I can only give a precise hint from the uploaded class material. "
        "Open the relevant section and align your next step to that example or definition."
    )


# ── Hint gating (explicit-first) ──────────────────────────────────
# Automatic triggers (idle / pause) may deliver at most one Level-1 nudge; Levels 2 and 3
# need an explicit help request or evidence the student changed their code since the last
# hint. Every student also has a cooldown between hints and a hard cap after Level 3.

DEFAULT_HINT_COOLDOWN_SECONDS = 60.0
MAX_HINT_LEVEL = 3
MAX_HINTS_PER_STUDENT = 5
EXPLICIT_HINT_REASON = "help_request"

# Per-student delivery record: {"last_hint_at": float, "code_at_last_hint": str}
_hint_state: dict[str, dict] = {}


def _hint_cooldown_seconds() -> float:
    return _float_env("HINT_COOLDOWN_SECONDS", DEFAULT_HINT_COOLDOWN_SECONDS)


def _remember_hint_delivery(student: StudentState, now: float | None = None) -> None:
    _hint_state[student.student_id] = {
        "last_hint_at": time.time() if now is None else now,
        "code_at_last_hint": student.current_code,
    }


def code_at_last_hint(student: StudentState) -> str:
    return _hint_state.get(student.student_id, {}).get("code_at_last_hint", "")


def code_changed_since_last_hint(student: StudentState) -> bool:
    """True when the student's code differs from what it was when their last hint was delivered.
    Unknown (no delivery recorded in this process) counts as unchanged."""
    state = _hint_state.get(student.student_id)
    if state is None:
        return False
    return student.current_code.strip() != (state["code_at_last_hint"] or "").strip()


class HintGate:
    """Outcome of ``gate_hint``: whether to call the hint generator, why not, and what
    (if anything) to tell the student instead."""

    def __init__(self, allowed: bool, reason: str = "ok", message: str = ""):
        self.allowed = allowed
        self.reason = reason
        self.message = message


def gate_hint(student: StudentState, hint_reason: str, now: float | None = None) -> HintGate:
    now = time.time() if now is None else now
    explicit = hint_reason == EXPLICIT_HINT_REASON
    state = _hint_state.get(student.student_id)

    if student.hints_given >= MAX_HINTS_PER_STUDENT:
        message = (
            f"{student.name}, please ask your teacher for help with the next step."
        ) if explicit else ""
        return HintGate(False, "hint_cap", message)

    if student.hint_level >= MAX_HINT_LEVEL:
        message = (
            f"{student.name}, you've already received all {MAX_HINT_LEVEL} hint levels for this task. "
            "Try applying them to your code — your teacher can see you asked for help."
        ) if explicit else ""
        return HintGate(False, "level_cap", message)

    if state is not None:
        elapsed = now - state["last_hint_at"]
        cooldown = _hint_cooldown_seconds()
        if elapsed < cooldown:
            wait = max(1, int(cooldown - elapsed + 0.999))
            message = (
                f"{student.name}, give the last hint a try first — the next hint unlocks in {wait}s."
            ) if explicit else ""
            return HintGate(False, "cooldown", message)

    if not explicit and student.hint_level >= 1 and not code_changed_since_last_hint(student):
        return HintGate(False, "needs_explicit_request")

    return HintGate(True)


# ── Hint Generation ───────────────────────────────────────────────

HINT_SYSTEM_PROMPT = """You are EduPulse, an empathetic AI teaching assistant embedded in a live coding lab.
Your role is to help a struggling student WITHOUT giving them the answer.

Rules:
- Be warm, encouraging and empathetic. The student is frustrated.
- Adjust tone based on frustration level (0-1 scale). Higher = more empathetic.
- NEVER give complete code solutions.
- Prioritize the provided class material context first when giving guidance.
- Do not invent requirements or examples that are not supported by class material.
- Reference the student's current code/input when pointing them to the next step.
- If class material is available, avoid generic advice that ignores it.
- For Level 1: Give a conceptual hint only. Explain the underlying concept.
- For Level 2: Give a structural hint. Point to a specific area of their code.
- For Level 3: Give a partial solution — show the structure but leave key parts blank.
- Keep responses concise (2-4 sentences max).
- Use encouraging language and emoji sparingly."""


async def generate_hint(
    student: StudentState,
    task_description: str,
    hint_reason: str = "idle",
    help_message: str = "",
    class_material: str = "",
    force_level: int | None = None,
    session_id: str | None = None,
) -> str:
    """Generate a progressive hint for a struggling student."""
    # Determine hint level
    if force_level in (1, 2, 3):
        level = force_level
    else:
        level = min(3, student.hint_level + 1)
    student.hint_level = level
    student.hints_given += 1
    student.last_support_at = datetime.now().timestamp()
    _remember_hint_delivery(student)
    if not (class_material or "").strip():
        return _material_required_hint(student.name)
    anchors = _extract_material_anchors(class_material)

    client = _get_client(session_id)
    if client is None:
        return _mock_hint(
            student=student,
            level=level,
            reason=hint_reason,
            task_description=task_description,
            class_material=class_material,
            help_message=help_message,
            anchors=anchors,
        )

    frustration = student.frustration_score
    latest_line = ""
    if student.current_code.strip():
        latest_line = student.current_code.strip().splitlines()[-1][:200]
    user_prompt = f"""Student "{student.name}" needs help.
- Hint Level: {level}/3
- Frustration Score: {frustration:.2f}
- Trigger: {hint_reason}
- Idle time: {student.idle_seconds:.0f}s
- Help message from student: "{help_message}"
- Latest typed line: "{latest_line}"
- Student answer / current work ({student.current_code.count(chr(10)) + 1} lines):
```
{student.current_code[:1500]}
```

Task they are working on (secondary context only, class material has priority):
"{task_description[:500]}"

Class material context (if available):
"{class_material[:1200]}"

Material anchor terms (use at least one term exactly in your hint): {anchors}

Generate a Level {level} hint that is grounded in the class material and student's current input.
You MUST mention at least one material anchor term exactly.
Remember: be empathetic, concise, and do NOT give the answer."""

    try:
        response = await _chat(
            client, HINT_SYSTEM_PROMPT, user_prompt, max_tokens=250, temperature=0.7,
            operation="hint", session_id=session_id,
        )
        content = (response.choices[0].message.content or "").strip()
        return _ensure_anchor_in_hint(content, anchors)
    except Exception as e:
        _handle_ai_exception(e, "OpenAI")
        return _mock_hint(
            student=student,
            level=level,
            reason=hint_reason,
            task_description=task_description,
            class_material=class_material,
            help_message=help_message,
            anchors=anchors,
        )


def _mock_hint(
    student: StudentState,
    level: int,
    reason: str,
    task_description: str = "",
    class_material: str = "",
    help_message: str = "",
    anchors: list[str] | None = None,
) -> str:
    """Fallback hints when no API key is available."""
    name = student.name
    if not (class_material or "").strip():
        return _material_required_hint(name)
    material_note = "class material"
    anchors = anchors or _extract_material_anchors(class_material)
    anchor = anchors[0] if anchors else "the core concept"
    student_line = ""
    if student.current_code.strip():
        student_line = student.current_code.strip().splitlines()[-1][:120]
    help_note = help_message.strip()[:120]

    if level == 1:
        if student_line:
            return (
                f"Hey {name}, use '{anchor}' from the {material_note} to verify expected input/output before changing `{student_line}`. "
                f"Break the task into 2 small steps and test each one."
            )
        return (
            f"Hey {name}, start from '{anchor}' in the {material_note} and restate the exact requirement in one sentence. "
            f"Then write pseudocode for the first step only."
        )
    elif level == 2:
        if help_note:
            return (
                f"{name}, you asked: \"{help_note}\". Use the {material_note} example for '{anchor}', "
                f"then align your next function/loop block to that pattern without copying the full answer."
            )
        return (
            f"{name}, compare your current structure with '{anchor}' in the {material_note}: "
            "check loop bounds, base case, and return value in this order."
        )

    return (
        f"{name}, use this structure from '{anchor}' in the {material_note}: define input -> process each item -> handle edge case -> return result. "
        "Fill in the exact condition and update logic yourself."
    )


# ── Post-Class Summary ────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """You are EduPulse, generating a post-class analytics summary for a teacher.
Write a professional, actionable report. Include:
1. Overall class performance assessment
2. Which concepts caused the most confusion
3. Individual student highlights (both struggling and excelling)
4. Specific recommendations for the next class

Do not speculate about plagiarism or academic integrity: paste counts are neutral observations
that the teacher interprets themselves, so do not include a section about them.

Use clear sections with headers. Be specific and data-driven."""


async def generate_session_summary(session: SessionState) -> str:
    """Generate AI-powered post-class summary."""
    client = _get_client(session.session_id)

    # Build student data summary
    student_summaries = []
    for idx, (_, s) in enumerate(session.students.items(), start=1):
        student_summaries.append({
            "name": f"Student {idx}",
            "quiz_score": s.quiz_score,
            "support_signals": s.support_signals,
            "status": s.status,
            "keystrokes": s.total_keystrokes,
            "paste_events": len(s.paste_events),
            "large_pastes": sum(1 for p in s.paste_events if p["length"] >= 200),
            "hints_used": s.hints_given,
            "idle_time": round(s.idle_seconds, 0),
            "help_requests": s.help_requests,
            "frustration": round(s.frustration_score, 2),
            "code_lines": s.current_code.count("\n") + 1 if s.current_code else 0,
        })

    if client is None:
        return _mock_summary(session, student_summaries)

    user_prompt = f"""Generate a post-class analytics report.

Task: "{session.task_description[:1000]}"

Student Data:
{json.dumps(student_summaries, indent=2)}

Total students: {len(session.students)}
Session duration: active session

Generate a comprehensive but concise teaching report."""

    try:
        response = await _chat(
            client, SUMMARY_SYSTEM_PROMPT, user_prompt, max_tokens=800, temperature=0.5,
            operation="summary", session_id=session.session_id,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        _handle_ai_exception(e, "Summary generation")
        return _mock_summary(session, student_summaries)


def _mock_summary(session: SessionState, student_data: list[dict]) -> str:
    """Fallback summary when no API key is available."""
    total = len(student_data)
    if total == 0:
        return "📊 **Session Summary**\n\nNo students participated in this session."

    graded = [s for s in student_data if s["quiz_score"] is not None]
    avg_score = sum(s["quiz_score"] for s in graded) / len(graded) if graded else None
    struggling = [s for s in graded if s["quiz_score"] < 50]
    excelling = [s for s in graded if s["quiz_score"] >= 80]
    unsupported = [s for s in student_data if s["support_signals"] >= 3]
    accuracy_line = (
        f"{avg_score:.1f}% across {len(graded)}/{total} submissions"
        if avg_score is not None else "no quiz evidence yet"
    )

    report = f"""📊 **EduPulse Session Report**

## Overall Performance
- **Students:** {total}
- **Quiz Accuracy:** {accuracy_line}
- **Students Needing Support (3+ hints/help requests):** {len(unsupported)}

## Student Breakdown
"""
    if not graded:
        report += "\nNo quiz was submitted, so this session has no correctness evidence yet.\n"

    if excelling:
        report += f"\n### 🌟 Answered correctly ({len(excelling)} students)\n"
        for s in excelling:
            report += f"- **{s['name']}**: quiz {s['quiz_score']}%, {s['keystrokes']} keystrokes\n"

    if struggling:
        report += f"\n### ⚠️ Struggled on the quiz ({len(struggling)} students)\n"
        for s in struggling:
            report += f"- **{s['name']}**: quiz {s['quiz_score']}%, {s['hints_used']} hints used, {s['idle_time']}s idle\n"

    if unsupported:
        report += f"\n### 🙋 Asked for the most support ({len(unsupported)} students)\n"
        for s in unsupported:
            report += f"- **{s['name']}**: {s['support_signals']} support signals ({s['hints_used']} hints)\n"

    report += f"""
## Recommendations
- {"Run a quiz next session — there is no evidence of what the class understood." if avg_score is None else "Focus next class on reviewing the core concepts — quiz accuracy below 60%." if avg_score < 60 else "Class is progressing well. Consider introducing more advanced challenges."}
- {"Schedule one-on-one time with struggling students." if struggling else "No individual interventions needed."}
"""
    return report


# ── Quiz Generation ────────────────────────────────────────────────

QUIZ_SYSTEM_PROMPT = """You are EduPulse, generating quiz questions for a classroom lab session.
Create questions that test understanding of the concepts in the provided material.

Rules:
- Ground every question in the provided class material and avoid unrelated topics.
- Generate exactly the number of questions requested.
- Each question must be multiple choice with 4 options (A, B, C, D).
- There must be exactly one correct answer per question.
- Questions should range from easy to hard.
- Questions should test conceptual understanding, not just memorization.
- Return ONLY valid JSON, no markdown formatting.

Return format (JSON object):
{
  "questions": [
    {
      "question": "What does X do?",
      "options": {"A": "option1", "B": "option2", "C": "option3", "D": "option4"},
      "correct": "B",
      "explanation": "Brief explanation of why B is correct",
      "task_description": "For practical questions, a task description the student should solve"
    }
  ]
}"""


def _quiz_schema(mode: str) -> dict:
    """JSON schema enforcing the quiz shape for models supporting structured outputs."""
    question_props = {
        "question": {"type": "string"},
        "options": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in ("A", "B", "C", "D")},
            "required": ["A", "B", "C", "D"],
            "additionalProperties": False,
        },
        "correct": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "explanation": {"type": "string"},
        "task_description": {"type": "string", "minLength": 20},
    }
    required = ["question", "options", "correct", "explanation"]
    if mode == "practical":
        required.append("task_description")
    else:
        question_props.pop("task_description")
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": question_props,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }


async def generate_quiz(
    task_description: str,
    pdf_text: str = "",
    num_questions: int = 5,
    difficulty: str = "medium",
    mode: str = "practical",
    session_id: str | None = None,
) -> list[dict]:
    """Generate quiz questions from PDF/class material.

    `mode` may be "practical" (code/problems) or "theoretical" (conceptual).
    """
    client = _get_client(session_id)

    difficulty_guide = {
        "easy": "Ask basic recall and definition questions. Focus on fundamental concepts. Suitable for beginners.",
        "medium": "Ask questions that require understanding and application. Mix conceptual and practical questions.",
        "hard": "Ask questions that require analysis, edge-case reasoning, and deep understanding. Include tricky distractors.",
    }
    diff_instruction = difficulty_guide.get(difficulty, difficulty_guide["medium"])

    # Require class material. Do not fall back to generic task text.
    if not (pdf_text or "").strip():
        return []

    context = f"Lecture/Reference Material:\n{pdf_text[:4000]}"
    if task_description and task_description.strip():
        context += f"\n\nSession Task (secondary context only): {task_description[:500]}"
    
    # add mode description
    if mode == "theoretical":
        context += "\n\nPlease generate conceptual/theoretical questions that require written explanations rather than coding."
    
    mode_instruction = ""
    if mode == "practical":
        mode_instruction = (
            "\n\nFor PRACTICAL questions: every question object MUST contain a non-empty "
            "'task_description' field describing a coding or problem-solving task the student "
            "completes to answer the question. A question without it is invalid."
        )
    
    if client is None:
        return _mock_quiz(num_questions)

    user_prompt = f"""Generate {num_questions} multiple-choice quiz questions based on this material:

{context}

Difficulty: {difficulty.upper()}
{diff_instruction}{mode_instruction}

Return ONLY a valid JSON object of the form {{"questions": [...]}}. No markdown, no code blocks, just the JSON."""

    try:
        response = await _chat(
            client,
            QUIZ_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=1500,
            temperature=0.6,
            json_schema=_quiz_schema(mode),
            operation="quiz",
            session_id=session_id,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        questions = json.loads(raw)
        if isinstance(questions, dict):
            questions = questions.get("questions")
        # Validate structure
        if not isinstance(questions, list):
            return []
        if len(questions) != num_questions:
            return []
        for q in questions:
            if not isinstance(q, dict):
                return []
            if "question" not in q or "options" not in q or "correct" not in q:
                return []
            options = q.get("options", {})
            if not isinstance(options, dict):
                return []
            if set(options.keys()) != {"A", "B", "C", "D"}:
                return []
            if q.get("correct") not in {"A", "B", "C", "D"}:
                return []
            if mode == "practical" and not str(q.get("task_description", "")).strip():
                return []
        return questions
    except Exception as e:
        _handle_ai_exception(e, "Quiz generation")
        return _mock_quiz(num_questions)


def _mock_quiz(num_questions: int) -> list[dict]:
    """Fallback quiz when no API key is available."""
    mock_questions = [
        {
            "question": "What is the purpose of a loop in programming?",
            "options": {
                "A": "To define a variable",
                "B": "To repeat a block of code multiple times",
                "C": "To import a library",
                "D": "To print output"
            },
            "correct": "B",
            "explanation": "Loops allow you to execute a block of code repeatedly."
        },
        {
            "question": "What does a function return statement do?",
            "options": {
                "A": "It prints a value to the console",
                "B": "It stops the program entirely",
                "C": "It sends a value back to the caller",
                "D": "It creates a new variable"
            },
            "correct": "C",
            "explanation": "The return statement sends a value back to where the function was called."
        },
        {
            "question": "Which data structure uses key-value pairs?",
            "options": {
                "A": "List",
                "B": "Tuple",
                "C": "Set",
                "D": "Dictionary"
            },
            "correct": "D",
            "explanation": "Dictionaries store data as key-value pairs for fast lookup."
        },
        {
            "question": "What is an 'off-by-one' error?",
            "options": {
                "A": "Using the wrong variable name",
                "B": "A loop that runs one too many or one too few times",
                "C": "A syntax error in the code",
                "D": "Forgetting to import a module"
            },
            "correct": "B",
            "explanation": "Off-by-one errors occur when loop boundaries are incorrectly set."
        },
        {
            "question": "What is the time complexity of a linear search?",
            "options": {
                "A": "O(1)",
                "B": "O(log n)",
                "C": "O(n)",
                "D": "O(n²)"
            },
            "correct": "C",
            "explanation": "Linear search checks each element one by one, giving O(n) complexity."
        },
    ]
    return mock_questions[:num_questions]


# ── PDF Analysis ───────────────────────────────────────────────────

PDF_ANALYSIS_PROMPT = """You are EduPulse, analyzing lecture material for a teacher.
Provide a structured analysis including:
1. Key concepts covered (bulleted list)
2. Learning objectives students should achieve
3. Potential difficulty areas for students
4. Suggested focus areas for the lab session

Be concise and actionable. Use markdown formatting."""


async def analyze_pdf_content(pdf_text: str, task_description: str = "", session_id: str | None = None) -> str:
    """Analyze PDF content and generate teaching insights."""
    client = _get_client(session_id)
    if client is None:
        return _mock_pdf_analysis(pdf_text)

    context = f"Lecture Material:\n{pdf_text[:4000]}"
    if task_description:
        context += f"\n\nLab Task:\n{task_description[:500]}"

    try:
        response = await _chat(
            client,
            PDF_ANALYSIS_PROMPT,
            f"Analyze this material:\n\n{context}",
            max_tokens=600,
            temperature=0.5,
            operation="pdf-analysis",
            session_id=session_id,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        _handle_ai_exception(e, "PDF analysis")
        return _mock_pdf_analysis(pdf_text)


MIN_TASK_STEPS = 2
MAX_TASK_STEPS = 3


async def generate_task_description_from_pdf(
    pdf_text: str,
    mode: str = "practical",
    difficulty: str = "medium",
    session_id: str | None = None,
) -> str:
    """Generate a concise class task description from PDF material."""
    description, _ = await generate_task_from_pdf(pdf_text, mode, difficulty, session_id)
    return description


async def generate_task_from_pdf(
    pdf_text: str,
    mode: str = "practical",
    difficulty: str = "medium",
    session_id: str | None = None,
) -> tuple[str, list[str]]:
    """Generate a task description plus 2-3 numbered steps students tick off as they work."""
    client = _get_client(session_id)
    normalized_mode = "theoretical" if str(mode).lower() == "theoretical" else "practical"
    normalized_difficulty = str(difficulty).lower()
    if normalized_difficulty not in {"easy", "medium", "hard"}:
        normalized_difficulty = "medium"

    if client is None:
        description = _mock_task_description(pdf_text, normalized_mode, normalized_difficulty)
        return description, task_steps_from_description(description)

    prompt = f"""Create one concise lab task description from this class material.

Mode: {normalized_mode}
Difficulty: {normalized_difficulty}

Rules:
- Keep it specific to the material, not generic.
- For practical mode: output one coding task with expected behavior and constraints.
- For theoretical mode: output one concept-application task with short written reasoning expected.
- Use EXACTLY 4 short lines, plain text only, in this format:
  Task: ...
  Input: ...
  Output: ...
  Edge Case: ...
- Then add 2 or 3 short numbered steps a student completes in order to finish the task:
  Step 1: ...
  Step 2: ...
  Step 3: ...
- Each step must be a concrete, checkable action (not "think about" or "understand").
- Keep each line under 120 characters.
- Do not include markdown, bullets, emojis, or extra notes.

Material:
{pdf_text[:4500]}
"""

    try:
        response = await _chat(
            client,
            "You create high-quality classroom tasks grounded in the provided material.",
            prompt,
            max_tokens=380,
            temperature=0.4,
            operation="task-generation",
            session_id=session_id,
        )
        task = (response.choices[0].message.content or "").strip()
        if not task:
            description = _mock_task_description(pdf_text, normalized_mode, normalized_difficulty)
            return description, task_steps_from_description(description)
        description = _normalize_task_description(task, normalized_mode, pdf_text)
        steps = parse_task_steps(task)
        if len(steps) < MIN_TASK_STEPS:
            steps = task_steps_from_description(description)
        return description, steps
    except Exception as e:
        _handle_ai_exception(e, "Task description generation")
        description = _mock_task_description(pdf_text, normalized_mode, normalized_difficulty)
        return description, task_steps_from_description(description)


def parse_task_steps(raw: str) -> list[str]:
    """Pull ``Step N: ...`` lines out of model output, in order, capped at MAX_TASK_STEPS."""
    steps: list[str] = []
    for ln in (raw or "").splitlines():
        m = re.match(r"^\s*(?:step\s*)?(\d+)\s*[:.)-]\s*(.+)$", ln.strip(), flags=re.IGNORECASE)
        if not m:
            continue
        text = re.sub(r"\s+", " ", m.group(2)).strip()
        if text:
            steps.append(text[:130])
    return steps[:MAX_TASK_STEPS]


def task_steps_from_description(description: str) -> list[str]:
    """Deterministic steps derived from the Task / Input / Output / Edge Case lines.

    Used in mock mode and whenever the model does not return usable steps, so every
    session has a step list for students to tick off.
    """
    parts: dict[str, str] = {}
    for ln in (description or "").splitlines():
        if ":" not in ln:
            continue
        key, value = ln.split(":", 1)
        parts[key.strip().lower()] = value.strip()
    steps: list[str] = []
    if parts.get("input"):
        steps.append(f"Read the input: {parts['input']}")
    if parts.get("task"):
        steps.append(f"Implement the core logic: {parts['task']}")
    elif parts.get("output"):
        steps.append(f"Produce the output: {parts['output']}")
    if parts.get("edge case"):
        steps.append(f"Handle the edge case: {parts['edge case']}")
    if len(steps) < MIN_TASK_STEPS:
        steps = [
            "Write a first version that solves the main case.",
            "Test it on one example and fix anything that is wrong.",
        ]
    return [s[:130] for s in steps[:MAX_TASK_STEPS]]


def _mock_task_description(pdf_text: str, mode: str, difficulty: str) -> str:
    topic = _extract_topic(pdf_text)
    if mode == "theoretical":
        return "\n".join([
            f"Task: Explain the key concept from {topic} and apply it to one simple example.",
            "Input: A short explanation (3-5 lines) with one concrete case.",
            "Output: Clear reasoning that matches the concept from class material.",
            "Edge Case: Mention one common misconception and correct it.",
        ])
    return (
        "\n".join([
            f"Task: Build a small {difficulty} solution based on {topic}.",
            "Input: A list/array of integers read from user input or predefined test data.",
            "Output: Print/return the computed result in a clear format.",
            "Edge Case: Handle empty input (and negatives if relevant).",
        ])
    )


def _extract_topic(pdf_text: str) -> str:
    """Get a clean short topic phrase from PDF text."""
    text = " ".join((pdf_text or "").split())
    if not text:
        return "the uploaded class material"
    # Prefer the first strong phrase before punctuation.
    first = re.split(r"[.!?:;]", text, maxsplit=1)[0].strip()
    first = re.sub(r"\([^)]*\)", "", first)  # remove inline parenthetical clutter
    first = re.sub(r"\s+", " ", first).strip(" -_,")
    if not first:
        first = text[:80].strip()
    if len(first) > 80:
        first = first[:80].rstrip()
    return first


def _derive_task_goal_from_material(pdf_text: str, mode: str) -> str:
    """Derive a concrete task goal phrase from material text."""
    text = " ".join((pdf_text or "").lower().split())
    if not text:
        return "implement the required logic from class material"

    # Strong deterministic matches for common lab topics.
    if "count" in text and "odd" in text and ("list" in text or "array" in text):
        return "count how many odd numbers are in a list/array of integers"
    if "count" in text and "odd" in text:
        return "count odd numbers from the given integer input"
    if "sum" in text and "odd" in text:
        return "compute the sum of odd numbers from the given input"
    if "even" in text and "count" in text:
        return "count how many even numbers are in the given list/array"
    if "factorial" in text:
        return "compute factorial for a given non-negative integer"
    if "prime" in text and "check" in text:
        return "check whether a given number is prime"

    # Generic fallback from first meaningful chunk.
    first = _extract_topic(pdf_text).lower()
    if first:
        return f"solve a small coding task about {first}"
    return "implement the required logic from class material"


def _normalize_task_description(raw: str, mode: str, pdf_text: str = "") -> str:
    """Force a short, consistently structured 4-line task description."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    line_map = {}
    for ln in lines:
        lower = ln.lower()
        if lower.startswith("task:"):
            line_map["Task"] = ln.split(":", 1)[1].strip()
        elif lower.startswith("input:"):
            line_map["Input"] = ln.split(":", 1)[1].strip()
        elif lower.startswith("output:"):
            line_map["Output"] = ln.split(":", 1)[1].strip()
        elif lower.startswith("edge case:"):
            line_map["Edge Case"] = ln.split(":", 1)[1].strip()

    # If model didn't follow exact format, rebuild from best-effort content.
    if not line_map:
        joined = " ".join(lines)
        joined = re.sub(r"\s+", " ", joined).strip()
        if len(joined) > 240:
            joined = joined[:240].rstrip()
        if mode == "theoretical":
            return "\n".join([
                f"Task: {joined or 'Explain one core concept from the uploaded material.'}",
                "Input: A short written explanation and one example.",
                "Output: Clear, correct reasoning.",
                "Edge Case: Include one common mistake and fix it.",
            ])
        return "\n".join([
            f"Task: {joined or 'Implement the core logic from the uploaded material.'}",
            "Input: Problem input in the required format.",
            "Output: Correct computed result.",
            "Edge Case: Handle empty or minimal input safely.",
        ])

    goal = _derive_task_goal_from_material(pdf_text, mode)
    task_line = line_map.get("Task", "")
    task_lower = task_line.lower()
    # Replace vague task lines with concrete material-grounded goal.
    if (
        not task_line
        or "python basics" in task_lower
        or "uploaded material" in task_lower
        or "class material" in task_lower
        or len(task_line.split()) < 5
    ):
        task_line = f"Build a {mode} solution to {goal}."

    ordered = [
        f"Task: {task_line}",
        f"Input: {line_map.get('Input', 'Use valid input in the expected format.')}",
        f"Output: {line_map.get('Output', 'Produce the correct result clearly.')}",
        f"Edge Case: {line_map.get('Edge Case', 'Handle empty or boundary input.')}",
    ]
    # Keep each line reasonably short and clean.
    cleaned = []
    for ln in ordered:
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) > 130:
            ln = ln[:130].rstrip()
        cleaned.append(ln)
    return "\n".join(cleaned)


def _mock_pdf_analysis(pdf_text: str) -> str:
    word_count = len(pdf_text.split())
    return f"""## 📄 Document Analysis

**Words extracted:** {word_count}

### Key Concepts
- Core programming concepts identified in the material
- Data structures and algorithms mentioned
- Problem-solving patterns covered

### Learning Objectives
- Students should understand the fundamental concepts
- Apply knowledge to practical coding exercises
- Debug and test their solutions

### Potential Difficulty Areas
- Abstract concepts that require hands-on practice
- Edge cases students commonly miss
- Integration of multiple concepts

### Recommendations
- Start with simple examples before the full task
- Encourage students to write pseudocode first
- Use the hint system for guided learning
"""

