"""AI hint generation and post-class summary via Azure OpenAI (with mock fallback)."""

from __future__ import annotations
import os
import json
import re
from models import StudentState, SessionState

# ── OpenAI client (lazy init) ─────────────────────────────────────
_client = None
_ai_disabled = False


def _handle_ai_exception(error: Exception, operation: str):
    global _ai_disabled
    message = str(error)
    if "DeploymentNotFound" in message:
        if not _ai_disabled:
            print("[AI Engine] Azure deployment not found. Falling back to mock hints until restart. Set OPENAI_MODEL to a valid Azure deployment name.")
        _ai_disabled = True
        return
    print(f"[AI Engine] {operation} error: {error}")


def _get_client():
    global _client
    if _ai_disabled:
        return None
    if _client is not None:
        return _client
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    if endpoint:
        from openai import AsyncAzureOpenAI
        _client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=endpoint,
        )
    else:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(api_key=api_key)
    return _client


def _get_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o")


def is_ai_available() -> bool:
    """Return True when a usable AI client is configured."""
    return _get_client() is not None


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
) -> str:
    """Generate a progressive hint for a struggling student."""
    # Determine hint level
    if force_level in (1, 2, 3):
        level = force_level
    else:
        level = min(3, student.hint_level + 1)
    student.hint_level = level
    student.hints_given += 1
    if not (class_material or "").strip():
        return _material_required_hint(student.name)
    anchors = _extract_material_anchors(class_material)

    client = _get_client()
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
        response = await client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": HINT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=250,
            temperature=0.7,
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
5. Any plagiarism concerns that need follow-up

Use clear sections with headers. Be specific and data-driven."""


async def generate_session_summary(session: SessionState) -> str:
    """Generate AI-powered post-class summary."""
    client = _get_client()

    # Build student data summary
    student_summaries = []
    for idx, (_, s) in enumerate(session.students.items(), start=1):
        student_summaries.append({
            "name": f"Student {idx}",
            "understanding_score": round(s.understanding_score, 1),
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
        response = await client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
            temperature=0.5,
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

    avg_score = sum(s["understanding_score"] for s in student_data) / total
    struggling = [s for s in student_data if s["understanding_score"] < 50]
    excelling = [s for s in student_data if s["understanding_score"] >= 70]
    paste_concerns = [s for s in student_data if s["large_pastes"] > 0]

    report = f"""📊 **EduPulse Session Report**

## Overall Performance
- **Students:** {total}
- **Average Understanding Score:** {avg_score:.1f}/100
- **Class Status:** {"✅ Good" if avg_score >= 60 else "⚠️ Needs Attention" if avg_score >= 40 else "🚨 Critical"}

## Student Breakdown
"""
    if excelling:
        report += f"\n### 🌟 Excelling ({len(excelling)} students)\n"
        for s in excelling:
            report += f"- **{s['name']}**: Score {s['understanding_score']}, {s['keystrokes']} keystrokes\n"

    if struggling:
        report += f"\n### ⚠️ Struggling ({len(struggling)} students)\n"
        for s in struggling:
            report += f"- **{s['name']}**: Score {s['understanding_score']}, {s['hints_used']} hints used, {s['idle_time']}s idle\n"

    if paste_concerns:
        report += f"\n### 🚨 Plagiarism Concerns ({len(paste_concerns)} students)\n"
        for s in paste_concerns:
            report += f"- **{s['name']}**: {s['large_pastes']} large paste event(s) — requires verbal follow-up\n"

    report += f"""
## Recommendations
- {"Focus next class on reviewing the core concepts — average score below 60." if avg_score < 60 else "Class is progressing well. Consider introducing more advanced challenges."}
- {"Schedule one-on-one time with struggling students." if struggling else "No individual interventions needed."}
- {"Address potential academic integrity concerns with flagged students." if paste_concerns else "No plagiarism concerns detected."}
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

Return format (JSON array):
[
  {
    "question": "What does X do?",
    "options": {"A": "option1", "B": "option2", "C": "option3", "D": "option4"},
    "correct": "B",
    "explanation": "Brief explanation of why B is correct",
    "task_description": "Optional: For practical questions, a task description the student should solve"
  }
]"""


async def generate_quiz(
    task_description: str,
    pdf_text: str = "",
    num_questions: int = 5,
    difficulty: str = "medium",
    mode: str = "practical",
) -> list[dict]:
    """Generate quiz questions from PDF/class material.

    `mode` may be "practical" (code/problems) or "theoretical" (conceptual).
    """
    client = _get_client()

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
        mode_instruction = "\n\nFor PRACTICAL questions: Include a 'task_description' field for each question that describes a coding or problem-solving task the student should complete to answer the question."
    
    if client is None:
        return _mock_quiz(num_questions)

    user_prompt = f"""Generate {num_questions} multiple-choice quiz questions based on this material:

{context}

Difficulty: {difficulty.upper()}
{diff_instruction}{mode_instruction}

Return ONLY a valid JSON array. No markdown, no code blocks, just the JSON."""

    try:
        response = await client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.6,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        questions = json.loads(raw)
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


async def analyze_pdf_content(pdf_text: str, task_description: str = "") -> str:
    """Analyze PDF content and generate teaching insights."""
    client = _get_client()
    if client is None:
        return _mock_pdf_analysis(pdf_text)

    context = f"Lecture Material:\n{pdf_text[:4000]}"
    if task_description:
        context += f"\n\nLab Task:\n{task_description[:500]}"

    try:
        response = await client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": PDF_ANALYSIS_PROMPT},
                {"role": "user", "content": f"Analyze this material:\n\n{context}"},
            ],
            max_tokens=600,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        _handle_ai_exception(e, "PDF analysis")
        return _mock_pdf_analysis(pdf_text)


async def generate_task_description_from_pdf(
    pdf_text: str,
    mode: str = "practical",
    difficulty: str = "medium",
) -> str:
    """Generate a concise class task description from PDF material."""
    client = _get_client()
    normalized_mode = "theoretical" if str(mode).lower() == "theoretical" else "practical"
    normalized_difficulty = str(difficulty).lower()
    if normalized_difficulty not in {"easy", "medium", "hard"}:
        normalized_difficulty = "medium"

    if client is None:
        return _mock_task_description(pdf_text, normalized_mode, normalized_difficulty)

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
- Keep each line under 120 characters.
- Do not include markdown, bullets, emojis, or extra notes.

Material:
{pdf_text[:4500]}
"""

    try:
        response = await client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": "You create high-quality classroom tasks grounded in the provided material."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=280,
            temperature=0.4,
        )
        task = (response.choices[0].message.content or "").strip()
        if not task:
            return _mock_task_description(pdf_text, normalized_mode, normalized_difficulty)
        return _normalize_task_description(task, normalized_mode, pdf_text)
    except Exception as e:
        _handle_ai_exception(e, "Task description generation")
        return _mock_task_description(pdf_text, normalized_mode, normalized_difficulty)


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

