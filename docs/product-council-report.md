# EduPulse Product Council — Synthesis Report

Six role-based agents each read the repo (`nurtayzhanbol-svg/EduPulse`) and the engineering improvement plan, stated a position, then debated each other's positions in a second round. Full raw transcript: `edupulse_council_results.json`.

Roles: Product Lead, Classroom Teacher, Learning Scientist, Backend Architect, Privacy & Ethics Officer, Skeptical Investor.

---

## 0. Important context: the repo moved during the review

While the council ran, 7 PRs (#1–#7, from other Devin sessions) were merged to `main`. Several plan items are now **done**: bearer-token auth for teacher/student (`backend/auth.py`), longer session IDs, Socket.IO origin allowlist, pytest suite (135 tests) + CI, dead frontend assets removed, `.env.example`, README setup docs, OpenAI model compatibility.

Still open from the original plan: students keyed by display name, unlimited quiz resubmission, room-wide broadcasts, confusion-spike dedup bug (`main.py:702`), two contradictory understanding scores (`main.py:85` vs `telemetry._update_understanding_score`), unbounded event retention.

The council's conclusions below were reached against the pre-merge code but remain valid — none of the merged PRs touched the product-level issues.

---

## 1. Where all six roles agree (unanimous)

1. **The "understanding" score does not measure understanding.** It is `100 − hints·18 − idle − frustration − paste` (`telemetry.py`), and the report uses a *different* formula (`100 − hints·25`). Neither term reflects correctness. `progress = lines × 5`. The only correctness signal that exists — `session.quiz_results` — is never read by analytics or the report. *Every role independently called this the #1 problem.*

2. **Help-seeking is punished.** Pressing "I'm confused" costs 18 points, +0.25 frustration, and locks the student yellow (red after 3 hints) with **no path back to green** for the rest of the session. The Learning Scientist cites this as textbook "help avoidance" training; the Teacher says by minute 15 the attention queue contains most of the class.

3. **The plagiarism feature is a liability, not a feature.** A single ≥200-char paste broadcasts "High plagiarism risk" to the *entire room* (all students), sets `frustration = 0` with the code comment "they're cheating", and forces permanent red. Pasting the PDF's own example triggers it. Every role, including the Investor, wants it downgraded to a neutral, teacher-only "large paste" signal.

4. **The infrastructure-heavy batches of the engineering plan are premature.** Postgres + Redis + multi-worker, Docker compose, frontend module refactor, mypy: all six roles (including the Architect) said defer these until the metric is trustworthy and a pilot exists. The Architect's alternative: single process is fine for 40 students *if* the hot path is fixed; persist ended sessions only, to SQLite.

5. **README overpromises.** QR join, email summary, PDF export (`window.print()`), student-vs-student similarity, "keystrokes not content" — none true. Remove rather than roadmap.

6. **The PDF-grounded progressive hint ladder is the one genuinely differentiated idea** and should be protected while everything else is cut.

## 2. Findings only one role surfaced (but nobody disputed)

| Finding | Raised by | Location |
|---|---|---|
| Every keystroke event carries the **full editor contents** and is broadcast as full class state to **every student socket** → O(N²) traffic + student-to-student privacy leak | Architect, Privacy | `student.html sendTelemetry`, `main.py telemetry` handler `room=session_id` |
| `StudentState.events` stores full code per keystroke, forever; sessions never evicted | Architect | `models.py`, `telemetry.py:39` |
| **Clipboard contents** (first 100 chars of anything pasted) are stored — could be a password or a private message | Privacy | `student.html onPaste`, `telemetry.py:93` |
| No consent screen; student name + 1500 chars of code sent to Azure/OpenAI with no disclosure | Privacy | `student.html` join flow, `ai_engine.generate_hint` |
| An AFK student at hint level 3 triggers an LLM call **every 45 seconds** for the rest of class; `help` events have no cooldown | Architect | `PAUSE_HINT_COOLDOWN_SECONDS`, `telemetry.py:77–84` |
| Hint level is displayed **off-by-one** on both screens (first hint shows as "Level 2 / Structural") | Teacher | `teacher.html:1654`, `student.html:1573` |
| Teacher **cannot review/edit the AI-generated task** before students see it | Teacher | `create_session_from_pdf` |
| Report bucket "Struggling = 5+ hints" is unreachable — hints cap at level 3 and status is red at 3 | Product | `_build_report_payload` |
| Hint "anchor" regex is hardcoded to ~13 CS-101 tokens (`loop`, `modulo`, `odd`…) and bolts "Focus on loop" onto theory hints | Product, Architect | `ai_engine._extract_material_anchors` |
| No owner/roster/institution on a session → nothing a buyer or Stripe can attach to | Investor | `models.SessionState` |

## 3. The real disagreements

### A. Should idle-triggered auto-hints exist at all?
- **Learning Scientist:** remove entirely. Silence ≠ confusion; escalating to a Level-3 partial solution on silence alone is anti-scaffolding.
- **Teacher, Architect, Product:** objected — that kills the differentiated feature.
- **Resolution reached in round 2:** keep **one** restrained Level-1 nudge after a teacher-configurable idle threshold (longer than today's 60s). Level 2/3 require either an explicit help click **or** evidence the student changed their code since the last hint (`code_at_last_hint`). Never lower the score or change status for a hint.

### B. Should code execution be the first thing built?
- **Product & Investor:** yes — a Run button (Pyodide or Judge0) with 2–3 LLM-generated test cases is the single change that makes the word "understanding" honest.
- **Teacher, Architect, Privacy, Learning Scientist:** objected — too large a first step; adds a sandbox, latency, cost, and processes more student code before consent/retention rules exist.
- **Resolution:** **expose quiz correctness first** (already collected, currently ignored). Show "No evidence yet" until a quiz or check exists. Add code execution as a second step, for a constrained flow ("Mark Task Done" runs teacher-supplied tests), only after the pilot.

### C. Pilot now vs. fix first?
- **Investor:** run 3–5 instructors on the current build before building anything.
- **Teacher, Learning Scientist, Privacy:** objected — piloting a dashboard that mislabels behaviour as understanding, publicly flags cheaters, and leaks state to students would produce false negatives and reputational/legal risk.
- **Resolution:** a **thin "trust slice"** first (Section 4), *then* pilot.

### D. Is SQLite persistence safe?
- **Architect:** persist ended sessions to SQLite, cheap.
- **Privacy:** SQLite is not automatically safe — persisting raw events, code snapshots, paste previews, help messages makes the privacy problem durable.
- **Resolution:** persist **aggregates, quiz results, hint metadata, consent state only**. Never durably store keystroke logs, code snapshots or clipboard previews. Add delete endpoint + TTL purge.

## 4. Council-agreed next step: the "trust slice" (before any pilot)

Converged priority across all six round-2 responses, ordered:

1. **Stop calling it understanding.** Rename the metric to "support signals"; show explicit help requests and quiz correctness as *separate* columns; status = "needs attention now", not mastery. Delete the `100 − hints·25` formula and the Mastered/Partial/Struggling-by-hints buckets. Fold `quiz_results` into dashboard + report.
2. **Make status recoverable.** Derive yellow/red from *current* stall (idle ≥ threshold or help request in last 2–3 min); clear after ~20 keystrokes or a larger `code_update`. Cap the Attention Radar at 5 students sorted by seconds-stuck.
3. **Hints: explicit-first, evidence-gated.** One optional Level-1 idle nudge; Level 2/3 only on help click or changed code; per-student cooldown (e.g. 1 hint/60s), hard cap after level 3, per-session LLM budget, 20s timeout, circuit breaker instead of permanent `_ai_disabled`.
4. **Privacy/data contract.** Stop sending `current_code` on every keystroke (only on `help` / debounced `code_update`); drop `content_preview` from pastes; ring-buffer `events` to type+timestamp; separate `{session_id}:teacher` socket room so students never receive class state, other students' hints or alerts; consent notice before join naming Azure/OpenAI as processor; delete endpoint + TTL; make the README privacy section truthful.
5. **Plagiarism → "large paste".** Neutral copy, teacher-only, no automatic red, no `frustration = 0`, dismissable, teacher must confirm before it appears in any report.
6. **Remaining P0 correctness bugs:** server-generated `student_id` instead of name-keyed state; one quiz submission per student; fix confusion-spike dedup to a real time window (and count *currently* stuck students, not lifetime yellow); fix hint-level off-by-one; teacher "review & edit task" step before launch.

Estimated effort: ~2–3 Devin sessions for items 1–6 combined.

**Then pilot** (Investor's criteria, accepted by all): 3–5 instructors, bootcamp / CS1 labs, one promise — *"within five minutes, see the five students who need you and the evidence why"*. Measure: second session run within 14 days, report re-opens, teacher intervention actions, hint accept/dismiss rate, time-to-first-useful-alert, willingness to pay.

## 5. What the council would cut

Consensus cuts: theoretical mode, `frustration_score` in any UI, 9 of the 13 analytics bars (Avg Keystrokes / Code Lines / Frustration / Time in Session…), "Hardest Topics" word-frequency, the auto-generated "Plagiarism Concerns" report section, free-text `POST /api/sessions` create path, README roadmap items (LMS, burnout, emotion detection, similarity engine).

Contested cut (Investor only): quiz generation — "a worse Kahoot". Others kept it because it is currently the *only* correctness signal. **Recommendation: keep.**

## 6. Corrections to the original engineering plan

- P0 items 1, 5 and P2/P3 items 13–15, 17, 19, 20, 22 are **done** (merged PRs #1–#7).
- P0 item 7 ("unify the two scores") is **wrong as stated** — the Learning Scientist's point that both formulas are invalid proxies won the debate. Replace, don't reconcile.
- P1 item 8 (Postgres + Redis + SQLAlchemy) → **downgrade** to SQLite for ended-session aggregates only.
- P3 item 21 (Docker), P4 items 23–24 (module split, vendoring) → **defer** past pilot.
- **Missing from the plan entirely:** data minimisation, consent, socket-room scoping, LLM cost caps, status recovery, the plagiarism UX, teacher task-edit step. These are now the core of the recommended next batch.
