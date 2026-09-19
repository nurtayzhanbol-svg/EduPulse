# EduPulse

Real-time classroom analytics and AI-assisted learning platform (FastAPI + Socket.IO backend, static HTML frontend).

## Quick Start

Requirements: Python 3.11+

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp ../.env.example ../.env        # fill in your AI credentials
set -a && source ../.env && set +a

python main.py                    # http://localhost:8000
```

Pages:

| URL | Purpose |
| --- | --- |
| `/` | Landing page |
| `/teacher.html` | Teacher dashboard (create session, upload PDF, live metrics) |
| `/student.html?session=<id>` | Student workspace |
| `/report.html?session=<id>` | Post-class report |

## Configuration

All settings come from environment variables — see `.env.example`.

| Variable | Description | Default |
| --- | --- | --- |
| `PORT` | Server port | `8000` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key | — |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint; when empty, plain OpenAI is used | — |
| `AZURE_OPENAI_API_VERSION` | Azure API version | `2024-12-01-preview` |
| `OPENAI_API_KEY` | OpenAI key (non-Azure) | — |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint (OpenRouter, Groq, Ollama, …) | OpenAI default |
| `OPENAI_MODEL` | Model name / Azure deployment name | `gpt-5.6-luna` |
| `OPENAI_TIMEOUT_SECONDS` | Per-request timeout | `30` |
| `OPENAI_MAX_RETRIES` | Retries on transient API errors (429/5xx) | `2` |
| `AI_TOKEN_BUDGET` | Token cap per process; `0` means unlimited | `0` |
| `EDUPULSE_DB` | SQLite file for sessions and student aggregates | `./edupulse.db` |
| `SESSION_RETENTION_HOURS` | Hours an ended session is kept before it is deleted automatically | `24` |

Without credentials the AI engine falls back to mock hints, so the app still runs end-to-end.
The same fallback kicks in once `AI_TOKEN_BUDGET` is reached, so a runaway loop can't drain
your API credit — every call logs its token usage and the running total.

## Project Layout

```
backend/
  main.py             FastAPI app, REST endpoints, Socket.IO events
  session_manager.py  session/student lifecycle
  telemetry.py        event processing, attention status & frustration scoring
  ai_engine.py        hints, quiz generation, PDF analysis
  pdf_engine.py       PDF text extraction
  models.py           pydantic models
frontend/             self-contained HTML pages (inline CSS/JS)
samples/              demo/test PDF fixtures
  sample_assignment.pdf  realistic two-page assignment (use this for demos)
  long_sample.pdf        exactly the 20-word upload minimum
  too_short.pdf          3 words — rejected by the upload path (used in tests)
```

## What EduPulse does

- **Teacher** creates a session (optionally from a class PDF), shares a join link, and watches a live
  dashboard: per-student attention status (green / yellow / red = "needs attention now"), idle time,
  help requests, hints sent, large-paste observations and quiz scores. The teacher can end the session (post-class
  report) or delete all of its data.
- **Students** join by link, work in a plain text editor, can ask "I'm confused" for a progressive AI
  hint, and answer the teacher's quiz. Quiz results are the only correctness evidence EduPulse shows;
  behaviour-derived signals (idle time, help requests) are attention signals, not a measure of
  learning. Paste length is only ever an observation: it never changes a student's status.
- **Hints** come from the configured AI provider (Azure OpenAI or OpenAI) or from built-in mock hints
  when no key is set.

## Privacy & data

This section describes what the code actually does; there are no other data flows.

### What the student page sends

| Event | Payload | When |
| --- | --- | --- |
| `keystroke` / `backspace` | `count`, `key_ts` (timestamp) | on each key — never the key itself |
| `idle` | `idle_seconds` | every 5 s |
| `paste` | `length` only — the pasted text is never sent | on paste |
| `help` | `message`, `current_code` (full editor contents) | when the student asks for a hint |
| `code_update` | `code` (full editor contents) | at most once per 15 s while typing, plus once 3 s after typing stops |
| `task_complete` | `task_id`, `tasks_completed` | when the student marks a task done |
| quiz submission (REST) | chosen answers | when the student submits the quiz |

The server ignores `current_code` / `code` on every other event type and caps stored code at
20 000 characters.

### What the server keeps

- **In memory only, while the session is live:** the latest editor code per student (for the teacher's
  live code view and for hint generation). It is never written to disk and is gone when the session
  is deleted or the process restarts.
- **Persisted in SQLite (`EDUPULSE_DB`):** per-student aggregates (keystroke/backspace counts, idle
  seconds, frustration score, status, progress), quiz results, help-request messages, hint metadata
  (count, level, timestamps), teacher nudges as `{message, ts}` (last 50), paste events as
  `{length, timestamp}`, a ring buffer of the last 500 events as `{type, ts}` only, and `consented_at`. Session rows hold the task, uploaded PDF text /
  analysis, alerts, the end-of-session summary and analytics.
- **Never stored:** keystroke logs, code snapshots, clipboard contents or previews.

### Who processes it

Help messages and the editor code sent with `help` / `code_update` are forwarded to the configured AI
provider to generate hints and quizzes. `GET /api/config` reports which one is in use so the consent
notice is truthful: `{"ai_provider": "Azure OpenAI" | "OpenAI" | "none (mock hints)", "session_retention_hours": 24}`.
With `none (mock hints)` nothing leaves the server.

### Who can see what

- Teachers (session token) receive `dashboard_update`, `hint_given`, `alert` (including large-paste
  and confusion-spike alerts) and `quiz_result` in a teacher-only Socket.IO room.
- Each student receives their own `hint` and `teacher_message` (a nudge typed by the teacher) in a
  per-student room. The shared session room carries only `quiz_available`, `task_updated` and an
  empty `session_ended`; class analytics go to the teacher room only. Students never see other
  students' hints, alerts, quiz scores or dashboard state.
- Sessions created from a PDF stay closed until the teacher has reviewed/edited the generated task
  and pressed Launch (`POST /api/sessions/{id}/launch`); joining earlier returns 409.
- Large-paste alerts (200+ characters pasted at once) are a neutral, teacher-only, dismissable
  observation, not an automatic cheating label: they never turn a student red, never reset any
  score, and the post-class summary has no plagiarism/integrity section (only an aggregate paste
  count appears in the dashboard stats). The teacher decides whether to follow up.

### Consent, retention, deletion

- The join form shows a consent notice (what is collected, who processes it, how long it is kept,
  that the teacher can delete it). Joining requires `{"student_name": ..., "consent": true}`;
  `POST /api/sessions/{id}/join` returns 400 otherwise and records `consented_at`.
- Ended sessions are deleted `SESSION_RETENTION_HOURS` (default 24) after they end. A background
  task runs the purge every 15 minutes; it also runs at startup and when a session ends. Active
  sessions are never purged automatically.
- `DELETE /api/sessions/{id}` (teacher token) emits `session_ended` to the session, removes the
  session and all of its students from SQLite and memory, and returns 204. The teacher dashboard
  exposes this as "Delete session data".

## Development

```bash
cd backend && pytest -q
python -m flake8 --select=E9,F backend --exclude backend/venv
```

See `AGENTS.md` and `docs/product-council-report.md` for product constraints and the agreed scope.
