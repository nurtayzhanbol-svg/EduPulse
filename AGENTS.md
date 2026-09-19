# EduPulse — notes for agents

## Read first

- `docs/product-council-report.md` — synthesis of a six-role product council review (product, teacher, learning science, architecture, privacy, go-to-market). It records what is agreed, what is disputed, what to cut, and the ordered next batch ("trust slice"). **Check it before proposing or implementing product-level changes** (scores, statuses, hints, plagiarism/paste handling, telemetry, persistence).

## Product direction (from the council)

- Do not present `understanding_score` or any behaviour-derived metric as a measure of learning. Correctness evidence is quiz results and, later, task checks.
- Status (green/yellow/red) means "needs attention now" and must be recoverable; never lower a score or lock a status because a student asked for help.
- Hints: explicit-first. Automatic hints are at most one Level-1 nudge; deeper levels need a help request or changed code.
- Large-paste detection is a neutral, teacher-only signal — never an automatic cheating label, never broadcast to students.
- Minimise telemetry: no full code on every keystroke, no clipboard previews, no student-visible class state. Add consent/retention before persisting anything.
- Defer Postgres/Redis/multi-worker/Docker until a teacher pilot justifies them.

## Engineering

- Backend: `backend/` (FastAPI + python-socketio, in-memory sessions). Frontend: plain HTML in `frontend/`, served by the backend.
- Tests: `cd backend && pytest`. CI runs lint + tests + server smoke (`.github/workflows/tests.yml`).
- Browser checks: see `.agents/skills/edupulse-browser-testing/SKILL.md`.
- Keep PRs small and scoped; the repo is being worked on by several agents in parallel — rebase on `main` before opening a PR.
