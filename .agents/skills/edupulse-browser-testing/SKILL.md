---
name: edupulse-browser-testing
description: Run local EduPulse teacher/student/report browser checks with PDF fixtures and mock AI.
---

# Runtime setup

- Start from the repository's backend directory with `PORT=8000 python main.py`.
- For an explicitly mock-only run, inherited credentials may exist. Launch with
  `env -u OPENAI_API_KEY -u AZURE_OPENAI_API_KEY PORT=8000 python main.py`
  rather than assuming the shell is keyless. Do not print secret values.
- Requirements are in `backend/requirements.txt`; the PDF parser is PyMuPDF
  (`fitz`), not `pypdf`.
- The backend serves `/`, `/teacher.html`, `/student.html`, and `/report.html`
  directly. No separate frontend build/server or account login is needed.
- Check port ownership before starting. After Python changes, restart the local
  backend; do not silently test an old process. Sessions persist in SQLite
  (`EDUPULSE_DB`, default `backend/edupulse.db`) and survive a restart.
- Teacher/student authorization tokens are acquired through create/join UI.
  Keep teacher/report in the same tab because the teacher token is in
  sessionStorage. Reloading the teacher tab (or opening
  `teacher.html?session=<id>` in the same tab) resumes the dashboard from
  sessionStorage; a new tab has no token and shows a readable error plus the
  create-session form.
- External fonts, marked, and socket.io load from CDNs; record network errors
  separately from local asset errors.

# Browser flow

1. Open landing, observe Average understanding across several 3.2-second intervals,
   and check the console before opening the teacher dashboard.
2. Check the current creation form before assuming plain-text creation exists.
   The PDF form requires at least 20 extracted words.
3. Fixtures live in top-level `samples/`: `sample_assignment.pdf` is a realistic
   two-page assignment (use it for demos); `long_sample.pdf` is exactly at the
   20-word minimum; `too_short.pdf` (3 words) is expected to be rejected.
4. Launch a session, open its displayed student URL in another tab, and join with
   a unique name. Session IDs may contain non-hexadecimal characters and hyphens;
   do not hard-code a hex-only ID matcher in evidence collectors.
5. Type multiline code longer than 140 characters and press Return to generate
   telemetry. Open the teacher student-detail card; verify the final line beyond
   the compact-preview boundary. Leave the modal open while editing student code
   again; verify live refresh without reopening. Close it and verify later
   updates do not reopen it.
6. Prefer the visible "I'm confused — get a hint" button, fill its modal and submit.
   Expect a nonempty hint and teacher hint count/activity update.
7. If initial help is unavailable, report that limitation and use genuine idle
   detection instead of hidden JS. Easy hints trigger around 60 seconds, Medium
   90, Hard 120, polled every five seconds. Hard reduces incidental idle hints
   when testing manual help. A visible deeper-hint control can test follow-ups.
8. To exercise integrity alerts, copy >200 editor characters and paste using
   Ctrl+V. A typing tool alone may not emit a native paste event. Expect student
   warning, teacher integrity count and alert/activity update.
9. End Session, confirm student's Session Ended overlay, then Open Full Report.
   Participant/hint counts should match; inspect populated breakdown and topics.
   Hint-based report understanding need not equal live dashboard understanding.

# Evidence

- Record GUI interactions and capture screenshots of styled pages and important
  state changes. Capture passive websocket/console/network logs separately.
- Teacher `dashboard_update` frames should contain `current_code`. Student sockets
  should receive normal hints/alerts but no teacher dashboard frames.
- To inspect public serialization without using browser credentials from shell,
  navigate a new browser tab to `/api/sessions/<id>`. Expect compact
  `current_code_preview` and line count, but no full `current_code`.
- Redact teacher/student tokens in saved headers, JSON and Socket.IO frames.
- Audit removed `/css/` and `/js/` paths separately from favicon/CDN errors.
- Missing controls or unrelated runtime errors must be reported, not bypassed
  with hidden JS. Keep rendering checks separate from metric-label correctness.

# Devin Secrets Needed

None for local browser testing with expected mock hints. Real AI requires
`AZURE_OPENAI_API_KEY` with `AZURE_OPENAI_ENDPOINT`, or `OPENAI_API_KEY`;
deployment/model selection uses `OPENAI_MODEL`. Only test real AI if explicitly
requested and configured.
