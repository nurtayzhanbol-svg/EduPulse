---
name: edupulse-browser-testing
description: Run local EduPulse teacher/student/report browser checks with PDF fixtures and mock AI.
---

# Runtime setup

- Start from the repository's backend directory with `PORT=8000 python main.py`.
- For an explicitly mock-only run, launch with
  `env -u OPENAI_API_KEY -u AZURE_OPENAI_API_KEY PORT=8000 python main.py`.
  Inherited credentials may exist; never print secret values.
- Use the existing environment's Python if dependencies are installed in a venv.
  Requirements are in `backend/requirements.txt`; PDF parsing uses PyMuPDF (`fitz`).
- Backend serves `/`, `/teacher.html`, `/student.html`, and `/report.html`; no
  separate frontend build/server or account login is needed.
- Check port ownership and restart after Python changes. Sessions persist in
  SQLite (`EDUPULSE_DB`, default `backend/edupulse.db`) across restarts.
- Teacher/student tokens come from create/join UI. Keep teacher/report in the
  same tab: credentials are in sessionStorage. A new teacher tab lacks them.
- External fonts, marked, and socket.io load from CDNs; distinguish CDN errors
  from local asset errors.

# Browser flow

1. Upload `samples/sample_assignment.pdf` for realistic demos. PDF creation needs
   at least 20 extracted words; `long_sample.pdf` is exactly at that minimum,
   while `too_short.pdf` should be rejected.
2. Choose Hard to reduce incidental idle hints when testing unrelated features.
   Generate Task, review the draft, then Launch Session.
3. Open the displayed student URL in another tab. Session IDs are not necessarily
   hexadecimal and may contain hyphens.
4. Before joining, enter a name and verify both disabled Join and Enter submission
   are blocked while the data-notice checkbox is unchecked. Tick "I understand"
   and Join to enter the connected workspace.
5. Open the teacher's student-detail modal. Type multiline code longer than the
   compact preview and stop. Verify the final line arrives without reopening the
   modal. Close it and confirm later updates do not reopen it.
6. For timing checks, passively capture outgoing `code_update` and keystroke
   timestamps without tokens. Expect a settled update about 3 seconds after
   typing stops and no more than one active update per 15 seconds. The editor
   uses `onkeyup`: held-key autorepeat is not a substitute for normal typing.
   Use a long ordinary typing action and measure its actual duration rather than
   assuming the typing tool's speed. Show timing output on screen.
7. For large paste, select/copy at least 200 editor characters and paste via
   Ctrl+V. A typing tool alone may not emit native paste. Expect:
   - Teacher-only teal "Large paste — Name" alert with neutral observation text.
   - Green status remains green, "Paste Events" increments, no integrity/risk label.
   - Student sees only its local toast about length, not the teacher alert.
   - Teacher alert × dismisses the card without resetting the count.
8. For hints, use the visible "I'm confused — get a hint" flow. If unavailable,
   report it rather than bypassing with hidden JS. Genuine idle hints trigger
   around Easy 60s, Medium 90s, Hard 120s, polled every five seconds.
9. End Session and verify the student overlay/editor-disabled state. Open Full
   Report in the same teacher tab; inspect counts and search rendered text for
   removed plagiarism/integrity commentary. Verify the UI actually exercises
   any generated report path before claiming AI-report coverage.
10. To delete after report inspection, try browser Back to the existing teacher
    dashboard rather than a new tab. If resume cannot restore the ended dashboard,
    report that limitation; do not inject credentials or force hidden controls.
    Click Delete session data and confirm.
11. Navigate a new browser tab to `/api/sessions/<id>` and verify
    `{"detail":"Session not found"}` plus navigation HTTP status 404. Check student
    socket state separately from the Session Ended overlay: disabling the editor
    does not by itself prove transport disconnection or stopped telemetry.

# Evidence

- Record GUI interactions with annotations and screenshot meaningful states.
- Passively inspect teacher `dashboard_update` frames for `current_code`.
  Student sockets should never receive teacher dashboards or large-paste alerts.
- Public session serialization is compact; teacher authentication enables full
  code. Do not use browser credentials in shell API requests.
- Redact tokens from saved headers, JSON, and Socket.IO frames.
- Keep rendering assertions separate from metric-label correctness.

# Devin Secrets Needed

None for local browser testing with mock hints. Real AI requires
`AZURE_OPENAI_API_KEY` with `AZURE_OPENAI_ENDPOINT`, or `OPENAI_API_KEY`;
deployment/model selection uses `OPENAI_MODEL`. Only test real AI when explicitly
requested and configured.
