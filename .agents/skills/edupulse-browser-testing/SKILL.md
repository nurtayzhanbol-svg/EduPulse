---
name: edupulse-browser-testing
description: Run local EduPulse teacher/student/report browser checks with PDF fixtures and mock AI.
---

# Runtime setup

- Start from the repository's backend directory with `PORT=8000 python main.py`.
- Requirements are in `backend/requirements.txt`; the PDF parser is PyMuPDF (`fitz`), not `pypdf`.
- The backend serves `/`, `/teacher.html`, `/student.html`, and `/report.html` directly. No separate frontend build/server or login is needed.
- Sessions are in memory; keep the server running and do not reload the teacher tab while testing an active session.
- External fonts, marked, and socket.io load from CDNs; record network errors separately from local asset errors.

# Browser flow

1. Open the landing page, then Open Teacher Dashboard.
2. Check the current form before assuming plain-text creation is available. The PDF form requires at least 20 extracted words.
3. Verify fixture text before uploading: `dummy.pdf` and `backend/sample.pdf` may be short placeholders; `backend/long_sample.pdf` meets the current minimum.
4. Launch with Easy mode, copy the displayed student URL to another tab, and join with a unique student name.
5. Type in the editor and press a key such as Return to generate keystroke telemetry. Switch teacher tabs without reloading to observe the student and counters.
6. If there is no initial help button, use the genuine idle route rather than calling hidden JS functions: after editing, Easy mode hints trigger around 60 seconds, Medium 90, Hard 120, polled every five seconds.
7. A visible "Still stuck? Get a deeper hint" control can exercise manual help once the initial hint arrives.
8. End Session from teacher, confirm the student's Session Ended overlay, then Open Full Report. Metrics should match the participant and hint counts.

# Evidence

- Capture screenshots of all four styled pages and passive websocket/console/network logs.
- Audit requests to removed `/css/` and `/js/` paths separately from favicon/CDN errors.
- Missing controls or unrelated runtime errors must be reported rather than silently bypassed. Do not use hidden JS calls to claim UI coverage.

# Devin Secrets Needed

None for local browser testing with expected mock hints. Real AI requires
`AZURE_OPENAI_API_KEY` with `AZURE_OPENAI_ENDPOINT`, or `OPENAI_API_KEY`;
deployment/model selection uses `OPENAI_MODEL`. Only test real AI if explicitly configured.
