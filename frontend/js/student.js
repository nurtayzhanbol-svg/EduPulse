// ═══════════════════════════════════════════════════════════════════
// EduPulse — Student Workspace · Socket.IO Backend Integration
// ═══════════════════════════════════════════════════════════════════

const API = '';
let socket = null;
let sessionId = null;
let studentName = null;

// Telemetry state
let keystrokeCount = 0;
let pasteCount = 0;
let idleSeconds = 0;
let lastKeystroke = Date.now();

const $ = (id) => document.getElementById(id);

// ── Initialization ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Extract session ID from URL
    const params = new URLSearchParams(window.location.search);
    sessionId = params.get('session');

    if (!sessionId) {
        showError('No session ID in URL. Ask your teacher for the join link.');
        return;
    }

    // Wire up join button
    const joinBtn = $('joinBtn');
    if (joinBtn) {
        joinBtn.addEventListener('click', joinSession);
    }

    // Allow Enter key in name input
    const nameInput = $('nameInput');
    if (nameInput) {
        nameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') joinSession();
        });
    }
});

// ── Join Session ──────────────────────────────────────────────────
async function joinSession() {
    const nameInput = $('nameInput');
    const name = (nameInput?.value || '').trim();
    if (!name) {
        showError('Please enter your name.');
        return;
    }

    studentName = name;
    const joinBtn = $('joinBtn');
    joinBtn.disabled = true;
    joinBtn.textContent = '⏳ Joining…';

    try {
        const res = await fetch(`${API}/api/sessions/${sessionId}/join`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ student_name: name }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Session not found');

        showWorkspace();
        initSocket();
    } catch (e) {
        showError(e.message);
        joinBtn.disabled = false;
        joinBtn.textContent = '🚀  Join Session';
    }
}

function showError(msg) {
    const el = $('joinError');
    if (el) {
        el.textContent = msg;
        el.classList.remove('hidden');
    }
}

function showWorkspace() {
    // Hide join modal, show workspace
    const modal = $('joinModal');
    if (modal) modal.classList.add('hidden');
    const ws = $('workspace');
    if (ws) ws.classList.remove('hidden');

    // Update navbar
    const nameDisplay = $('studentNameDisplay');
    if (nameDisplay) nameDisplay.textContent = `${studentName} · ${sessionId}`;

    // Setup editor events
    setupEditor();
    startTelemetryLoop();
}

// ── Socket.IO ─────────────────────────────────────────────────────
function initSocket() {
    socket = io({ transports: ['websocket', 'polling'] });

    socket.on('connect', () => {
        setConnection(true);
        socket.emit('join_room', {
            session_id: sessionId,
            role: 'student',
            student_name: studentName,
        });
    });

    socket.on('disconnect', () => setConnection(false));
    socket.on('connect_error', () => setConnection(false));

    // Receive hint from AI
    socket.on('hint', (data) => {
        showHint(data.hint || data.message || '', data.level || 0);
    });

    // Receive quiz from teacher
    socket.on('quiz_available', (data) => {
        if (data && data.questions) {
            renderQuiz(data.questions);
        }
    });

    // Session ended by teacher
    socket.on('session_ended', (data) => {
        const editor = $('codeEditor');
        if (editor) editor.disabled = true;
        alert('The teacher has ended this session. Thank you for participating!');
    });

    // Dashboard update (contains task description)
    socket.on('dashboard_update', (data) => {
        if (data && data.task_description) {
            renderTask(data.task_description);
        }
    });
}

function setConnection(connected) {
    const el = $('connectionStatus');
    if (!el) return;
    const dot = el.querySelector('.connection-dot');
    const label = el.querySelector('span');
    if (connected) {
        dot.classList.remove('disconnected');
        label.textContent = 'Connected';
    } else {
        dot.classList.add('disconnected');
        label.textContent = 'Disconnected';
    }
}

// ── Task Rendering ────────────────────────────────────────────────
function renderTask(description) {
    const panel = $('taskPanel');
    if (!panel) return;
    panel.innerHTML = `
    <div style="font-size:0.95rem;line-height:1.8;white-space:pre-wrap;color:var(--text-primary);">
      ${escapeHtml(description)}
    </div>`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Editor Setup ──────────────────────────────────────────────────
function setupEditor() {
    const editor = $('codeEditor');
    if (!editor) return;

    // Keystroke tracking
    editor.addEventListener('keydown', (e) => {
        keystrokeCount++;
        lastKeystroke = Date.now();
        idleSeconds = 0;

        // Send batch telemetry every 20 keystrokes
        if (keystrokeCount % 20 === 0) {
            sendTelemetry('keystroke', { total_keystrokes: keystrokeCount });
        }
    });

    // Paste tracking
    editor.addEventListener('paste', (e) => {
        const text = (e.clipboardData || window.clipboardData).getData('text');
        pasteCount++;
        lastKeystroke = Date.now();

        sendTelemetry('paste', {
            length: text.length,
            content_preview: text.slice(0, 80),
        });
    });

    // Periodic code snapshots on edit
    let codeTimer = null;
    editor.addEventListener('input', () => {
        // Update line count
        const lines = editor.value.split('\n').length;
        const lineEl = $('lineCount');
        if (lineEl) lineEl.textContent = `Lines: ${lines}`;

        // Debounced code update
        clearTimeout(codeTimer);
        codeTimer = setTimeout(() => {
            sendTelemetry('code_update', {
                code: editor.value,
                line_count: editor.value.split('\n').length,
            });
        }, 3000);
    });

    // Tab key support in editor
    editor.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = editor.selectionStart;
            const end = editor.selectionEnd;
            editor.value = editor.value.substring(0, start) + '    ' + editor.value.substring(end);
            editor.selectionStart = editor.selectionEnd = start + 4;
        }
    });
}

// ── Telemetry ─────────────────────────────────────────────────────
function sendTelemetry(eventType, payload = {}) {
    if (!socket || !sessionId) return;
    socket.emit('telemetry', {
        session_id: sessionId,
        student_name: studentName,
        event: {
            event_type: eventType,
            payload: payload,
        },
    });
}

function startTelemetryLoop() {
    setInterval(() => {
        const now = Date.now();
        idleSeconds = Math.floor((now - lastKeystroke) / 1000);
        sendTelemetry('idle', { idle_seconds: idleSeconds });
    }, 10000);
}

// ── Hints ─────────────────────────────────────────────────────────
function showHint(text, level) {
    const container = $('hintContainer');
    if (!container) return;

    // Remove existing hints
    container.innerHTML = '';

    const levels = ['💡 Concept Hint', '🔧 Structural Hint', '📝 Partial Solution'];
    const levelLabel = levels[Math.min(level, 2)];

    const popup = document.createElement('div');
    popup.className = 'hint-popup';
    popup.innerHTML = `
    <button class="hint-close" onclick="this.parentElement.remove()">✕</button>
    <div class="hint-header">${levelLabel}</div>
    <div class="hint-body">${escapeHtml(text)}</div>`;
    container.appendChild(popup);

    // Auto-dismiss after 30s
    setTimeout(() => {
        if (popup.parentElement) {
            popup.classList.add('hiding');
            setTimeout(() => popup.remove(), 300);
        }
    }, 30000);
}

// ── Quiz Rendering ────────────────────────────────────────────────
function renderQuiz(questions) {
    const modal = $('quizModal');
    const container = $('quizQuestions');
    if (!modal || !container) return;

    container.innerHTML = questions.map((q, i) => `
    <div class="glass-card" style="padding:16px;" data-qi="${i}">
      <div style="font-weight:600;margin-bottom:12px;font-size:0.95rem;">
        ${i + 1}. ${escapeHtml(q.question)}
      </div>
      ${q.task_description ? `<div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:12px;font-style:italic;">${escapeHtml(q.task_description)}</div>` : ''}
      <div style="display:flex;flex-direction:column;gap:8px;">
        ${Object.entries(q.options).map(([key, val]) => `
          <label style="display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--bg-glass);border:1px solid var(--border);border-radius:var(--radius-sm);cursor:pointer;transition:var(--transition);"
                 onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--border)'">
            <input type="radio" name="q${i}" value="${key}" style="accent-color:var(--accent);">
            <span><strong>${key}.</strong> ${escapeHtml(val)}</span>
          </label>
        `).join('')}
      </div>
    </div>
  `).join('');

    modal.classList.remove('hidden');

    // Wire submit button
    const submitBtn = $('submitQuizBtn');
    submitBtn.onclick = () => submitQuiz(questions);
}

async function submitQuiz(questions) {
    const answers = {};
    questions.forEach((_, i) => {
        const selected = document.querySelector(`input[name="q${i}"]:checked`);
        if (selected) answers[String(i)] = selected.value;
    });

    if (Object.keys(answers).length < questions.length) {
        alert('Please answer all questions before submitting.');
        return;
    }

    const btn = $('submitQuizBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Submitting…';

    try {
        const res = await fetch(`${API}/api/sessions/${sessionId}/submit-quiz`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ student_name: studentName, answers }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Submission failed');

        showQuizFeedback(data);
    } catch (e) {
        alert('Error: ' + e.message);
        btn.disabled = false;
        btn.textContent = '📝  Submit Answers';
    }
}

function showQuizFeedback(data) {
    const feedback = $('quizFeedback');
    const btn = $('submitQuizBtn');
    if (!feedback) return;

    btn.classList.add('hidden');

    const scoreColor = data.score >= 70 ? 'var(--green)' : data.score >= 40 ? 'var(--yellow)' : 'var(--red)';

    let html = `
    <div class="glass-card" style="padding:20px;text-align:center;margin-bottom:16px;">
      <div style="font-size:2.5rem;font-weight:800;color:${scoreColor};">${data.score}%</div>
      <div style="color:var(--text-secondary);font-size:0.9rem;">${data.correct} of ${data.total} correct</div>
    </div>`;

    if (data.results) {
        html += data.results.map((r, i) => `
      <div class="glass-card" style="padding:14px;margin-bottom:8px;border-left:3px solid ${r.is_correct ? 'var(--green)' : 'var(--red)'};">
        <div style="font-weight:600;margin-bottom:6px;">${i + 1}. ${escapeHtml(r.question)}</div>
        <div style="font-size:0.85rem;color:${r.is_correct ? 'var(--green)' : 'var(--red)'};">
          Your answer: ${r.student_answer} ${r.is_correct ? '✅' : `❌ (Correct: ${r.correct_answer})`}
        </div>
        ${r.explanation ? `<div style="font-size:0.82rem;color:var(--text-muted);margin-top:6px;font-style:italic;">${escapeHtml(r.explanation)}</div>` : ''}
      </div>
    `).join('');
    }

    html += `<button class="btn btn-ghost w-full mt-4" style="justify-content:center;" onclick="document.getElementById('quizModal').classList.add('hidden')">Close Quiz</button>`;

    feedback.innerHTML = html;
    feedback.classList.remove('hidden');
}
