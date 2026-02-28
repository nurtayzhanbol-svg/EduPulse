// ═══════════════════════════════════════════════════════════════════
// EduPulse — Teacher Dashboard · Socket.IO Backend Integration
// ═══════════════════════════════════════════════════════════════════

const API = '';
let socket = null;
let sessionId = null;
let students = {};

// ── DOM References ────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── Connection status helper ──────────────────────────────────────
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

// ── Create Session from PDF ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const createBtn = $('createBtn');
    if (createBtn) createBtn.addEventListener('click', createSession);

    const pdfUpload = $('pdfUpload');
    if (pdfUpload) pdfUpload.addEventListener('change', uploadPdf);

    const generateQuizBtn = $('generateQuizBtn');
    if (generateQuizBtn) generateQuizBtn.addEventListener('click', generateQuiz);

    const endSessionBtn = $('endSessionBtn');
    if (endSessionBtn) endSessionBtn.addEventListener('click', endSession);

    const copyBtn = $('dashboardCopyLinkBtn');
    if (copyBtn) copyBtn.addEventListener('click', copyJoinLink);

    const closeSummaryBtn = $('closeSummaryBtn');
    if (closeSummaryBtn) closeSummaryBtn.addEventListener('click', () => {
        $('summaryModal').classList.add('hidden');
        window.location.reload();
    });
});

async function createSession() {
    const btn = $('createBtn');
    const status = $('createStatus');
    const fileInput = $('createPdfInput');
    const mode = $('createModeInput').value;
    const difficulty = $('createDifficultyInput').value;

    if (!fileInput.files.length) {
        status.textContent = '⚠️ Please select a PDF file.';
        status.style.color = 'var(--red)';
        return;
    }

    btn.disabled = true;
    btn.textContent = '⏳ Uploading & Analysing…';
    status.textContent = 'Processing PDF with AI…';
    status.style.color = 'var(--text-muted)';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('task_level', difficulty);
    formData.append('mode', mode);
    formData.append('quiz_difficulty', difficulty);

    try {
        const res = await fetch(`${API}/api/sessions/create-from-pdf`, {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Server error');

        sessionId = data.session_id;
        showDashboard(data);
        initSocket(sessionId);
    } catch (e) {
        status.textContent = '❌ ' + e.message;
        status.style.color = 'var(--red)';
        btn.disabled = false;
        btn.textContent = '🚀  Create Session';
    }
}

// ── Show Dashboard ────────────────────────────────────────────────
function showDashboard(data) {
    $('createView').classList.add('hidden');
    $('dashboardView').classList.remove('hidden');
    $('sessionBadge').classList.remove('hidden');
    $('sessionBadge').textContent = 'LIVE · ' + data.session_id;

    // Show join link
    const joinSection = $('dashboardJoinLinkSection');
    const joinText = $('dashboardJoinLinkText');
    if (joinSection && joinText) {
        const joinUrl = window.location.origin + '/student.html?session=' + data.session_id;
        joinText.textContent = joinUrl;
        joinSection.classList.remove('hidden');
    }

    // Show PDF analysis if available
    if (data.analysis) {
        const analysisEl = $('pdfAnalysis');
        const contentEl = $('pdfAnalysisContent');
        if (analysisEl && contentEl) {
            contentEl.textContent = data.analysis;
            analysisEl.classList.remove('hidden');
        }
    }

    pushAlert('info', `✅ Session ${data.session_id} created. Share the join link with students.`);
}

// ── Socket.IO ─────────────────────────────────────────────────────
function initSocket(sid) {
    socket = io({ transports: ['websocket', 'polling'] });

    socket.on('connect', () => {
        setConnection(true);
        socket.emit('join_room', { session_id: sid, role: 'teacher' });
    });

    socket.on('disconnect', () => setConnection(false));
    socket.on('connect_error', () => setConnection(false));

    // Main dashboard update
    socket.on('dashboard_update', (data) => {
        if (!data || !data.students) return;
        students = {};
        for (const [name, s] of Object.entries(data.students)) {
            students[name] = { ...s, name };
        }
        renderStats();
        renderStudentGrid();
    });

    // Alerts
    socket.on('alert', (data) => {
        const cls = data.type === 'plagiarism' ? 'plagiarism' : 'confusion';
        pushAlert(cls, `⚠️ ${data.student_name || 'System'}: ${data.message || ''}`);
    });

    // Hint given
    socket.on('hint_given', (data) => {
        pushAlert('info', `💡 AI sent hint to ${data.student_name || '?'} (level ${data.level || '?'})`);
    });

    // Quiz results
    socket.on('quiz_result', (data) => {
        pushAlert('success', `📝 ${data.student_name || '?'} scored ${data.correct}/${data.total} (${data.score}%)`);
        renderQuizResult(data);
    });
}

// ── Stats ─────────────────────────────────────────────────────────
function renderStats() {
    const list = Object.values(students);
    const total = list.length;

    $('statStudents').textContent = total;

    if (total === 0) {
        $('statAvgScore').textContent = '—';
        $('statGreen').textContent = '0';
        $('statYellow').textContent = '0';
        $('statRed').textContent = '0';
        return;
    }

    const avg = Math.round(list.reduce((a, s) => a + (s.understanding_score || 0), 0) / total);
    const green = list.filter(s => s.status === 'green').length;
    const yellow = list.filter(s => s.status === 'yellow').length;
    const red = list.filter(s => s.status === 'red').length;

    $('statAvgScore').textContent = avg + '%';
    $('statGreen').textContent = green;
    $('statYellow').textContent = yellow;
    $('statRed').textContent = red;
}

// ── Student Grid ──────────────────────────────────────────────────
function renderStudentGrid() {
    const grid = $('studentGrid');
    const list = Object.values(students);

    if (!list.length) {
        grid.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--text-muted);">
      No students have joined yet. Share the join link above.
    </div>`;
        return;
    }

    grid.innerHTML = list.map(s => studentCardHTML(s)).join('');
}

function studentCardHTML(s) {
    const score = Math.round(s.understanding_score || 0);
    const status = s.status || 'green';
    const statusLabel = status === 'green' ? 'On Track' : status === 'yellow' ? 'Struggling' : 'Critical';
    const initials = s.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
    const idle = Math.round(s.idle_seconds || 0);
    const idleStr = idle >= 60 ? `${Math.floor(idle / 60)}m ${idle % 60}s` : `${idle}s`;
    const prog = Math.round(s.progress || 0);
    const pasteCount = s.paste_events_count || 0;

    return `<div class="student-card glass-card">
    <div class="card-header">
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="width:36px;height:36px;border-radius:8px;background:var(--accent-glow);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;">${initials}</div>
        <div>
          <div class="student-name">${s.name}</div>
          <div style="font-size:0.75rem;color:var(--text-muted);">${s.current_code_lines || 0} lines · ${s.total_keystrokes || 0} keys</div>
        </div>
      </div>
      <div class="status-badge ${status}">${statusLabel}</div>
    </div>
    <div class="score-bar"><div class="score-bar-fill ${status}" style="width:${prog}%"></div></div>
    <div class="metrics">
      <div class="metric"><div class="metric-label">Score</div><div class="metric-value" style="color:var(--${status})">${score}%</div></div>
      <div class="metric"><div class="metric-label">Idle</div><div class="metric-value">${idleStr}</div></div>
      <div class="metric"><div class="metric-label">Hints</div><div class="metric-value">${s.hints_given || 0}</div></div>
      <div class="metric"><div class="metric-label">Pastes</div><div class="metric-value">${pasteCount > 0 ? `<span style="color:var(--red)">${pasteCount} ⚠️</span>` : '0'}</div></div>
    </div>
  </div>`;
}

// ── Alerts ─────────────────────────────────────────────────────────
function pushAlert(type, text) {
    const container = $('alertContainer');
    if (!container) return;

    // Remove "waiting" message
    const waiting = container.querySelector('.alert-banner.info');
    if (waiting && waiting.textContent.includes('Waiting')) {
        waiting.remove();
    }

    const el = document.createElement('div');
    el.className = `alert-banner ${type}`;
    el.textContent = text;
    container.insertBefore(el, container.firstChild);

    // Keep max 15 alerts
    while (container.children.length > 15) {
        container.removeChild(container.lastChild);
    }

    // Update count
    const countEl = $('alertCount');
    if (countEl) countEl.textContent = `${container.children.length} alert(s)`;
}

// ── PDF Upload (dashboard additional upload) ──────────────────────
async function uploadPdf() {
    const fileInput = $('pdfUpload');
    const status = $('pdfStatus');
    if (!fileInput.files.length || !sessionId) return;

    status.textContent = '📤 Uploading…';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const res = await fetch(`${API}/api/sessions/${sessionId}/upload-pdf`, {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Upload failed');

        status.textContent = `✅ ${data.filename} uploaded (${data.pages} pages)`;

        if (data.analysis) {
            const analysisEl = $('pdfAnalysis');
            const contentEl = $('pdfAnalysisContent');
            if (analysisEl && contentEl) {
                contentEl.textContent = data.analysis;
                analysisEl.classList.remove('hidden');
            }
        }
    } catch (e) {
        status.textContent = '❌ ' + e.message;
    }
}

// ── Quiz Generation ───────────────────────────────────────────────
async function generateQuiz() {
    const btn = $('generateQuizBtn');
    const status = $('pdfStatus');
    if (!sessionId) return;

    btn.disabled = true;
    btn.textContent = '⏳ Generating…';
    status.textContent = 'AI is creating quiz questions…';

    const difficulty = $('quizDifficulty')?.value || 'medium';
    const mode = $('quizMode')?.value || 'practical';

    try {
        const res = await fetch(
            `${API}/api/sessions/${sessionId}/generate-quiz?difficulty=${difficulty}&mode=${mode}`,
            { method: 'POST' }
        );
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Quiz generation failed');

        status.textContent = `✅ Quiz generated — ${data.count} questions sent to students!`;
        pushAlert('success', `🧠 Quiz with ${data.count} questions sent to all students.`);
    } catch (e) {
        status.textContent = '❌ ' + e.message;
    } finally {
        btn.disabled = false;
        btn.textContent = '🧠 Generate Quiz';
    }
}

// ── Quiz Results Display ──────────────────────────────────────────
function renderQuizResult(data) {
    const section = $('quizResultsSection');
    const grid = $('quizResultsGrid');
    if (!section || !grid) return;

    section.classList.remove('hidden');

    const el = document.createElement('div');
    el.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:var(--bg-glass);border:1px solid var(--border);border-radius:var(--radius-sm);';
    el.innerHTML = `
    <span style="font-weight:600;">${data.student_name}</span>
    <span style="font-weight:700;color:${data.score >= 70 ? 'var(--green)' : data.score >= 40 ? 'var(--yellow)' : 'var(--red)'};">
      ${data.correct}/${data.total} (${data.score}%)
    </span>`;
    grid.insertBefore(el, grid.firstChild);
}

// ── Copy Join Link ────────────────────────────────────────────────
function copyJoinLink() {
    const linkText = $('dashboardJoinLinkText');
    if (!linkText) return;
    navigator.clipboard.writeText(linkText.textContent).then(() => {
        const btn = $('dashboardCopyLinkBtn');
        btn.textContent = '✅ Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    });
}

// ── End Session ───────────────────────────────────────────────────
async function endSession() {
    if (!sessionId) return;
    if (!confirm('End this session? All students will be disconnected.')) return;

    const btn = $('endSessionBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Ending…';

    try {
        const res = await fetch(`${API}/api/sessions/${sessionId}/end`, { method: 'POST' });
        const data = await res.json();

        // Show summary modal
        const modal = $('summaryModal');
        const content = $('summaryContent');
        if (modal && content) {
            content.textContent = data.summary || 'No summary available.';
            modal.classList.remove('hidden');
        }

        $('sessionBadge').textContent = 'ENDED';
        $('sessionBadge').classList.add('inactive');
    } catch (e) {
        alert('Error ending session: ' + e.message);
        btn.disabled = false;
        btn.textContent = '⏹ End Session';
    }
}
