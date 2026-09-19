"""Trust-slice tests for the AI engine: per-session budget, circuit breaker, hint gating."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ai_engine
import telemetry
from conftest import make_event, start_work, student_id

def _response(prompt_tokens: int, completion_tokens: int, content: str = "AI hint about lists"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeClient:
    """Stands in for AsyncOpenAI: returns canned responses or raises, and counts calls."""

    def __init__(self, responses=None, error: Exception | None = None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.responses.pop(0) if self.responses else _response(10, 5)


@pytest.fixture(autouse=True)
def _reset_engine(monkeypatch):
    monkeypatch.setattr(ai_engine, "_usage", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    monkeypatch.setattr(ai_engine, "_budget_exhausted", False)
    monkeypatch.setattr(ai_engine, "_session_usage", {})
    monkeypatch.setattr(ai_engine, "_breaker_failures", 0)
    monkeypatch.setattr(ai_engine, "_breaker_open_until", 0.0)
    monkeypatch.setattr(ai_engine, "_hint_state", {})
    for var in ("AI_TOKEN_BUDGET", "AI_SESSION_TOKEN_BUDGET", "AI_BREAKER_FAILURES",
                "AI_BREAKER_COOLDOWN_SECONDS", "HINT_COOLDOWN_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ai_engine, "_client", client)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return client


MATERIAL = "Python lists are ordered collections. A for loop iterates over any iterable."


# ── Per-session budget ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_budget_exhaustion_falls_back_to_mock_for_that_session_only(fake_client, session, monkeypatch):
    monkeypatch.setenv("AI_SESSION_TOKEN_BUDGET", "100")
    student = session.student_by_name("student0")
    fake_client.responses = [_response(80, 30), _response(10, 5)]

    first = await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL, session_id="s1")
    assert first.startswith("AI hint") and fake_client.calls == 1
    assert ai_engine.session_budget_exhausted("s1") is True
    assert ai_engine.get_usage("s1")["session_tokens"] == 110

    # Session s1 is over budget: mock output, no provider call, no error.
    second = await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL, session_id="s1")
    assert fake_client.calls == 1
    assert "student0" in second and not second.startswith("AI hint")

    # Another session still gets real AI; the process-wide budget is untouched.
    assert ai_engine.is_ai_available("s2") is True
    third = await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL, session_id="s2")
    assert third.startswith("AI hint") and fake_client.calls == 2
    assert ai_engine.get_usage()["budget_exhausted"] is False


def test_session_budget_zero_means_unlimited():
    ai_engine._record_usage(_response(10_000, 10_000), "hint", "s1")
    assert ai_engine.session_budget_exhausted("s1") is False
    assert ai_engine.session_budget_exhausted(None) is False


# ── Circuit breaker ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_breaker_opens_after_consecutive_failures_and_recovers(monkeypatch, session):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AI_BREAKER_FAILURES", "2")
    monkeypatch.setenv("AI_BREAKER_COOLDOWN_SECONDS", "120")
    client = FakeClient(error=RuntimeError("503 upstream unavailable"))
    monkeypatch.setattr(ai_engine, "_client", client)
    student = session.student_by_name("student0")

    for _ in range(2):
        text = await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL, session_id="s1")
        assert "student0" in text  # provider failed -> mock hint, never an exception
    assert client.calls == 2
    assert ai_engine.breaker_is_open() is True
    assert ai_engine.is_ai_available() is False

    # While open, no provider calls are made at all.
    await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL, session_id="s1")
    assert client.calls == 2

    # After the cool-off the breaker lets calls through again (no permanent latch).
    ai_engine._breaker_open_until = ai_engine.time.time() - 1
    assert ai_engine.breaker_is_open() is False
    client.error = None
    text = await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL, session_id="s1")
    assert text.startswith("AI hint") and client.calls == 3
    assert ai_engine._breaker_failures == 0


def test_no_permanent_ai_disabled_latch_remains():
    assert not hasattr(ai_engine, "_ai_disabled")


def test_deployment_not_found_opens_breaker_with_cooldown(monkeypatch):
    monkeypatch.setenv("AI_BREAKER_COOLDOWN_SECONDS", "30")
    now = 1_000.0
    monkeypatch.setattr(ai_engine.time, "time", lambda: now)
    ai_engine._handle_ai_exception(RuntimeError("Error code: 404 - DeploymentNotFound"), "hint")
    assert ai_engine.breaker_is_open(now) is True
    assert ai_engine.breaker_is_open(now + 31) is False


def test_success_resets_failure_count(monkeypatch):
    monkeypatch.setenv("AI_BREAKER_FAILURES", "3")
    ai_engine._handle_ai_exception(RuntimeError("boom"), "hint")
    ai_engine._handle_ai_exception(RuntimeError("boom"), "hint")
    ai_engine._record_success()
    ai_engine._handle_ai_exception(RuntimeError("boom"), "hint")
    assert ai_engine.breaker_is_open() is False


def test_timeout_and_retries_are_applied_to_the_provider_client(monkeypatch):
    """Every provider call goes through the single client built here."""
    captured = {}

    class Spy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", Spy)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "1")
    assert ai_engine._get_client() is not None
    assert captured["timeout"] == 20.0
    assert captured["max_retries"] == 1


# ── Hint gating (explicit-first) ──────────────────────────────────


def test_hint_cap_requires_teacher_help(session):
    student = session.student_by_name("student0")
    student.hints_given = ai_engine.MAX_HINTS_PER_STUDENT
    gate = ai_engine.gate_hint(student, "help_request", now=0.0)
    assert gate.allowed is False
    assert gate.reason == "hint_cap"
    assert "teacher" in gate.message.lower()


def test_first_automatic_nudge_is_level_one(session):
    student = session.student_by_name("student0")
    assert student.hint_level == 0
    gate = ai_engine.gate_hint(student, "idle_threshold_exceeded", now=0.0)
    assert gate.allowed


@pytest.mark.asyncio
async def test_only_one_automatic_nudge_without_explicit_help_or_code_change(session, monkeypatch):
    student = session.student_by_name("student0")
    student.current_code = "x = 1"
    t = 0.0
    monkeypatch.setattr(ai_engine.time, "time", lambda: t)
    await ai_engine.generate_hint(student, "task", "idle_threshold_exceeded", class_material=MATERIAL, force_level=1)
    assert student.hint_level == 1

    # Long after the cooldown, an automatic trigger still may not escalate to Level 2.
    t = 1_000.0
    for reason in ("idle_threshold_exceeded", "pause_threshold_exceeded"):
        gate = ai_engine.gate_hint(student, reason)
        assert not gate.allowed and gate.reason == "needs_explicit_request"

    # An explicit help click unlocks the next level...
    assert ai_engine.gate_hint(student, "help_request").allowed
    await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL)
    assert student.hint_level == 2

    # ...and so does changing the code since the last hint.
    t = 2_000.0
    assert not ai_engine.gate_hint(student, "idle_threshold_exceeded").allowed
    student.current_code = "x = 1\nprint(x)"
    assert ai_engine.code_changed_since_last_hint(student)
    assert ai_engine.gate_hint(student, "idle_threshold_exceeded").allowed


@pytest.mark.asyncio
async def test_per_student_cooldown_applies_even_to_explicit_requests(session, monkeypatch):
    monkeypatch.setenv("HINT_COOLDOWN_SECONDS", "60")
    student = session.student_by_name("student0")
    t = 100.0
    monkeypatch.setattr(ai_engine.time, "time", lambda: t)
    await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL)

    t = 130.0
    gate = ai_engine.gate_hint(student, "help_request")
    assert not gate.allowed and gate.reason == "cooldown"
    assert "30s" in gate.message  # readable to the student

    # Automatic triggers are withheld silently.
    assert ai_engine.gate_hint(student, "idle_threshold_exceeded").message == ""

    t = 161.0
    assert ai_engine.gate_hint(student, "help_request").allowed


@pytest.mark.asyncio
async def test_cooldown_is_per_student(session_factory, monkeypatch):
    session = session_factory(2)
    a, b = session.student_by_name("student0"), session.student_by_name("student1")
    monkeypatch.setattr(ai_engine.time, "time", lambda: 50.0)
    await ai_engine.generate_hint(a, "task", "help_request", class_material=MATERIAL)
    assert not ai_engine.gate_hint(a, "help_request").allowed
    assert ai_engine.gate_hint(b, "help_request").allowed


@pytest.mark.asyncio
async def test_hard_cap_after_level_three(session, monkeypatch):
    student = session.student_by_name("student0")
    t = 0.0
    monkeypatch.setattr(ai_engine.time, "time", lambda: t)
    for _ in range(3):
        assert ai_engine.gate_hint(student, "help_request").allowed
        await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL)
        t += 1_000.0
    assert student.hint_level == 3

    gate = ai_engine.gate_hint(student, "help_request")
    assert not gate.allowed and gate.reason == "level_cap"
    assert "3 hint levels" in gate.message
    assert not ai_engine.gate_hint(student, "idle_threshold_exceeded").allowed

    # Level 3 is terminal even if a generator is forced.
    await ai_engine.generate_hint(student, "task", "help_request", class_material=MATERIAL)
    assert student.hint_level == 3


def test_telemetry_repeated_idle_escalation_is_blocked_by_the_gate(session, monkeypatch):
    """Idle alone never proposes a second hint; and even if a Level-2 proposal reached the
    ai_engine gate without changed code, the gate would refuse it too."""
    student = session.student_by_name("student0")
    start_work(student)
    student.current_code = "x = 1"
    sid = student_id(session, "student0")
    long_idle = session.pause_threshold_seconds * 3

    actions = telemetry.process_telemetry(session, sid, make_event("idle", idle_seconds=long_idle))
    assert actions.get("should_hint") and actions["force_hint_level"] == 1
    assert ai_engine.gate_hint(student, actions["hint_reason"], now=0.0).allowed
    ai_engine._remember_hint_delivery(student, now=0.0)
    student.hint_level = 1

    # telemetry's own 45s pause cooldown has elapsed, but unchanged code means no proposal.
    actions = telemetry.process_telemetry(session, sid, make_event("idle", idle_seconds=long_idle))
    assert not actions.get("should_hint")
    gate = ai_engine.gate_hint(student, "pause_hint", now=1_000.0)
    assert not gate.allowed and gate.reason == "needs_explicit_request"
