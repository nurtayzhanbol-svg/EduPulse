"""Tests for the AI engine's spend guardrails: usage accounting and token budget."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ai_engine


def _response(prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hint"))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@pytest.fixture(autouse=True)
def _reset_usage(monkeypatch):
    monkeypatch.setattr(ai_engine, "_usage", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    monkeypatch.setattr(ai_engine, "_budget_exhausted", False)
    monkeypatch.delenv("AI_TOKEN_BUDGET", raising=False)
    yield


def test_usage_accumulates_across_calls():
    ai_engine._record_usage(_response(100, 20), "hint")
    ai_engine._record_usage(_response(50, 5), "quiz")

    usage = ai_engine.get_usage()
    assert usage["calls"] == 2
    assert usage["prompt_tokens"] == 150
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 175
    assert usage["budget_exhausted"] is False


def test_response_without_usage_is_ignored():
    ai_engine._record_usage(SimpleNamespace(choices=[]), "hint")
    assert ai_engine.get_usage()["calls"] == 0


@pytest.mark.parametrize("budget_value", ["not-a-number", "-5"])
def test_invalid_budget_means_unlimited(monkeypatch, budget_value):
    monkeypatch.setenv("AI_TOKEN_BUDGET", budget_value)
    assert ai_engine._token_budget() == 0


def test_budget_exhaustion_disables_the_client(monkeypatch):
    monkeypatch.setenv("AI_TOKEN_BUDGET", "100")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    ai_engine._record_usage(_response(60, 10), "hint")
    assert ai_engine._budget_exhausted is False

    ai_engine._record_usage(_response(30, 10), "hint")
    assert ai_engine._budget_exhausted is True
    assert ai_engine.get_usage()["total_tokens"] == 110
    # Once the budget is gone, no client is handed out, so callers use mock output.
    assert ai_engine._get_client() is None


@pytest.mark.asyncio
async def test_hint_falls_back_to_mock_when_budget_exhausted(monkeypatch, student):
    monkeypatch.setattr(ai_engine, "_budget_exhausted", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    hint = await ai_engine.generate_hint(
        student=student,
        task_description="Write a loop",
        hint_reason="idle",
        class_material="Key Concepts: loop, counter",
    )
    assert hint
    assert student.name in hint
