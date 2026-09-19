"""Task steps: 2-3 numbered, checkable steps generated with the task (mock + real AI paths)."""

from __future__ import annotations

import pytest

import ai_engine
from test_trust_slice_ai import FakeClient, _response


MATERIAL = " ".join(["Python lists are ordered collections. A for loop iterates over any iterable."] * 6)


@pytest.fixture(autouse=True)
def _mock_mode(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ai_engine, "_client", None)
    yield


@pytest.mark.asyncio
async def test_mock_mode_returns_deterministic_steps():
    desc1, steps1 = await ai_engine.generate_task_from_pdf(MATERIAL, "practical", "medium")
    desc2, steps2 = await ai_engine.generate_task_from_pdf(MATERIAL, "practical", "medium")
    assert desc1 == desc2 and steps1 == steps2
    assert ai_engine.MIN_TASK_STEPS <= len(steps1) <= ai_engine.MAX_TASK_STEPS
    assert all(isinstance(s, str) and s and len(s) <= 130 for s in steps1)
    assert steps1[0].startswith("Read the input:")
    assert steps1[1].startswith("Implement the core logic:")
    assert steps1[2].startswith("Handle the edge case:")


@pytest.mark.asyncio
async def test_mock_mode_theoretical_steps():
    _, steps = await ai_engine.generate_task_from_pdf(MATERIAL, "theoretical", "easy")
    assert ai_engine.MIN_TASK_STEPS <= len(steps) <= ai_engine.MAX_TASK_STEPS


@pytest.mark.asyncio
async def test_legacy_description_helper_still_returns_a_string():
    desc = await ai_engine.generate_task_description_from_pdf(MATERIAL)
    assert isinstance(desc, str) and desc.startswith("Task:")


def test_parse_task_steps_reads_numbered_lines_in_order():
    raw = (
        "Task: Sum a list\nInput: ints\nOutput: total\nEdge Case: empty\n"
        "Step 1: Read the numbers.\nstep 2 - Add them up.\n3) Print the total.\nStep 4: Extra ignored."
    )
    assert ai_engine.parse_task_steps(raw) == ["Read the numbers.", "Add them up.", "Print the total."]


def test_parse_task_steps_ignores_non_step_lines():
    assert ai_engine.parse_task_steps("Task: x\nInput: y\nOutput: z\nEdge Case: w") == []
    assert ai_engine.parse_task_steps("") == []


def test_steps_from_description_falls_back_when_description_is_unstructured():
    steps = ai_engine.task_steps_from_description("just some free text")
    assert len(steps) == 2
    assert steps[0].startswith("Write a first version")


@pytest.mark.asyncio
async def test_real_ai_prompt_asks_for_steps_and_parses_them(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    content = (
        "Task: Count vowels in a string.\nInput: One line of text.\nOutput: The vowel count.\n"
        "Edge Case: Empty string gives 0.\nStep 1: Read the line.\nStep 2: Loop over characters and count vowels.\n"
        "Step 3: Print the count."
    )
    client = FakeClient(responses=[_response(50, 60, content)])
    captured = {}

    async def _create(**kwargs):
        captured.update(kwargs)
        return await FakeClient._create(client, **kwargs)

    client.chat.completions.create = _create
    monkeypatch.setattr(ai_engine, "_client", client)

    desc, steps = await ai_engine.generate_task_from_pdf(MATERIAL, "practical", "medium", session_id="s-steps")
    prompt = captured["messages"][-1]["content"]
    assert "Step 1:" in prompt and "2 or 3 short numbered steps" in prompt
    assert desc.startswith("Task: Count vowels")
    assert "Step 1" not in desc
    assert steps == ["Read the line.", "Loop over characters and count vowels.", "Print the count."]


@pytest.mark.asyncio
async def test_real_ai_without_steps_falls_back_to_derived_steps(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    content = "Task: Count vowels.\nInput: text.\nOutput: count.\nEdge Case: empty."
    client = FakeClient(responses=[_response(50, 60, content)])
    monkeypatch.setattr(ai_engine, "_client", client)
    desc, steps = await ai_engine.generate_task_from_pdf(MATERIAL, "practical", "medium", session_id="s-nosteps")
    assert steps == ai_engine.task_steps_from_description(desc)
    assert len(steps) == 3
