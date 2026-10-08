"""Shared fixtures for the CPU target-lane tests (tiny in-process model, no downloads)."""

from __future__ import annotations

import pytest

from npbench.target import TINY_MODEL_NAME, HFTargetAdapter, build_tiny_llama, render_chat

# Three chats of distinct token lengths (6, 11 and 15 tokens under the tiny template).
MESSAGE_SETS: list[list[dict[str, str]]] = [
    [{"role": "user", "content": "hello world"}],
    [{"role": "user", "content": "the cat sat on the mat ."}],
    [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hello world"},
        {"role": "user", "content": "what is the cat ?"},
    ],
]


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_llama(seed=0)


@pytest.fixture(scope="module")
def adapter(tiny):
    model, tokenizer = tiny
    return HFTargetAdapter(model, tokenizer, model_id=TINY_MODEL_NAME, revision=None, dtype_name="float32")


@pytest.fixture(scope="module")
def message_sets():
    return [list(map(dict, ms)) for ms in MESSAGE_SETS]


@pytest.fixture(scope="module")
def prompts(tiny, message_sets):
    _, tokenizer = tiny
    rendered = [render_chat(tokenizer, ms) for ms in message_sets]
    assert len({r.token_count for r in rendered}) == len(rendered), (
        "fixture prompts must have distinct lengths"
    )
    return rendered
