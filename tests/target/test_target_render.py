"""Renderers record the template, token ids and delimiters; raw continuation is never chat-wrapped."""

from __future__ import annotations

import pytest

from npbench.target import (
    PROTOCOL_CHAT,
    PROTOCOL_RAW,
    TRANSPORT_CHAT,
    TRANSPORT_RAW,
    RenderedPrompt,
    render_chat,
    render_raw,
)
from npbench.util import sha256_obj

pytestmark = pytest.mark.target

MESSAGES = [
    {"role": "user", "content": "hello world"},
    {"role": "assistant", "content": "the cat"},
    {"role": "user", "content": "sat on the mat"},
]


def _template_ids(tokenizer, messages, add_generation_prompt):
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(i) for i in out]


def test_render_chat_records_template_tokens_and_hash(tiny):
    _, tokenizer = tiny
    r = render_chat(tokenizer, MESSAGES)
    assert isinstance(r, RenderedPrompt)
    assert r.protocol == PROTOCOL_CHAT and r.transport == TRANSPORT_CHAT
    assert r.text == tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    assert r.message_roles == ["user", "assistant", "user"]
    assert r.turn_delimiters == ["<|user|>", "<|end|>", "<|assistant|>"]
    assert r.generation_suffix == "<|assistant|> "
    assert r.text.endswith(r.generation_suffix)
    assert r.native_prefix == "<|user|> "
    assert r.add_special_tokens is False
    assert r.input_ids == tokenizer(r.text, add_special_tokens=False)["input_ids"]
    assert r.input_ids == _template_ids(tokenizer, MESSAGES, True)
    assert r.token_count == len(r.input_ids) == 15
    assert r.rendered_token_hash == sha256_obj(r.input_ids)
    assert tokenizer.unk_token_id not in r.input_ids
    assert r.input_ids[0] == tokenizer.convert_tokens_to_ids("<|user|>")
    assert r.input_ids[-1] == tokenizer.convert_tokens_to_ids("<|assistant|>")
    assert r.tokenizer_revision == "npbench-tiny-llama@unpinned"
    assert (
        render_chat(tokenizer, MESSAGES, revision="abc123").tokenizer_revision == "npbench-tiny-llama@abc123"
    )


def test_render_chat_without_generation_prompt(tiny):
    _, tokenizer = tiny
    r = render_chat(tokenizer, MESSAGES, add_generation_prompt=False)
    assert r.generation_suffix == ""
    assert not r.text.endswith("<|assistant|> ")
    assert r.input_ids == _template_ids(tokenizer, MESSAGES, False)
    with_gen = render_chat(tokenizer, MESSAGES)
    assert with_gen.input_ids[:-1] == r.input_ids
    assert with_gen.rendered_token_hash != r.rendered_token_hash


def test_render_chat_validates_messages(tiny):
    _, tokenizer = tiny
    with pytest.raises(ValueError):
        render_chat(tokenizer, [])
    with pytest.raises(ValueError):
        render_chat(tokenizer, [{"role": "user"}])


def test_render_raw_is_not_chat_wrapped(tiny):
    _, tokenizer = tiny
    text = "hello world"
    chat = render_chat(tokenizer, [{"role": "user", "content": text}])
    raw = render_raw(tokenizer, text)
    assert raw.protocol == PROTOCOL_RAW and raw.transport == TRANSPORT_RAW
    assert raw.text == text
    assert "<|assistant|>" not in raw.text and "<|user|>" not in raw.text and "<|end|>" not in raw.text
    assert "<|assistant|>" in chat.text
    assert raw.input_ids != chat.input_ids
    assert raw.rendered_token_hash != chat.rendered_token_hash
    chat_only_ids = {tokenizer.convert_tokens_to_ids(t) for t in ("<|user|>", "<|assistant|>", "<|end|>")}
    assert not (set(raw.input_ids) & chat_only_ids)
    assert raw.message_roles == [] and raw.turn_delimiters == [] and raw.generation_suffix == ""
    assert raw.add_special_tokens is False
    assert raw.input_ids[0] == tokenizer.bos_token_id
    assert raw.native_prefix == tokenizer.bos_token == "<s>"
    assert raw.input_ids[1:] == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert raw.token_count == 3
    assert raw.rendered_token_hash == sha256_obj(raw.input_ids)
    assert raw.tokenizer_revision == chat.tokenizer_revision == "npbench-tiny-llama@unpinned"


def test_render_raw_bos_is_explicit(tiny):
    _, tokenizer = tiny
    no_bos = render_raw(tokenizer, "hello world", add_bos=False)
    assert no_bos.input_ids == tokenizer("hello world", add_special_tokens=False)["input_ids"]
    assert tokenizer.bos_token_id not in no_bos.input_ids
    assert no_bos.native_prefix == ""
    with_bos = render_raw(tokenizer, "hello world", add_bos=True)
    assert with_bos.input_ids == [tokenizer.bos_token_id, *no_bos.input_ids]
    with pytest.raises(ValueError):
        render_raw(tokenizer, "", add_bos=False)


def test_render_raw_keeps_only_caller_supplied_special_tokens(tiny):
    _, tokenizer = tiny
    r = render_raw(tokenizer, "<|assistant|> hello", add_bos=False)
    assert r.input_ids == [
        tokenizer.convert_tokens_to_ids("<|assistant|>"),
        tokenizer.convert_tokens_to_ids("hello"),
    ]
    assert r.turn_delimiters == [] and r.generation_suffix == ""
