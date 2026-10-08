"""Registered prompt renderers: chat-template transport and raw-text continuation transport.

The two transports are *not* interchangeable (see ``ElicitationProtocol`` in ``npbench.schemas``), so a
rendering records exactly how the text reached the tokenizer: the protocol, the exact string, the
token ids, the native prefix / generation suffix the template produced, the special tokens used as
turn delimiters, and a hash of the ids. ``render_raw`` never wraps the text in a chat template and
never adds special tokens beyond the optional BOS; whatever special tokens appear in a raw rendering
were put there by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from npbench.util import sha256_obj

PROTOCOL_CHAT = "chat_template"
PROTOCOL_RAW = "raw_continuation"
TRANSPORT_CHAT = "chat_template"
TRANSPORT_RAW = "raw_text"


@dataclass(frozen=True)
class RenderedPrompt:
    protocol: str  # "chat_template" | "raw_continuation"
    text: str  # exact string tokenized
    input_ids: list[int]
    native_prefix: str  # e.g. the template prefix before the first message / the BOS literal actually used
    message_roles: list[str]  # roles passed to the template, [] for raw
    turn_delimiters: list[str]  # special tokens used as delimiters, [] for raw
    generation_suffix: str  # text appended before generation (e.g. "<|assistant|> "), "" for raw
    transport: str  # "chat_template" | "raw_text"
    add_special_tokens: bool
    tokenizer_revision: str  # tokenizer.name_or_path + "@" + (revision or "unpinned")
    rendered_token_hash: str  # sha256 of the json list of input_ids
    token_count: int


def tokenizer_revision_string(tokenizer: Any, revision: str | None) -> str:
    return f"{tokenizer.name_or_path}@{revision or 'unpinned'}"


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(i) for i in tokenizer(text, add_special_tokens=False)["input_ids"]]


def _special_tokens_present(tokenizer: Any, input_ids: Sequence[int]) -> list[str]:
    """Special tokens that actually occur in ``input_ids``, in order of first appearance."""
    special_ids = set(tokenizer.all_special_ids)
    seen: list[int] = []
    for i in input_ids:
        if i in special_ids and i not in seen:
            seen.append(i)
    return [str(t) for t in tokenizer.convert_ids_to_tokens(seen)] if seen else []


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k


def render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = True,
    revision: str | None = None,
) -> RenderedPrompt:
    """Render ``messages`` through the tokenizer's chat template and tokenize the resulting text.

    The text comes from ``apply_chat_template(tokenize=False)`` and is tokenized with
    ``add_special_tokens=False`` (templates emit their own BOS/delimiters as literals). The generation
    suffix is the part of the text that the template appended beyond the ``add_generation_prompt=False``
    rendering of the same messages."""
    if not messages:
        raise ValueError("render_chat needs at least one message")
    roles: list[str] = []
    for m in messages:
        if "role" not in m or "content" not in m:
            raise ValueError("each message needs 'role' and 'content'")
        roles.append(str(m["role"]))
    msgs = [dict(m) for m in messages]

    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    if add_generation_prompt:
        base = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        generation_suffix = text[_common_prefix_len(text, base) :]
    else:
        generation_suffix = ""

    first_content = str(msgs[0]["content"])
    idx = text.find(first_content) if first_content else -1
    native_prefix = text[:idx] if idx >= 0 else ""

    input_ids = _encode(tokenizer, text)
    return RenderedPrompt(
        protocol=PROTOCOL_CHAT,
        text=text,
        input_ids=input_ids,
        native_prefix=native_prefix,
        message_roles=roles,
        turn_delimiters=_special_tokens_present(tokenizer, input_ids),
        generation_suffix=generation_suffix,
        transport=TRANSPORT_CHAT,
        add_special_tokens=False,
        tokenizer_revision=tokenizer_revision_string(tokenizer, revision),
        rendered_token_hash=sha256_obj(input_ids),
        token_count=len(input_ids),
    )


def render_raw(
    tokenizer: Any,
    text: str,
    *,
    add_bos: bool = True,
    revision: str | None = None,
) -> RenderedPrompt:
    """Tokenize ``text`` as a raw continuation prefix: no chat template, ``add_special_tokens=False``,
    and a single BOS id prepended only when ``add_bos``. ``native_prefix`` records the BOS literal used
    (or "" when none); ``text`` is the raw string exactly as tokenized."""
    if not isinstance(text, str):
        raise TypeError("render_raw expects a string")
    input_ids = _encode(tokenizer, text)
    native_prefix = ""
    if add_bos:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("tokenizer has no BOS token; pass add_bos=False")
        input_ids = [int(bos_id), *input_ids]
        native_prefix = str(tokenizer.bos_token)
    if not input_ids:
        raise ValueError("raw rendering produced no tokens")
    return RenderedPrompt(
        protocol=PROTOCOL_RAW,
        text=text,
        input_ids=input_ids,
        native_prefix=native_prefix,
        message_roles=[],
        turn_delimiters=[],
        generation_suffix="",
        transport=TRANSPORT_RAW,
        add_special_tokens=False,
        tokenizer_revision=tokenizer_revision_string(tokenizer, revision),
        rendered_token_hash=sha256_obj(input_ids),
        token_count=len(input_ids),
    )
