"""Open-weight target-model lane: pure PyTorch + transformers extraction, steering and readouts.

Everything here runs on CPU with the in-process :func:`build_tiny_llama` model for engineering tests;
real checkpoints go through :meth:`HFTargetAdapter.from_pretrained` with a pinned revision.
"""

from .adapter import (
    CAPTURE_POSITION,
    DECODER_BLOCK_PATHS,
    HOOK_SITE_DEFINITION,
    HOOK_SITE_TEMPLATE,
    STEER_POSITION_LAST_PREFIX,
    ActionCodeResult,
    CaptureResult,
    ContinuationScore,
    HFTargetAdapter,
    SteerHandle,
    SteeringSpec,
    last_nonpad_positions,
    make_steering,
)
from .render import (
    PROTOCOL_CHAT,
    PROTOCOL_RAW,
    TRANSPORT_CHAT,
    TRANSPORT_RAW,
    RenderedPrompt,
    render_chat,
    render_raw,
    tokenizer_revision_string,
)
from .tiny import (
    CHAT_SPECIAL_TOKENS,
    CHAT_TEMPLATE,
    DEFAULT_VOCAB,
    SPECIAL_TOKENS,
    TINY_MODEL_NAME,
    build_tiny_llama,
    build_tiny_tokenizer,
)

__all__ = [
    "CAPTURE_POSITION",
    "CHAT_SPECIAL_TOKENS",
    "CHAT_TEMPLATE",
    "DECODER_BLOCK_PATHS",
    "DEFAULT_VOCAB",
    "HOOK_SITE_DEFINITION",
    "HOOK_SITE_TEMPLATE",
    "PROTOCOL_CHAT",
    "PROTOCOL_RAW",
    "SPECIAL_TOKENS",
    "STEER_POSITION_LAST_PREFIX",
    "TINY_MODEL_NAME",
    "TRANSPORT_CHAT",
    "TRANSPORT_RAW",
    "ActionCodeResult",
    "CaptureResult",
    "ContinuationScore",
    "HFTargetAdapter",
    "RenderedPrompt",
    "SteerHandle",
    "SteeringSpec",
    "build_tiny_llama",
    "build_tiny_tokenizer",
    "last_nonpad_positions",
    "make_steering",
    "render_chat",
    "render_raw",
    "tokenizer_revision_string",
]
