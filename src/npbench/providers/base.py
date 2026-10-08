from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..schemas import ProviderCallMeta
from ..util import sha256_text, utc_now_iso


class ProviderError(RuntimeError):
    """Truthful provider failure (network, auth, refusal of the whole chain, parse failure)."""

    def __init__(self, message: str, *, kind: str = "error", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


@dataclass
class ProviderRequest:
    system: str | None
    messages: list[dict[str, Any]]  # [{"role": "user"|"assistant", "content": str | list}]
    max_tokens: int = 2048
    json_schema: dict[str, Any] | None = None  # structured output request
    tools: list[dict[str, Any]] | None = None
    settings: dict[str, Any] = field(default_factory=dict)  # provider-specific effective settings
    idempotency_key: str | None = None
    purpose: str = "judge"


@dataclass
class ProviderResponse:
    text: str
    content_blocks: list[dict[str, Any]]  # provider-native blocks (text, tool_use, ...) as dicts
    stop_reason: str | None
    meta: ProviderCallMeta
    raw: dict[str, Any] = field(default_factory=dict)


class Provider:
    name = "base"
    is_mock = False

    def __init__(self, model_id: str, settings: dict[str, Any] | None = None):
        self.model_id = model_id
        self.settings = dict(settings or {})

    def complete(self, req: ProviderRequest) -> ProviderResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def list_models(self) -> list[dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model_id": self.model_id,
            "is_mock": self.is_mock,
            "settings": self.settings,
        }

    def _meta(
        self,
        started: float,
        *,
        returned_model: str | None,
        request_id: str | None,
        sdk_version: str | None,
        usage: dict[str, int | None],
        stop_reason: str | None,
        raw_text: str,
        effective: dict[str, Any],
        error: str | None = None,
    ) -> ProviderCallMeta:
        now = time.time()
        return ProviderCallMeta(
            provider=self.name,
            requested_model=self.model_id,
            returned_model=returned_model,
            provider_request_id=request_id,
            sdk_version=sdk_version,
            started_at=_iso(started),
            completed_at=utc_now_iso(),
            latency_s=round(now - started, 4),
            input_tokens=usage.get("input"),
            output_tokens=usage.get("output"),
            cached_input_tokens=usage.get("cached"),
            stop_reason=stop_reason,
            raw_response_hash=sha256_text(raw_text) if raw_text else None,
            effective_settings=effective,
            is_mock=self.is_mock,
            error=error,
        )


def _iso(ts: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds")


def build_provider(kind: str, model_id: str, settings: dict[str, Any] | None = None, **kw: Any) -> Provider:
    settings = settings or {}
    if kind == "fake":
        from .fake import FakeProvider

        return FakeProvider(model_id, settings, **kw)
    if kind == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model_id, settings, **kw)
    if kind == "openai_compatible":
        from .openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(model_id, settings, **kw)
    raise ValueError(f"unknown provider kind {kind!r}")
