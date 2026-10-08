"""Official Anthropic SDK adapter.

Credentials resolve through the SDK (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an ``ant auth login``
profile). Model ids are never invented here: ``doctor --live`` lists them from the Models API and the
configuration must name the exact id to use. Unsupported parameter combinations raise a clear error
rather than being silently dropped."""

from __future__ import annotations

import json
import time
from typing import Any

from .base import Provider, ProviderError, ProviderRequest, ProviderResponse


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model_id: str, settings: dict[str, Any] | None = None, **_: Any):
        super().__init__(model_id, settings)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=0)  # retries are the runner's job (logged, capped)
        self.sdk_version = anthropic.__version__

    def list_models(self) -> list[dict[str, Any]]:
        out = []
        for m in self.client.models.list():
            out.append(
                {
                    "id": m.id,
                    "display_name": getattr(m, "display_name", None),
                    "created_at": str(getattr(m, "created_at", "")),
                    "max_input_tokens": getattr(m, "max_input_tokens", None),
                    "max_tokens": getattr(m, "max_tokens", None),
                }
            )
        return out

    def _effective(self, req: ProviderRequest) -> dict[str, Any]:
        s = {**self.settings, **req.settings}
        eff: dict[str, Any] = {"max_tokens": req.max_tokens}
        if "effort" in s:
            eff["output_config.effort"] = s["effort"]
        if "thinking" in s:
            eff["thinking"] = s["thinking"]
        if "temperature" in s:
            # Current Claude models reject sampling params; surface it instead of silently dropping.
            raise ProviderError(
                "temperature is not supported for current Claude models via this adapter; remove it from settings",
                kind="unsupported_parameter",
            )
        return eff

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        a = self._anthropic
        started = time.time()
        eff = self._effective(req)
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": req.max_tokens,
            "messages": req.messages,
        }
        if req.system:
            kwargs["system"] = req.system
        if "output_config.effort" in eff:
            kwargs["output_config"] = {"effort": eff["output_config.effort"]}
        if eff.get("thinking"):
            kwargs["thinking"] = eff["thinking"]
        if req.json_schema is not None:
            kwargs.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": req.json_schema,
            }
        if req.tools:
            kwargs["tools"] = req.tools
        if req.idempotency_key:
            kwargs["metadata"] = {"user_id": req.idempotency_key[:64]}
        try:
            raw = self.client.messages.with_raw_response.create(**kwargs)
            msg = raw.parse()
            request_id = raw.headers.get("request-id")
        except a.AuthenticationError as e:
            raise ProviderError(f"authentication failed: {e.message}", kind="auth") from e
        except a.BadRequestError as e:
            raise ProviderError(
                f"bad request (check model id / parameters): {e.message}", kind="bad_request"
            ) from e
        except a.NotFoundError as e:
            raise ProviderError(f"model or endpoint not found: {e.message}", kind="not_found") from e
        except a.RateLimitError as e:
            raise ProviderError(f"rate limited: {e.message}", kind="rate_limit", retryable=True) from e
        except a.APIStatusError as e:
            raise ProviderError(
                f"api status {e.status_code}: {e.message}", kind="status", retryable=e.status_code >= 500
            ) from e
        except a.APIConnectionError as e:
            raise ProviderError(f"connection error: {e}", kind="connection", retryable=True) from e
        blocks = [b.model_dump() for b in msg.content]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        stop = msg.stop_reason
        usage = msg.usage
        meta = self._meta(
            started,
            returned_model=msg.model,
            request_id=request_id or getattr(msg, "_request_id", None),
            sdk_version=self.sdk_version,
            usage={
                "input": getattr(usage, "input_tokens", None),
                "output": getattr(usage, "output_tokens", None),
                "cached": getattr(usage, "cache_read_input_tokens", None),
            },
            stop_reason=stop,
            raw_text=json.dumps(blocks, sort_keys=True, default=str),
            effective=eff,
        )
        if stop == "refusal":
            details = getattr(msg, "stop_details", None)
            meta.error = f"refusal: {getattr(details, 'category', None)}"
        return ProviderResponse(
            text=text, content_blocks=blocks, stop_reason=stop, meta=meta, raw=msg.to_dict()
        )
