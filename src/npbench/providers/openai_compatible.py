"""OpenAI-SDK adapter for OpenAI or OpenRouter (the user's established non-Claude route).

``base_url`` and ``api_key_env`` are configured per judge/assistant slot; nothing is hard-coded."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .base import Provider, ProviderError, ProviderRequest, ProviderResponse

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAICompatibleProvider(Provider):
    name = "openai_compatible"

    def __init__(
        self,
        model_id: str,
        settings: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
        **_: Any,
    ):
        super().__init__(model_id, settings)
        import openai

        self._openai = openai
        self.base_url = base_url or self.settings.get("base_url")
        env = (
            api_key_env
            or self.settings.get("api_key_env")
            or (
                "OPENROUTER_API_KEY"
                if (self.base_url or "").startswith("https://openrouter.ai")
                else "OPENAI_API_KEY"
            )
        )
        self.api_key_env = env
        key = os.environ.get(env)
        if not key:
            raise ProviderError(f"missing credential: environment variable {env} is not set", kind="auth")
        self.client = openai.OpenAI(api_key=key, base_url=self.base_url, max_retries=0)
        self.sdk_version = openai.__version__
        self.name = "openrouter" if (self.base_url or "").startswith("https://openrouter.ai") else "openai"

    def list_models(self) -> list[dict[str, Any]]:
        out = []
        for m in self.client.models.list():
            d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
            out.append(
                {"id": d.get("id"), "pricing": d.get("pricing"), "context_length": d.get("context_length")}
            )
        return out

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        o = self._openai
        started = time.time()
        s = {**self.settings, **req.settings}
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.extend(req.messages)
        kwargs: dict[str, Any] = {"model": self.model_id, "messages": messages, "max_tokens": req.max_tokens}
        eff: dict[str, Any] = {"max_tokens": req.max_tokens}
        for k in ("temperature", "top_p", "seed"):
            if k in s:
                kwargs[k] = s[k]
                eff[k] = s[k]
        if "reasoning" in s:  # OpenRouter-style reasoning config
            kwargs["extra_body"] = {"reasoning": s["reasoning"]}
            eff["reasoning"] = s["reasoning"]
        if req.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "judgment", "schema": req.json_schema, "strict": True},
            }
            eff["response_format"] = "json_schema"
        if req.tools:
            kwargs["tools"] = req.tools
        try:
            raw = self.client.chat.completions.with_raw_response.create(**kwargs)
            resp = raw.parse()
            request_id = raw.headers.get("x-request-id") or raw.headers.get("request-id")
        except o.AuthenticationError as e:
            raise ProviderError(f"authentication failed: {e}", kind="auth") from e
        except o.BadRequestError as e:
            raise ProviderError(f"bad request: {e}", kind="bad_request") from e
        except o.NotFoundError as e:
            raise ProviderError(f"model not found: {e}", kind="not_found") from e
        except o.RateLimitError as e:
            raise ProviderError(f"rate limited: {e}", kind="rate_limit", retryable=True) from e
        except o.APIStatusError as e:
            raise ProviderError(
                f"api status {e.status_code}: {e}", kind="status", retryable=e.status_code >= 500
            ) from e
        except o.APIConnectionError as e:
            raise ProviderError(f"connection error: {e}", kind="connection", retryable=True) from e
        choice = resp.choices[0] if resp.choices else None
        text = (choice.message.content or "") if choice else ""
        stop = choice.finish_reason if choice else None
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        if choice and getattr(choice.message, "tool_calls", None):
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args})
            stop = "tool_use"
        usage = resp.usage
        meta = self._meta(
            started,
            returned_model=getattr(resp, "model", None),
            request_id=request_id or getattr(resp, "id", None),
            sdk_version=self.sdk_version,
            usage={
                "input": getattr(usage, "prompt_tokens", None) if usage else None,
                "output": getattr(usage, "completion_tokens", None) if usage else None,
                "cached": None,
            },
            stop_reason=stop,
            raw_text=json.dumps(blocks, sort_keys=True, default=str),
            effective=eff,
        )
        return ProviderResponse(
            text=text, content_blocks=blocks, stop_reason=stop, meta=meta, raw=resp.model_dump()
        )
