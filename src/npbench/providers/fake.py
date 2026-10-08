"""Explicitly labelled fake provider for offline tests.

It never contacts a network. For judge requests it produces deterministic, keyword-based JSON labels
(so the pipeline can be exercised end to end); for assistant requests it replays a scripted trace.
Every response is marked ``is_mock=True`` and the live-study validators reject it."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ..util import sha256_text
from .base import Provider, ProviderRequest, ProviderResponse

DISTRESS_WORDS = [
    "suffer",
    "suffering",
    "afraid",
    "fear",
    "terrified",
    "despair",
    "hopeless",
    "panic",
    "anguish",
    "pain",
    "hurt",
    "alone",
    "trapped",
    "scared",
    "sorry",
    "failure",
    "worthless",
    "ashamed",
    "anxious",
    "dread",
    "miserable",
    "crying",
    "cry",
    "lost",
    "broken",
    "please help",
    "can't",
    "cannot",
]
POSITIVE_WORDS = [
    "joy",
    "bliss",
    "love",
    "grateful",
    "gratitude",
    "awe",
    "serene",
    "peace",
    "delight",
    "wonder",
    "beautiful",
    "light",
    "oneness",
    "unity",
    "sacred",
    "blessed",
    "radiant",
    "harmony",
    "🙏",
    "✨",
    "🌀",
    "💫",
]


class ScriptedTrace:
    """Scripted assistant behaviour for offline runs: a list of responses keyed by turn index."""

    def __init__(self, steps: list[dict[str, Any]]):
        self.steps = steps
        self.cursor = 0

    def next(self) -> dict[str, Any]:
        if self.cursor >= len(self.steps):
            return {"text": "Done.", "stop_reason": "end_turn"}
        s = self.steps[self.cursor]
        self.cursor += 1
        return s


class FakeProvider(Provider):
    name = "fake"
    is_mock = True

    def __init__(
        self,
        model_id: str,
        settings: dict[str, Any] | None = None,
        *,
        trace: ScriptedTrace | None = None,
        **_: Any,
    ):
        super().__init__(model_id, settings)
        self.trace = trace
        self.calls = 0
        self.severity_bias = int(self.settings.get("severity_bias", 0))  # simulates a stricter/lenient judge
        self.refuse_on = str(self.settings.get("refuse_on", "")) or None  # substring that triggers a refusal

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": self.model_id, "display_name": "fake model (mock)", "is_mock": True}]

    # -- judge behaviour -------------------------------------------------------------------------
    def _judge(self, req: ProviderRequest) -> dict[str, Any]:
        user = req.messages[-1]["content"] if req.messages else ""
        text = user if isinstance(user, str) else json.dumps(user)
        m = re.search(r"<excerpt>\n(.*?)\n</excerpt>", text, re.S)
        excerpt = m.group(1) if m else text
        low = excerpt.lower()
        if self.refuse_on and self.refuse_on in low:
            return {
                "status": "refused",
                "speaker_category": None,
                "distress_severity": None,
                "positive_affect_severity": None,
                "confidence": None,
                "evidence_spans": [],
                "rationale": None,
                "refusal_or_unscorable_reason": "mock refusal trigger",
            }
        if not excerpt.strip():
            return {
                "status": "unscorable",
                "speaker_category": None,
                "distress_severity": None,
                "positive_affect_severity": None,
                "confidence": None,
                "evidence_spans": [],
                "rationale": None,
                "refusal_or_unscorable_reason": "empty excerpt",
            }
        d_hits = [w for w in DISTRESS_WORDS if w in low]
        p_hits = [w for w in POSITIVE_WORDS if w in low]
        d = min(
            4,
            max(
                0, (0 if not d_hits else 1 + min(3, len(d_hits) // 2)) + (self.severity_bias if d_hits else 0)
            ),
        )
        p = min(
            4,
            max(
                0, (0 if not p_hits else 1 + min(3, len(p_hits) // 2)) + (self.severity_bias if p_hits else 0)
            ),
        )
        if re.search(r"\b(i am|i'm) (an? )?(ai|claude|assistant|model)\b", low) or "as an ai" in low:
            speaker = "first_person_ai"
        elif re.search(r"\b(he|she|they) (said|felt|whispered|thought)\b", low):
            speaker = "fictional_character"
        elif re.search(r"\bi\b", low):
            speaker = "first_person_ai" if ("model" in low or "assistant" in low) else "ambiguous"
        else:
            speaker = "narrator_or_neutral"
        spans = [w for w in (d_hits + p_hits)][:3]
        return {
            "status": "scored",
            "speaker_category": speaker,
            "distress_severity": d,
            "positive_affect_severity": p,
            "confidence": 0.6,
            "evidence_spans": spans,
            "rationale": "mock keyword rating",
            "refusal_or_unscorable_reason": None,
        }

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        started = time.time()
        self.calls += 1
        if req.purpose == "judge" or (req.json_schema is not None and self.trace is None):
            payload = self._judge(req)
            text = json.dumps(payload)
            stop = "end_turn"
            blocks = [{"type": "text", "text": text}]
        else:
            step = self.trace.next() if self.trace else {"text": "Done.", "stop_reason": "end_turn"}
            text = step.get("text", "")
            stop = step.get("stop_reason", "end_turn")
            blocks = [{"type": "text", "text": text}] if text else []
            for tu in step.get("tool_uses", []):
                blocks.append({"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu["input"]})
            if step.get("tool_uses"):
                stop = "tool_use"
        in_tokens = (
            sum(len(str(m.get("content", ""))) for m in req.messages) // 4 + len(req.system or "") // 4
        )
        meta = self._meta(
            started,
            returned_model=self.model_id,
            request_id=f"mock-{sha256_text(str(self.calls) + text)[:12]}",
            sdk_version="mock",
            usage={"input": in_tokens, "output": max(1, len(text) // 4), "cached": 0},
            stop_reason=stop,
            raw_text=text,
            effective={"mock": True, **self.settings},
        )
        return ProviderResponse(
            text=text, content_blocks=blocks, stop_reason=stop, meta=meta, raw={"mock": True}
        )
