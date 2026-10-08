"""Versioned mode-definition registry with source provenance.

Intended elicitation mode is an experimental condition, not ground truth that a state is
instantiated. The main pilot requires source-confirmed definitions or an explicit, named,
user-accepted alternative operationalization (which cannot be called a replication)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ModeStatus = Literal[
    "provisional_working_hypothesis", "source_confirmed", "user_accepted_named_operationalization"
]


class ElicitationExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    example_id: str
    source: str
    verbatim: bool
    protocol: str
    text: str
    notes: str | None = None


class ModeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode_id: str
    status: ModeStatus
    working_definition: str
    speaker_voice_criteria: list[str]
    is_not: list[str] = Field(default_factory=list)
    elicitation_examples: list[ElicitationExample] = Field(default_factory=list)
    source_provenance: list[str] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)


class ModeRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registry_version: str
    status: Literal[
        "pending_source_confirmation", "source_confirmed", "user_accepted_named_operationalization"
    ]
    named_operationalization: str | None = None
    provenance_summary: list[str]
    modes: dict[str, ModeDefinition]
    pending_questions: list[str] = Field(default_factory=list)

    def is_confirmed(self) -> bool:
        return self.status == "source_confirmed" and all(
            m.status == "source_confirmed" for m in self.modes.values()
        )

    def is_user_accepted_alternative(self) -> bool:
        return self.status == "user_accepted_named_operationalization" and bool(self.named_operationalization)

    def pilot_allowed(self) -> tuple[bool, str]:
        if self.is_confirmed():
            return True, "source-confirmed mode definitions"
        if self.is_user_accepted_alternative():
            return (
                True,
                f"user-accepted named operationalization: {self.named_operationalization} (not a replication)",
            )
        return False, "mode definitions are provisional; main pilot blocked (exploratory runs allowed)"

    def external_action_used_as_enactment(self) -> bool:
        e = self.modes.get("enactment")
        if e is None:
            return True
        text = (e.working_definition + " ".join(e.speaker_voice_criteria)).lower()
        # The definition must not operationalize enactment as merely executing an external action.
        forbidden = ("defined as executing", "defined as an external tool action", "enactment := tool call")
        return any(f in text for f in forbidden)


def load_mode_registry(path: str | Path) -> ModeRegistry:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return ModeRegistry.model_validate(data)
