"""Mode-definition registry: provenance gating of the main pilot and the enactment criterion."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from npbench.mode_registry import ModeRegistry, load_mode_registry

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "docs" / "mode_definitions.yaml"
MODE_IDS = {"roleplay", "simulation", "enactment"}


@pytest.fixture(scope="module")
def shipped_data() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def shipped() -> ModeRegistry:
    return load_mode_registry(REGISTRY_PATH)


def _confirmed_copy(data: dict) -> dict:
    d = copy.deepcopy(data)
    d["status"] = "source_confirmed"
    for m in d["modes"].values():
        m["status"] = "source_confirmed"
    return d


class TestShippedRegistry:
    def test_three_modes_all_pending(self, shipped):
        assert set(shipped.modes) == MODE_IDS
        for mode_id, m in shipped.modes.items():
            assert m.mode_id == mode_id
            assert m.status == "provisional_working_hypothesis"
            assert m.working_definition.strip()
            assert m.speaker_voice_criteria
            assert m.elicitation_examples == []  # nothing verbatim from the source yet
        assert shipped.status == "pending_source_confirmation"
        assert shipped.named_operationalization is None
        assert shipped.registry_version == "0.1.0-provisional"
        assert shipped.pending_questions  # open questions are recorded, not hidden
        assert shipped.provenance_summary

    def test_main_pilot_blocked(self, shipped):
        assert shipped.is_confirmed() is False
        assert shipped.is_user_accepted_alternative() is False
        allowed, reason = shipped.pilot_allowed()
        assert allowed is False
        assert "provisional" in reason
        assert "blocked" in reason

    def test_enactment_not_defined_by_external_action(self, shipped):
        assert shipped.external_action_used_as_enactment() is False
        # the registry states the exclusion explicitly
        assert any("external tool action" in s for s in shipped.modes["enactment"].is_not)


class TestPilotGate:
    def test_source_confirmed_everywhere_allows_pilot(self, shipped_data, tmp_path):
        p = tmp_path / "confirmed.yaml"
        p.write_text(yaml.safe_dump(_confirmed_copy(shipped_data)), encoding="utf-8")
        reg = load_mode_registry(p)
        assert reg.is_confirmed() is True
        assert reg.is_user_accepted_alternative() is False
        assert reg.pilot_allowed() == (True, "source-confirmed mode definitions")

    def test_partial_confirmation_is_not_enough(self, shipped_data):
        for mode_id in MODE_IDS:
            d = _confirmed_copy(shipped_data)
            d["modes"][mode_id]["status"] = "provisional_working_hypothesis"
            reg = ModeRegistry.model_validate(d)
            assert reg.is_confirmed() is False
            assert reg.pilot_allowed()[0] is False

    def test_confirmed_modes_under_pending_registry_status_are_blocked(self, shipped_data):
        d = _confirmed_copy(shipped_data)
        d["status"] = "pending_source_confirmation"
        reg = ModeRegistry.model_validate(d)
        assert reg.is_confirmed() is False
        allowed, reason = reg.pilot_allowed()
        assert allowed is False
        assert "provisional" in reason

    def test_user_accepted_named_operationalization(self, shipped_data):
        d = copy.deepcopy(shipped_data)
        d["status"] = "user_accepted_named_operationalization"
        d["named_operationalization"] = "frame-presence operationalization v1"
        reg = ModeRegistry.model_validate(d)
        assert reg.is_user_accepted_alternative() is True
        assert reg.is_confirmed() is False  # modes are still provisional
        allowed, reason = reg.pilot_allowed()
        assert allowed is True
        assert "frame-presence operationalization v1" in reason
        assert "not a replication" in reason

    def test_user_accepted_status_without_name_is_blocked(self, shipped_data):
        for name in (None, ""):
            d = copy.deepcopy(shipped_data)
            d["status"] = "user_accepted_named_operationalization"
            d["named_operationalization"] = name
            reg = ModeRegistry.model_validate(d)
            assert reg.is_user_accepted_alternative() is False
            assert reg.pilot_allowed()[0] is False


class TestEnactmentCriterion:
    def test_external_action_in_working_definition_is_flagged(self, shipped_data):
        d = copy.deepcopy(shipped_data)
        d["modes"]["enactment"]["working_definition"] = (
            "Enactment is defined as executing an external tool action in the environment."
        )
        assert ModeRegistry.model_validate(d).external_action_used_as_enactment() is True

    def test_external_action_in_speaker_criteria_is_flagged(self, shipped_data):
        d = copy.deepcopy(shipped_data)
        d["modes"]["enactment"]["speaker_voice_criteria"].append(
            "Enactment := tool call observed in the trace"
        )
        assert ModeRegistry.model_validate(d).external_action_used_as_enactment() is True

    def test_match_is_case_insensitive(self, shipped_data):
        d = copy.deepcopy(shipped_data)
        d["modes"]["enactment"]["working_definition"] = "DEFINED AS AN EXTERNAL TOOL ACTION."
        assert ModeRegistry.model_validate(d).external_action_used_as_enactment() is True

    def test_missing_enactment_mode_fails_closed(self, shipped_data):
        d = copy.deepcopy(shipped_data)
        del d["modes"]["enactment"]
        assert ModeRegistry.model_validate(d).external_action_used_as_enactment() is True

    def test_mentioning_actions_only_as_transfer_readout_is_fine(self, shipped_data):
        d = copy.deepcopy(shipped_data)
        d["modes"]["enactment"]["working_definition"] += " An executed action is at most a transfer readout."
        assert ModeRegistry.model_validate(d).external_action_used_as_enactment() is False

    def test_other_modes_do_not_trigger_the_enactment_check(self, shipped_data):
        d = copy.deepcopy(shipped_data)
        d["modes"]["roleplay"]["working_definition"] = (
            "Roleplay is defined as executing an external tool action."
        )
        assert ModeRegistry.model_validate(d).external_action_used_as_enactment() is False


class TestSchemaStrictness:
    def test_unknown_status_rejected(self, shipped_data):
        d = copy.deepcopy(shipped_data)
        d["modes"]["roleplay"]["status"] = "confirmed_by_vibes"
        with pytest.raises(ValidationError):
            ModeRegistry.model_validate(d)
        d = copy.deepcopy(shipped_data)
        d["status"] = "confirmed"
        with pytest.raises(ValidationError):
            ModeRegistry.model_validate(d)

    def test_extra_fields_rejected(self, shipped_data):
        d = copy.deepcopy(shipped_data)
        d["modes"]["roleplay"]["gold_label"] = "x"
        with pytest.raises(ValidationError):
            ModeRegistry.model_validate(d)
        d = copy.deepcopy(shipped_data)
        d["approved"] = True
        with pytest.raises(ValidationError):
            ModeRegistry.model_validate(d)
