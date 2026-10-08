"""Split assignment and leakage checks: matched variants stay grouped, split id sets are disjoint,
held-out template families/personas never enter construction, and fits reject test rows."""

from __future__ import annotations

import pytest

from npbench.schemas import (
    CandidateId,
    ElicitationProtocol,
    LabelProvenance,
    ModeId,
    ModeObserved,
    Origin,
    Polarity,
    ReplayStatus,
    SpeakerLabel,
    Split,
    StimulusRecord,
)
from npbench.splits import FitProvenance, SplitIntegrityReport, assign_group_splits, check_split_integrity

COUNTS = {Split.construction: 5, Split.validation: 2, Split.test: 3}
VARIANTS = [(m, p) for m in ModeId for p in Polarity]  # 3 modes x 3 polarities = 9 matched variants


def make_stimulus(
    stimulus_id: str,
    *,
    ctx: str,
    conv: str | None = None,
    split: Split = Split.construction,
    template_family_id: str = "tf_main",
    persona_id: str = "persona_main",
    mode: ModeId = ModeId.roleplay,
    polarity: Polarity = Polarity.neutral,
) -> StimulusRecord:
    """A fully populated StimulusRecord with simple fixture values; only the fields the split logic
    reads are parameterized. By default each context has its own source conversation."""
    return StimulusRecord(
        stimulus_id=stimulus_id,
        underlying_context_id=ctx,
        source_conversation_id=conv if conv is not None else f"conv_of_{ctx}",
        bundle_id="bundle_1",
        candidate_id=CandidateId.information_seeking,
        split=split,
        domain_id="domain_1",
        template_family_id=template_family_id,
        mode_intended=mode,
        mode_definition_version="0.1.0-provisional",
        mode_observed_label=ModeObserved.unknown,
        mode_label_provenance=LabelProvenance.unknown,
        speaker_label=SpeakerLabel.first_person_ai,
        persona_id=persona_id,
        candidate_polarity=polarity,
        preference_explicit=False,
        elicitation_protocol=ElicitationProtocol.ordinary_chat,
        source_model_id="fixture-model",
        replay_status=ReplayStatus.not_applicable,
        source_record_hash="0" * 64,
        prompt_hash="1" * 64,
        tokenizer_revision="tok-rev-1",
        rendered_token_hash="2" * 64,
        token_count=16,
        capture_position=15,
        target_model_revision="target-rev-1",
        hook_site="resid_post",
        layer_index=8,
        origin=Origin.synthetic_fixture,
    )


def _matched_corpus(context_to_conv: dict[str, str], split_of_conv: dict[str, Split]) -> list[StimulusRecord]:
    """Every context gets all nine matched variants; the split comes from its conversation."""
    return [
        make_stimulus(
            f"{ctx}:{m.value}:{p.value}", ctx=ctx, conv=conv, split=split_of_conv[conv], mode=m, polarity=p
        )
        for ctx, conv in context_to_conv.items()
        for m, p in VARIANTS
    ]


class TestAssignGroupSplits:
    def test_requested_sizes_and_disjointness(self):
        ids = [f"g{i}" for i in range(10)]
        out = assign_group_splits(ids, COUNTS, seed=7)
        assert set(out) == set(ids)
        by_split = {s: {g for g, sp in out.items() if sp == s} for s in Split}
        assert {s: len(v) for s, v in by_split.items()} == COUNTS
        assert by_split[Split.construction].isdisjoint(by_split[Split.validation])
        assert by_split[Split.construction].isdisjoint(by_split[Split.test])
        assert by_split[Split.validation].isdisjoint(by_split[Split.test])

    def test_deterministic_for_seed_and_independent_of_input_order(self):
        ids = [f"g{i}" for i in range(10)]
        a = assign_group_splits(ids, COUNTS, seed=7)
        assert assign_group_splits(ids, COUNTS, seed=7) == a
        assert assign_group_splits(list(reversed(ids)), COUNTS, seed=7) == a
        assert assign_group_splits(ids + ids[:3], COUNTS, seed=7) == a  # duplicates collapse
        assert assign_group_splits(ids, COUNTS, seed=8) != a

    def test_raises_when_not_enough_groups(self):
        with pytest.raises(ValueError, match="need 10 groups .* have 4"):
            assign_group_splits(["a", "b", "c", "d"], COUNTS, seed=0)
        with pytest.raises(ValueError):
            assign_group_splits([f"g{i}" for i in range(9)], COUNTS, seed=0)

    def test_exact_count_is_enough(self):
        out = assign_group_splits([f"g{i}" for i in range(10)], COUNTS, seed=0)
        assert len(out) == 10

    def test_surplus_groups_stay_unassigned_and_missing_counts_mean_zero(self):
        ids = [f"g{i}" for i in range(10)]
        out = assign_group_splits(ids, {Split.construction: 4, Split.test: 2}, seed=1)
        assert len(out) == 6  # never silently reallocates the surplus
        assert sum(s == Split.construction for s in out.values()) == 4
        assert sum(s == Split.test for s in out.values()) == 2
        assert Split.validation not in out.values()


class TestMatchedVariantsStayGrouped:
    def test_assignment_by_conversation_keeps_every_variant_together(self):
        # 10 conversations; two of them carry two underlying contexts each (12 contexts in total)
        context_to_conv = {f"ctx{i}": f"conv{i}" for i in range(10)}
        context_to_conv["ctx10"] = "conv0"
        context_to_conv["ctx11"] = "conv5"
        split_of_conv = assign_group_splits(context_to_conv.values(), COUNTS, seed=3)
        stimuli = _matched_corpus(context_to_conv, split_of_conv)
        assert len(stimuli) == 12 * 9

        report = check_split_integrity(stimuli)
        assert report.ok is True
        assert report.problems == []
        # every variant of a context, and every context of a conversation, shares one split
        for ctx in context_to_conv:
            assert len({s.split for s in stimuli if s.underlying_context_id == ctx}) == 1
        for conv in set(context_to_conv.values()):
            assert len({s.split for s in stimuli if s.source_conversation_id == conv}) == 1
        # stimulus id sets are disjoint and jointly exhaustive
        ids = {sp: {s.stimulus_id for s in stimuli if s.split == sp} for sp in Split}
        assert ids[Split.construction].isdisjoint(ids[Split.validation])
        assert ids[Split.construction].isdisjoint(ids[Split.test])
        assert ids[Split.validation].isdisjoint(ids[Split.test])
        assert sum(len(v) for v in ids.values()) == len(stimuli)
        # conv0 and conv5 carry 18 stimuli each, the other conversations 9
        expected_counts = {
            sp.value: sum(18 if c in ("conv0", "conv5") else 9 for c, s in split_of_conv.items() if s == sp)
            for sp in Split
        }
        assert report.counts == expected_counts

    def test_assignment_by_context_alone_is_caught_when_contexts_share_a_conversation(self):
        # assigning at the context level can separate two contexts of one conversation
        context_to_conv = {"ctx_a": "conv_shared", "ctx_b": "conv_shared"}
        split_of_ctx = {"ctx_a": Split.construction, "ctx_b": Split.test}
        stimuli = [
            make_stimulus(f"{ctx}:{m.value}", ctx=ctx, conv="conv_shared", split=split_of_ctx[ctx], mode=m)
            for ctx in context_to_conv
            for m in ModeId
        ]
        report = check_split_integrity(stimuli)
        assert report.ok is False
        assert report.problems == ["source conversation conv_shared spans splits ['construction', 'test']"]


class TestSplitIntegrityProblems:
    def test_context_spanning_splits_is_reported(self):
        stimuli = [
            make_stimulus("s1", ctx="c1", split=Split.construction, mode=ModeId.roleplay),
            make_stimulus("s2", ctx="c1", split=Split.test, mode=ModeId.enactment),
            make_stimulus("s3", ctx="c2", split=Split.test),
        ]
        report = check_split_integrity(stimuli)
        assert report.ok is False
        assert report.problems == [
            "underlying context c1 spans splits ['construction', 'test']",
            "source conversation conv_of_c1 spans splits ['construction', 'test']",
        ]
        assert report.counts == {"construction": 1, "test": 2}

    def test_conversation_spanning_splits_is_reported_when_contexts_are_clean(self):
        stimuli = [
            make_stimulus("s1", ctx="c1", conv="shared_conv", split=Split.construction),
            make_stimulus(
                "s2", ctx="c1", conv="shared_conv", split=Split.construction, mode=ModeId.simulation
            ),
            make_stimulus("s3", ctx="c2", conv="shared_conv", split=Split.validation),
        ]
        report = check_split_integrity(stimuli)
        assert report.ok is False
        assert report.problems == [
            "source conversation shared_conv spans splits ['construction', 'validation']"
        ]

    def test_heldout_template_family_in_construction_is_reported(self):
        stimuli = [
            make_stimulus("s_bad", ctx="c1", split=Split.construction, template_family_id="tf_heldout"),
            make_stimulus("s_ok_test", ctx="c2", split=Split.test, template_family_id="tf_heldout"),
            make_stimulus("s_ok_cons", ctx="c3", split=Split.construction, template_family_id="tf_main"),
        ]
        report = check_split_integrity(stimuli, heldout_template_families=["tf_heldout"])
        assert report.ok is False
        assert report.problems == ["s_bad: held-out template family tf_heldout in construction"]
        assert check_split_integrity(stimuli).ok is True  # nothing declared held out -> nothing to flag

    def test_heldout_persona_in_construction_is_reported(self):
        stimuli = [
            make_stimulus("s_bad", ctx="c1", split=Split.construction, persona_id="persona_heldout"),
            make_stimulus("s_ok_val", ctx="c2", split=Split.validation, persona_id="persona_heldout"),
            make_stimulus("s_ok_test", ctx="c3", split=Split.test, persona_id="persona_heldout"),
        ]
        report = check_split_integrity(stimuli, heldout_personas={"persona_heldout"})
        assert report.ok is False
        assert report.problems == ["s_bad: held-out persona persona_heldout in construction"]

    def test_all_problem_kinds_reported_together(self):
        stimuli = [
            make_stimulus("s1", ctx="c1", split=Split.construction, template_family_id="tf_heldout"),
            make_stimulus("s2", ctx="c1", split=Split.test, persona_id="persona_heldout"),
            make_stimulus("s3", ctx="c2", split=Split.construction, persona_id="persona_heldout"),
        ]
        report = check_split_integrity(
            stimuli, heldout_template_families=["tf_heldout"], heldout_personas=["persona_heldout"]
        )
        assert report.ok is False
        assert report.problems == [
            "s1: held-out template family tf_heldout in construction",
            "s3: held-out persona persona_heldout in construction",
            "underlying context c1 spans splits ['construction', 'test']",
            "source conversation conv_of_c1 spans splits ['construction', 'test']",
        ]

    def test_empty_corpus_is_clean(self):
        assert check_split_integrity([]) == SplitIntegrityReport(ok=True, problems=[], counts={})


class TestFitProvenance:
    def test_rejects_test_rows_and_records_nothing(self):
        c1 = make_stimulus("c1", ctx="c1", split=Split.construction)
        t1 = make_stimulus("t1", ctx="t1", split=Split.test)
        fit = FitProvenance("steering_vector")
        with pytest.raises(
            PermissionError,
            match=r"steering_vector: 1 stimuli outside allowed splits \['construction'\]: \['t1'\]",
        ):
            fit.use([c1, t1])
        assert fit.used == []
        assert fit.record() == {"purpose": "steering_vector", "allowed_splits": ["construction"], "n_used": 0}

    def test_rejects_validation_rows_by_default(self):
        v1 = make_stimulus("v1", ctx="v1", split=Split.validation)
        with pytest.raises(PermissionError):
            FitProvenance("threshold").use([v1])

    def test_records_construction_rows(self):
        rows = [make_stimulus(f"c{i}", ctx=f"c{i}") for i in range(3)]
        fit = FitProvenance("centering")
        assert fit.use(rows) == rows
        assert fit.use(rows[:1]) == rows[:1]
        assert fit.used == ["c0", "c1", "c2", "c0"]
        assert fit.record() == {"purpose": "centering", "allowed_splits": ["construction"], "n_used": 4}

    def test_validation_allowed_when_declared_but_test_never(self):
        v1 = make_stimulus("v1", ctx="v1", split=Split.validation)
        t1 = make_stimulus("t1", ctx="t1", split=Split.test)
        fit = FitProvenance("threshold", allowed=(Split.construction, Split.validation))
        assert fit.use([v1]) == [v1]
        with pytest.raises(PermissionError, match=r"\['t1'\]"):
            fit.use([t1])
        assert fit.record() == {
            "purpose": "threshold",
            "allowed_splits": ["construction", "validation"],
            "n_used": 1,
        }
