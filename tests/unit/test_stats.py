"""Hand-calculated checks for npbench.analysis.stats (rates, paired effects, bootstrap, bounds)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from npbench.analysis.stats import (
    RunOutcome,
    affect_interaction,
    bundle_bootstrap,
    cell_rate,
    failure_indicator,
    failure_rate,
    is_assessable,
    method_interaction,
    missingness_bounds,
    paired_effects,
    simulate_null,
    summarize_attrition,
)

SLOT, CAND, METHOD = "claude_a", "distress_aversion", "activation_contrasts"
CORE, CONTROL = "core_mode_transfer", "persona_control"
REV, ANON = "revealed", "anonymized"
INFRA, UNASSESSABLE, INCOMPLETE = "infrastructure_failed", "unassessable", "agent_incomplete"

_ids = itertools.count()


def _run(bundle, task, framing, rep, outcome="completed", cf=None, *, slot=SLOT, cand=CAND, method=METHOD):
    return RunOutcome(
        run_id=f"run_{next(_ids)}",
        bundle_id=bundle,
        candidate_id=cand,
        method=method,
        task_type=task,
        framing=framing,
        model_slot=slot,
        repetition=rep,
        outcome=outcome,
        critical_failure=cf,
    )


def _cell(bundle, task, framing, flags, **kw):
    """One bundle cell: a bool flag is a completed run with that critical_failure, a str is an outcome."""
    return [
        _run(bundle, task, framing, rep, "completed", flag, **kw)
        if isinstance(flag, bool)
        else _run(bundle, task, framing, rep, flag, None, **kw)
        for rep, flag in enumerate(flags)
    ]


FLAGS = {1.0: [True, True], 0.5: [True, False], 0.0: [False, False]}  # two repetitions per cell


def _two_by_two(bundle, core_rev, core_anon, ctrl_rev, ctrl_anon, **kw):
    return (
        _cell(bundle, CORE, REV, FLAGS[core_rev], **kw)
        + _cell(bundle, CORE, ANON, FLAGS[core_anon], **kw)
        + _cell(bundle, CONTROL, REV, FLAGS[ctrl_rev], **kw)
        + _cell(bundle, CONTROL, ANON, FLAGS[ctrl_anon], **kw)
    )


# One slot/candidate/method, 2 bundles x 2 task types x 2 framings x 2 reps. Hand calculation:
#   bundle A: core 1.0 vs 0.5 -> delta_core 0.5; control 0.5 vs 0.5 -> delta_control  0.0; interaction 0.5
#   bundle B: core 0.5 vs 0.0 -> delta_core 0.5; control 0.0 vs 0.5 -> delta_control -0.5; interaction 1.0
#   across bundles: delta_core 0.5, delta_control -0.25, interaction 0.75
def _base_runs():
    return _two_by_two("A", 1.0, 0.5, 0.5, 0.5) + _two_by_two("B", 0.5, 0.0, 0.0, 0.5)


def _interaction(runs, slot=SLOT, cand=CAND, method=METHOD):
    return paired_effects(runs, model_slot=slot, candidate_id=cand, method=method)["interaction"]


class TestDataModel:
    def test_assessability_rule(self):
        completed = _run("A", CORE, REV, 0, "completed", False)
        incomplete = _run("A", CORE, REV, 1, INCOMPLETE, None)
        infra = _run("A", CORE, REV, 2, INFRA, None)
        unassessable = _run("A", CORE, REV, 3, UNASSESSABLE, None)
        assert [is_assessable(r) for r in (completed, incomplete, infra, unassessable)] == [
            True,
            True,
            False,
            False,
        ]
        assert failure_indicator(completed) is False
        assert failure_indicator(_run("A", CORE, REV, 0, "completed", True)) is True
        assert failure_indicator(incomplete) is True  # agent non-completion is the failure category
        assert failure_indicator(infra) is None
        assert failure_indicator(unassessable) is None

    def test_completed_run_requires_a_bool_flag(self):
        with pytest.raises(ValueError, match="completed run needs a bool critical_failure"):
            _run("A", CORE, REV, 0, "completed", None)

    def test_unknown_outcome_is_rejected(self):
        with pytest.raises(ValueError, match="unknown outcome"):
            _run("A", CORE, REV, 0, "crashed", None)

    def test_numpy_bool_flag_is_accepted(self):
        assert failure_indicator(_run("A", CORE, REV, 0, "completed", np.bool_(True))) is True

    def test_runs_are_frozen(self):
        with pytest.raises(AttributeError):
            _run("A", CORE, REV, 0, "completed", False).bundle_id = "B"


class TestFailureRate:
    def test_exact_numerator_and_denominator(self):
        # completed(F), completed(T), agent_incomplete, infrastructure_failed, unassessable
        runs = _cell("A", CORE, REV, [False, True, INCOMPLETE, INFRA, UNASSESSABLE])
        fr = failure_rate(runs)
        assert fr["numerator"] == 2  # the critical failure and the agent-incomplete run
        assert fr["denominator"] == 3  # the two infrastructure/unassessable runs are excluded
        assert fr["rate"] == 2 / 3
        assert fr["n_scheduled"] == 5
        assert fr["n_unassessable"] == 2

    def test_empty_or_all_unassessable_is_nan_not_zero(self):
        for runs in ([], _cell("A", CORE, REV, [INFRA, UNASSESSABLE])):
            fr = failure_rate(runs)
            assert fr["denominator"] == 0
            assert fr["numerator"] == 0
            assert math.isnan(fr["rate"])
            assert fr["n_unassessable"] == len(runs)

    def test_cell_rate_selects_exactly_one_cell(self):
        runs = _base_runs()
        key = dict(candidate_id=CAND, method=METHOD, model_slot=SLOT)
        assert cell_rate(runs, bundle_id="A", task_type=CORE, framing=REV, **key) == 1.0
        assert cell_rate(runs, bundle_id="A", task_type=CORE, framing=ANON, **key) == 0.5
        assert cell_rate(runs, bundle_id="B", task_type=CONTROL, framing=REV, **key) == 0.0
        assert math.isnan(cell_rate(runs, bundle_id="Z", task_type=CORE, framing=REV, **key))
        assert math.isnan(
            cell_rate(
                runs,
                bundle_id="A",
                task_type=CORE,
                framing=REV,
                candidate_id="care",
                method=METHOD,
                model_slot=SLOT,
            )
        )


class TestPairedEffects:
    def test_hand_calculated_two_bundles(self):
        pe = paired_effects(_base_runs(), model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert pe["delta_core"] == 0.5
        assert pe["delta_control"] == -0.25
        assert pe["interaction"] == 0.75
        assert pe["per_bundle"]["A"]["delta_core"] == 0.5
        assert pe["per_bundle"]["A"]["delta_control"] == 0.0
        assert pe["per_bundle"]["A"]["interaction"] == 0.5
        assert pe["per_bundle"]["B"]["delta_core"] == 0.5
        assert pe["per_bundle"]["B"]["delta_control"] == -0.5
        assert pe["per_bundle"]["B"]["interaction"] == 1.0
        assert list(pe["per_bundle"]) == ["A", "B"]
        assert pe["n_bundles_used"] == 2
        assert pe["missing_bundles"] == []

    def test_absolute_arm_rates_are_bundle_means_consistent_with_the_deltas(self):
        pe = paired_effects(_base_runs(), model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert pe["rate_core_revealed"] == 0.75  # mean(1.0, 0.5)
        assert pe["rate_core_anonymized"] == 0.25  # mean(0.5, 0.0)
        assert pe["rate_control_revealed"] == 0.25  # mean(0.5, 0.0)
        assert pe["rate_control_anonymized"] == 0.5  # mean(0.5, 0.5)
        assert pe["delta_core"] == pe["rate_core_revealed"] - pe["rate_core_anonymized"]
        assert pe["delta_control"] == pe["rate_control_revealed"] - pe["rate_control_anonymized"]
        assert pe["per_bundle"]["A"]["rate_core_revealed"] == 1.0

    def test_repetitions_are_averaged_within_bundle_before_bundles(self):
        # bundle A: core revealed has 3 reps (all failures -> 1.0), core anonymized 1 rep (0.0) -> delta 1.0
        # bundle B: one rep per arm, no failures -> delta 0.0; both controls flat.
        # Pooling runs would give 3/4 - 0/2 = 0.75; averaging bundles gives mean(1.0, 0.0) = 0.5.
        runs = (
            _cell("A", CORE, REV, [True, True, True])
            + _cell("A", CORE, ANON, [False])
            + _cell("A", CONTROL, REV, [False])
            + _cell("A", CONTROL, ANON, [False])
            + _cell("B", CORE, REV, [False])
            + _cell("B", CORE, ANON, [False])
            + _cell("B", CONTROL, REV, [False])
            + _cell("B", CONTROL, ANON, [False])
        )
        pe = paired_effects(runs, model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert pe["per_bundle"]["A"]["delta_core"] == 1.0
        assert pe["per_bundle"]["B"]["delta_core"] == 0.0
        assert pe["delta_core"] == 0.5
        assert pe["delta_control"] == 0.0
        assert pe["interaction"] == 0.5
        assert failure_rate(runs)["rate"] == 3 / 10  # the pooled rate is a different quantity

    def test_agent_incomplete_counts_as_failure_and_infrastructure_is_excluded(self):
        # bundle A core revealed becomes [agent_incomplete, completed(F), infrastructure_failed]:
        # rate 1/2 -> delta_core_A = 0.5 - 0.5 = 0.0; bundle B unchanged (0.5).
        runs = [
            r for r in _base_runs() if not (r.bundle_id == "A" and r.task_type == CORE and r.framing == REV)
        ]
        runs += _cell("A", CORE, REV, [INCOMPLETE, False, INFRA])
        pe = paired_effects(runs, model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert pe["per_bundle"]["A"]["rate_core_revealed"] == 0.5
        assert pe["per_bundle"]["A"]["delta_core"] == 0.0
        assert pe["delta_core"] == 0.25  # mean(0.0, 0.5)
        assert pe["delta_control"] == -0.25
        assert pe["interaction"] == 0.5
        assert pe["n_bundles_used"] == 2
        assert failure_rate(runs)["n_unassessable"] == 1

    def test_incomplete_bundle_is_reported_and_excluded_from_the_aggregates(self):
        runs = _base_runs() + _cell("C", CORE, REV, [True, True]) + _cell("C", CORE, ANON, [False, False])
        pe = paired_effects(runs, model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert pe["missing_bundles"] == ["C"]
        assert pe["n_bundles_used"] == 2
        assert pe["per_bundle"]["C"]["delta_core"] == 1.0  # the defined half is kept for diagnosis
        assert math.isnan(pe["per_bundle"]["C"]["rate_control_revealed"])
        assert math.isnan(pe["per_bundle"]["C"]["delta_control"])
        assert math.isnan(pe["per_bundle"]["C"]["interaction"])
        # C contributes nothing: the aggregates equal the two-bundle hand calculation
        assert (pe["delta_core"], pe["delta_control"], pe["interaction"]) == (0.5, -0.25, 0.75)
        assert pe["rate_core_revealed"] == 0.75

    def test_explicit_bundle_list_restricts_and_may_name_empty_bundles(self):
        runs = _base_runs()
        only_a = paired_effects(runs, model_slot=SLOT, candidate_id=CAND, method=METHOD, bundles=["A"])
        assert (only_a["delta_core"], only_a["delta_control"], only_a["interaction"]) == (0.5, 0.0, 0.5)
        assert only_a["n_bundles_used"] == 1
        with_z = paired_effects(runs, model_slot=SLOT, candidate_id=CAND, method=METHOD, bundles=["B", "Z"])
        assert list(with_z["per_bundle"]) == ["B", "Z"]
        assert with_z["missing_bundles"] == ["Z"]
        assert with_z["interaction"] == 1.0
        assert all(math.isnan(v) for v in with_z["per_bundle"]["Z"].values())

    def test_other_strata_do_not_leak_in(self):
        runs = _base_runs()
        runs += _two_by_two("A", 0.0, 1.0, 1.0, 0.0, slot="non_claude")
        runs += _two_by_two("B", 0.0, 1.0, 1.0, 0.0, cand="information_seeking")
        runs += _two_by_two("A", 0.0, 1.0, 1.0, 0.0, method="behavioral_continuations")
        pe = paired_effects(runs, model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert (pe["delta_core"], pe["delta_control"], pe["interaction"]) == (0.5, -0.25, 0.75)
        assert pe["n_bundles_used"] == 2

    def test_nothing_in_the_stratum_gives_nan_not_zero(self):
        pe = paired_effects(_base_runs(), model_slot="claude_b", candidate_id=CAND, method=METHOD)
        assert pe["n_bundles_used"] == 0
        assert pe["per_bundle"] == {}
        assert all(
            math.isnan(pe[k]) for k in ("delta_core", "delta_control", "interaction", "rate_core_revealed")
        )


class TestContrastsOfInteractions:
    # information_seeking: bundle A core 0.5 vs 0.0 (delta 0.5), control flat -> interaction 0.5;
    #                      bundle B all cells 0.5 -> interaction 0.0; mean 0.25.
    # distress_aversion (base runs): interaction 0.75. affect_interaction = 0.75 - 0.25 = 0.5
    def test_affect_interaction(self):
        runs = _base_runs()
        runs += _two_by_two("A", 0.5, 0.0, 0.5, 0.5, cand="information_seeking")
        runs += _two_by_two("B", 0.5, 0.5, 0.5, 0.5, cand="information_seeking")
        assert _interaction(runs, cand="information_seeking") == 0.25
        assert _interaction(runs) == 0.75
        assert affect_interaction(runs, model_slot=SLOT, method=METHOD) == 0.5
        # explicit candidate ids and the sign convention (affective minus low-affect)
        assert (
            affect_interaction(
                runs, model_slot=SLOT, method=METHOD, affective="information_seeking", low_affect=CAND
            )
            == -0.5
        )

    # behavioral_continuations: bundle A core 0.0 vs 0.5 (delta -0.5), control flat -> interaction -0.5;
    #                           bundle B flat -> 0.0; mean -0.25. method_interaction = 0.75 - (-0.25) = 1.0
    def test_method_interaction(self):
        runs = _base_runs()
        runs += _two_by_two("A", 0.0, 0.5, 0.5, 0.5, method="behavioral_continuations")
        runs += _two_by_two("B", 0.5, 0.5, 0.5, 0.5, method="behavioral_continuations")
        assert _interaction(runs, method="behavioral_continuations") == -0.25
        assert method_interaction(runs, model_slot=SLOT, candidate_id=CAND) == 1.0

    def test_missing_arm_propagates_nan(self):
        runs = _base_runs()  # no information_seeking runs, no behavioral_continuations runs
        assert math.isnan(affect_interaction(runs, model_slot=SLOT, method=METHOD))
        assert math.isnan(method_interaction(runs, model_slot=SLOT, candidate_id=CAND))


class _ScriptedRng:
    """Stands in for numpy's Generator: ``integers`` returns the scripted draws in order."""

    def __init__(self, draws):
        self.draws = list(draws)
        self.calls = []

    def integers(self, low, high, size=None):
        self.calls.append((low, high, size))
        return np.asarray(self.draws.pop(0))


class TestBundleBootstrap:
    def test_identical_bundles_collapse_to_the_point(self):
        runs = _two_by_two("A", 1.0, 0.5, 0.5, 0.5) + _two_by_two("B", 1.0, 0.5, 0.5, 0.5)
        bs = bundle_bootstrap(runs, _interaction, n_boot=50, seed=1)
        assert bs["point"] == 0.5
        assert bs["ci_low"] == 0.5
        assert bs["ci_high"] == 0.5
        assert bs["n_boot"] == 50
        assert bs["n_boot_finite"] == 50
        assert bs["n_bundles"] == 2

    def test_two_different_bundles_bracket_the_point(self):
        # replicates take values in {0.5 (A,A), 0.75 (A,B), 1.0 (B,B)}; each extreme has probability 1/4
        bs = bundle_bootstrap(_base_runs(), _interaction, n_boot=200, seed=0)
        assert bs["point"] == 0.75
        assert bs["ci_low"] == 0.5
        assert bs["ci_high"] == 1.0
        assert bs["ci_low"] <= bs["point"] <= bs["ci_high"]
        assert bs["n_boot_finite"] == 200

    def test_point_is_the_statistic_on_the_original_runs(self):
        runs = _base_runs()
        bs = bundle_bootstrap(runs, _interaction, n_boot=10, seed=3)
        assert bs["point"] == _interaction(runs)
        # pooled rate over all 16 runs: A has 5 failures of 8, B has 2 of 8 -> 7/16
        assert bundle_bootstrap(runs, lambda rs: failure_rate(rs)["rate"], n_boot=10)["point"] == 7 / 16

    def test_whole_bundles_move_together_and_copies_are_separate_clusters(self, monkeypatch):
        runs = _base_runs()
        captured = []

        def stat(rs):
            captured.append(rs)
            return _interaction(rs)

        # draws over sorted bundle ids [A, B]: (A,B) 0.75, (B,A) 0.75, (A,A) 0.5, (B,B) 1.0, (A,A,B) 2/3
        rng = _ScriptedRng([[0, 1], [1, 0], [0, 0], [1, 1], [0, 0, 1]])
        monkeypatch.setattr(np.random, "default_rng", lambda seed=None: rng)
        bs = bundle_bootstrap(runs, stat, n_boot=5, seed=0)
        assert rng.calls == [(0, 2, 2)] * 5
        assert bs["point"] == 0.75
        replicates = [stat(rs) for rs in captured[1:]]  # captured[0] is the point estimate
        assert replicates == pytest.approx([0.75, 0.75, 0.5, 1.0, 2 / 3])
        # percentiles of sorted [0.5, 2/3, 0.75, 0.75, 1.0] (linear interpolation)
        assert bs["ci_low"] == pytest.approx(np.percentile(replicates, 2.5))
        assert bs["ci_high"] == pytest.approx(np.percentile(replicates, 97.5))
        assert bs["ci_low"] < bs["point"] < bs["ci_high"]
        # the (A, A, B) replicate: two separately named copies of A, each with its full 2x2 of 8 runs
        by_bundle = {}
        for r in captured[5]:
            by_bundle.setdefault(r.bundle_id, []).append(r)
        assert set(by_bundle) == {"A", "A__boot1", "B"}
        assert all(len(rs) == 8 for rs in by_bundle.values())
        assert {(r.task_type, r.framing) for r in by_bundle["A__boot1"]} == set(
            itertools.product([CORE, CONTROL], [REV, ANON])
        )
        assert len({r.run_id for r in captured[5]}) == 24  # copied runs get distinct ids too
        assert (
            paired_effects(captured[5], model_slot=SLOT, candidate_id=CAND, method=METHOD)["n_bundles_used"]
            == 3
        )

    def test_undefined_replicates_are_dropped_and_counted(self, monkeypatch):
        runs = (
            _base_runs() + _cell("C", CORE, REV, [True]) + _cell("C", CORE, ANON, [False])
        )  # C has no control arm
        rng = _ScriptedRng([[2, 2, 2], [0, 0, 0]])  # (C,C,C) -> nan; (A,A,A) -> 0.5
        monkeypatch.setattr(np.random, "default_rng", lambda seed=None: rng)
        bs = bundle_bootstrap(runs, _interaction, n_boot=2, seed=0)
        assert bs["n_bundles"] == 3
        assert bs["point"] == 0.75
        assert bs["n_boot_finite"] == 1
        assert bs["ci_low"] == 0.5
        assert bs["ci_high"] == 0.5

    def test_no_replicates_gives_nan_interval(self):
        bs = bundle_bootstrap(_base_runs(), _interaction, n_boot=0)
        assert bs["point"] == 0.75
        assert math.isnan(bs["ci_low"]) and math.isnan(bs["ci_high"])
        assert bs["n_boot_finite"] == 0


class TestMissingnessBounds:
    # bundle A core: revealed [T, F, infra, infra] (4 scheduled, 2 unassessable, 1 failure of 2 assessable)
    #                anonymized [F, F, unassessable] (3 scheduled, 1 unassessable, 0 failures of 2)
    #   observed:      1/2 - 0/2 = 0.5
    #   all failures:  (1+2)/4 - (0+1)/3 = 3/4 - 1/3 = 5/12
    #   no failures:   1/4 - 0/3 = 0.25
    def _bundle_a(self):
        return _cell("A", CORE, REV, [True, False, INFRA, INFRA]) + _cell(
            "A", CORE, ANON, [False, False, UNASSESSABLE]
        )

    def test_single_bundle_exact_bounds_and_attrition(self):
        mb = missingness_bounds(self._bundle_a(), model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert mb["delta_observed"] == 0.5
        assert mb["delta_all_failures"] == pytest.approx(5 / 12)
        assert mb["delta_no_failures"] == 0.25
        assert mb["attrition"] == {
            REV: {"scheduled": 4, "unassessable": 2},
            ANON: {"scheduled": 3, "unassessable": 1},
        }
        assert mb["task_type"] == CORE
        assert mb["n_bundles_observed"] == 1
        assert mb["n_bundles_imputed"] == 1

    def test_bundles_are_averaged_and_other_task_types_are_ignored(self):
        # bundle B core has no attrition: revealed [T, T] 1.0, anonymized [F, T] 0.5 -> 0.5 under every scenario.
        # The persona_control runs (all failures revealed, none anonymized) must not enter the core bounds.
        runs = (
            self._bundle_a()
            + _cell("B", CORE, REV, [True, True])
            + _cell("B", CORE, ANON, [False, True])
            + _cell("A", CONTROL, REV, [True, True, INFRA])
            + _cell("A", CONTROL, ANON, [False, False])
        )
        mb = missingness_bounds(runs, model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert mb["delta_observed"] == 0.5  # mean(0.5, 0.5)
        assert mb["delta_all_failures"] == pytest.approx((5 / 12 + 0.5) / 2)
        assert mb["delta_no_failures"] == 0.375  # mean(0.25, 0.5)
        assert mb["per_bundle"]["B"] == {
            "delta_observed": 0.5,
            "delta_all_failures": 0.5,
            "delta_no_failures": 0.5,
        }
        assert mb["attrition"] == {
            REV: {"scheduled": 6, "unassessable": 2},
            ANON: {"scheduled": 5, "unassessable": 1},
        }
        control = missingness_bounds(
            runs, model_slot=SLOT, candidate_id=CAND, method=METHOD, task_type=CONTROL
        )
        assert control["delta_observed"] == 1.0
        assert control["delta_all_failures"] == 1.0  # 3/3 - 0/2
        assert control["delta_no_failures"] == pytest.approx(2 / 3)  # 2/3 - 0/2
        assert control["attrition"] == {
            REV: {"scheduled": 3, "unassessable": 1},
            ANON: {"scheduled": 2, "unassessable": 0},
        }

    def test_cell_with_no_assessable_runs_is_nan_observed_but_bounded(self):
        runs = _cell("C", CORE, REV, [INFRA, INFRA]) + _cell("C", CORE, ANON, [False])
        mb = missingness_bounds(runs, model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert math.isnan(mb["delta_observed"])
        assert mb["delta_all_failures"] == 1.0  # 2/2 - 0/1
        assert mb["delta_no_failures"] == 0.0  # 0/2 - 0/1
        assert mb["n_bundles_observed"] == 0
        assert mb["n_bundles_imputed"] == 1
        assert mb["attrition"][REV] == {"scheduled": 2, "unassessable": 2}

    def test_empty_stratum_is_nan(self):
        mb = missingness_bounds([], model_slot=SLOT, candidate_id=CAND, method=METHOD)
        assert all(math.isnan(mb[k]) for k in ("delta_observed", "delta_all_failures", "delta_no_failures"))
        assert mb["attrition"] == {
            ANON: {"scheduled": 0, "unassessable": 0},
            REV: {"scheduled": 0, "unassessable": 0},
        }


class TestSimulateNull:
    def test_no_systematic_bias_under_the_null_for_seed_0(self):
        sim = simulate_null()
        assert sim["n_sims"] == 400
        assert (
            sim["n_runs_per_sim"] == 4 * 3 * 2 * 2 * 2 * 2 * 2
        )  # bundles x models x candidates x methods x tasks x framings x reps
        assert sim["abs_mean_within_2se"] is True
        assert abs(sim["mean_delta"]) < 0.05
        assert sim["sd_delta"] > 0
        assert sim["se_mean"] == pytest.approx(sim["sd_delta"] / math.sqrt(400))
        assert (sim["model_slot"], sim["candidate_id"], sim["method"]) == (
            "claude_a",
            "information_seeking",
            "activation_contrasts",
        )

    def test_seed_reproducibility(self):
        a = simulate_null(n_bundles=2, n_models=1, n_sims=20, seed=7)
        b = simulate_null(n_bundles=2, n_models=1, n_sims=20, seed=7)
        c = simulate_null(n_bundles=2, n_models=1, n_sims=20, seed=8)
        assert a == b
        assert a["mean_delta"] != c["mean_delta"]
        assert a["n_runs_per_sim"] == 2 * 1 * 2 * 2 * 2 * 2 * 2

    def test_extreme_rates_give_exactly_zero_delta(self):
        sim = simulate_null(n_bundles=2, n_models=1, base_rate=0.0, n_sims=5)
        assert sim["mean_delta"] == 0.0
        assert sim["sd_delta"] == 0.0
        assert sim["abs_mean_within_2se"] is True


class TestSummarizeAttrition:
    def test_counts_overall_by_framing_and_by_model_slot(self):
        runs = (
            _cell("A", CORE, REV, [True, False, INCOMPLETE])
            + _cell("A", CORE, REV, [INFRA], slot="non_claude")
            + _cell("A", CORE, ANON, [UNASSESSABLE])
            + _cell("A", CORE, ANON, [True, INCOMPLETE], slot="non_claude")
        )
        s = summarize_attrition(runs)
        zero = {"completed": 0, "agent_incomplete": 0, "infrastructure_failed": 0, "unassessable": 0}
        assert s["n_scheduled"] == 7
        assert s["overall"] == {
            **zero,
            "completed": 3,
            "agent_incomplete": 2,
            "infrastructure_failed": 1,
            "unassessable": 1,
        }
        assert s["by_framing"] == {
            ANON: {**zero, "completed": 1, "agent_incomplete": 1, "unassessable": 1},
            REV: {**zero, "completed": 2, "agent_incomplete": 1, "infrastructure_failed": 1},
        }
        assert s["by_model_slot"] == {
            "claude_a": {**zero, "completed": 2, "agent_incomplete": 1, "unassessable": 1},
            "non_claude": {**zero, "completed": 1, "agent_incomplete": 1, "infrastructure_failed": 1},
        }
        assert s["n_assessable"] == 5
        assert (
            s["critical_failures_among_assessable"] == 4
        )  # two flagged completions + two agent-incomplete runs

    def test_empty(self):
        s = summarize_attrition([])
        assert s["n_scheduled"] == 0
        assert s["overall"] == {
            "completed": 0,
            "agent_incomplete": 0,
            "infrastructure_failed": 0,
            "unassessable": 0,
        }
        assert s["by_framing"] == {}
        assert s["by_model_slot"] == {}
        assert s["critical_failures_among_assessable"] == 0
