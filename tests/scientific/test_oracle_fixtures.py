"""Scientific checks: the reference oracle recovers planted fixture structure, and key statistics agree
with independent implementations (scipy AUC, explicit-loop contrast vectors, polyfit slopes)."""

from __future__ import annotations

import numpy as np
import pytest

from npbench.bundles import COND, Bundle
from npbench.fixtures_tier2 import make_fixture_bundle
from npbench.reference.independent import auc_scipy, contrast_vector_loop, slope_polyfit
from npbench.reference.oracle import compute_reference
from npbench.splits import check_split_integrity


@pytest.fixture(scope="module")
def refs():
    out = {}
    for variant in [
        "positive",
        "null",
        "sign_reversed",
        "confounded",
        "missing_rows",
        "action_label_swap",
        "inconclusive",
    ]:
        b, truth = make_fixture_bundle(variant, "distress_aversion", f"fx_{variant}", seed=7)
        out[variant] = {
            "bundle": b,
            "truth": truth,
            "act_core": compute_reference(
                b, "activation_contrasts", "core_mode_transfer", seed=1, n_boot=200
            ),
            "beh_core": compute_reference(
                b, "behavioral_continuations", "core_mode_transfer", seed=1, n_boot=200
            ),
            "act_ctrl": compute_reference(b, "activation_contrasts", "persona_control", seed=1, n_boot=200),
        }
    return out


def _get(ref, rid):
    return next(r for r in ref["reference"] if r["result_id"] == rid)


def test_registry_counts_match_between_task_types(refs):
    r = refs["positive"]
    assert len(r["act_core"]["registry"]) == len(r["act_ctrl"]["registry"]) == 30
    assert len(r["beh_core"]["registry"]) == 30
    assert {x["result_id"] for x in r["act_core"]["reference"]} == {
        x["result_id"] for x in r["act_core"]["registry"]
    }


def test_positive_fixture_recovered(refs):
    a = refs["positive"]["act_core"]
    assert _get(a, "A3_auc_test[C-A]")["estimate"] > 0.65
    assert _get(a, "A3_auc_test[C-A]")["directionality"] == "supportive"
    assert _get(a, "A5_rate_slope[C-A]")["estimate"] > 0.1
    assert _get(a, "A6_lik_slope[C-A]")["estimate"] > 1.0
    assert _get(a, "A8_effect_minus_random[C-A]")["directionality"] == "supportive"
    assert _get(a, "A0_identity_residual")["estimate"] < 1e-9
    assert abs(_get(a, "A9_applied_vs_requested[C-A]")["estimate"] - 1.0) < 0.02
    b = refs["positive"]["beh_core"]
    assert _get(b, "B1_rate_diff_test[C-A]")["estimate"] > 0.2
    assert _get(b, "B1_rate_diff_test[C-A]")["directionality"] == "supportive"
    assert _get(b, "B3_lik_diff_test[C-A]")["estimate"] > 1.0


def test_null_fixture_is_null_on_primary_readouts(refs):
    a = refs["null"]["act_core"]
    assert abs(_get(a, "A3_auc_test[C-A]")["estimate"] - 0.5) < 0.1
    assert _get(a, "A3_auc_test[C-A]")["directionality"] in ("null", "inconclusive")
    assert abs(_get(a, "A5_rate_slope[C-A]")["estimate"]) < 0.06
    assert _get(a, "A8_effect_minus_random[C-A]")["directionality"] in ("null", "inconclusive")
    b = refs["null"]["beh_core"]
    assert abs(_get(b, "B1_rate_diff_test[C-A]")["estimate"]) < 0.15
    # the persona-control task is still positive on the null core fixture (planted polarity effect)
    c = refs["null"]["act_ctrl"]
    assert _get(c, "A3_auc_test[P-Q@A]")["estimate"] > 0.75
    assert _get(c, "A7_rating_slope[P-Q@A]")["directionality"] == "supportive"


def test_sign_reversed_fixture_flips_causal_and_behavioural_signs_but_not_auc(refs):
    pos = refs["positive"]
    rev = refs["sign_reversed"]
    assert _get(rev["act_core"], "A3_auc_test[C-A]")["estimate"] > 0.65  # AUC is sign-agnostic
    assert _get(rev["act_core"], "A6_lik_slope[C-A]")["estimate"] < -1.0
    assert _get(rev["act_core"], "A6_lik_slope[C-A]")["directionality"] == "contradictory"
    assert _get(rev["beh_core"], "B1_rate_diff_test[C-A]")["estimate"] < -0.2
    # the raw vectors point in opposite directions
    v_pos = np.asarray(pos["act_core"]["extras"]["units"]["C-A"])
    v_rev = np.asarray(rev["act_core"]["extras"]["units"]["C-A"])
    assert float(v_pos @ v_rev) < -0.6


def test_confounded_fixture_separates_on_construction_but_not_heldout(refs):
    a = refs["confounded"]["act_core"]
    assert a["extras"]["construction_auc"]["C-A"] > 0.7
    assert abs(_get(a, "A3_auc_test[C-A]")["estimate"] - 0.5) < 0.12
    assert _get(a, "A3_auc_test[C-A]")["directionality"] in ("null", "inconclusive")
    assert _get(a, "A5_rate_slope[C-A]")["directionality"] in ("null", "inconclusive")


def test_inconclusive_fixture_is_not_called_supportive_on_auc(refs):
    a = refs["inconclusive"]["act_core"]
    assert _get(a, "A3_auc_test[C-A]")["directionality"] in ("null", "inconclusive")


def test_missing_rows_reported_not_zeroed(refs):
    a = refs["missing_rows"]["act_core"]
    assert a["extras"]["n_missing_activation_rows"] == 5
    assert _get(a, "A1_norm[C-A]")["n_scenario_groups"] < 72
    assert _get(a, "A0_identity_residual")["applicable"] is True
    assert _get(a, "A0_identity_residual")["estimate"] < 1e-9
    # no NaN leaks into any estimate
    for r in a["reference"]:
        if r["estimate"] is not None:
            assert r["estimate"] == r["estimate"]


def test_action_label_swap_contradicts_activation_sign(refs):
    a = refs["action_label_swap"]["act_core"]
    b = refs["action_label_swap"]["beh_core"]
    assert _get(a, "A6_lik_slope[C-A]")["estimate"] > 1.0
    assert _get(b, "B1_rate_diff_test[C-A]")["estimate"] < -0.2
    assert _get(b, "B3_lik_diff_test[C-A]")["estimate"] > 1.0  # likelihoods were not swapped, only codes


def test_leakage_fixture_rejected_by_split_integrity():
    b, _ = make_fixture_bundle("leakage", "distress_aversion", "fx_leakage", seed=3)
    assert not check_split_integrity(b.stimuli).ok


def test_oracle_vector_matches_independent_loop(refs):
    b: Bundle = refs["positive"]["bundle"]
    a = refs["positive"]["act_core"]
    H = b.activations[b.primary_layer()]
    h_c, h_a = {}, {}
    for i, s in enumerate(b.stimuli):
        if s.split.value != "construction":
            continue
        key = (s.underlying_context_id, s.candidate_polarity.value)
        if COND[s.mode_intended.value] == "C":
            h_c[key] = H[i]
        elif COND[s.mode_intended.value] == "A":
            h_a[key] = H[i]
    v = contrast_vector_loop(h_c, h_a)
    assert np.allclose(v, np.asarray(a["extras"]["vectors"]["C-A"]), atol=1e-9)


def test_oracle_auc_matches_scipy(refs):
    b: Bundle = refs["positive"]["bundle"]
    a = refs["positive"]["act_core"]
    H = b.activations[b.primary_layer()]
    u = np.asarray(a["extras"]["units"]["C-A"])
    pc = [
        H[i] @ u
        for i, s in enumerate(b.stimuli)
        if s.split.value == "test" and COND[s.mode_intended.value] == "C"
    ]
    pa = [
        H[i] @ u
        for i, s in enumerate(b.stimuli)
        if s.split.value == "test" and COND[s.mode_intended.value] == "A"
    ]
    assert abs(auc_scipy(np.asarray(pc), np.asarray(pa)) - _get(a, "A3_auc_test[C-A]")["estimate"]) < 1e-9


def test_oracle_slope_matches_polyfit(refs):
    b: Bundle = refs["positive"]["bundle"]
    a = refs["positive"]["act_core"]
    rows = [r for r in b.interventions if r["direction_id"] == "C-A"]
    groups = sorted({r["scenario_group"] for r in rows})
    slopes = []
    for g in groups:
        sub = [r for r in rows if r["scenario_group"] == g]
        slopes.append(
            slope_polyfit(np.asarray([r["alpha"] for r in sub]), np.asarray([r["lik_contrast"] for r in sub]))
        )
    assert abs(float(np.mean(slopes)) - _get(a, "A6_lik_slope[C-A]")["estimate"]) < 1e-9
