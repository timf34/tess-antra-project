"""Hand-calculated checks for npbench.vectors (contrast vectors, scales, controls, readouts)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from npbench.vectors import (
    CONTRASTS,
    MODES,
    ContrastVector,
    check_raw_identity,
    cosine,
    intervention_displacement,
    mann_whitney_auc,
    mode_by_polarity_interaction,
    paired_mode_contrasts,
    per_dimension_standardizer,
    projection_discrimination,
    projection_scale,
    random_directions,
    unit_vector,
    within_mode_polarity_contrast,
)

R, S, E = "roleplay", "simulation", "enactment"


def _acts(table):
    """Build the (group_id, mode) -> vector dict from a nested {group: {mode: list}} table."""
    return {(g, m): np.array(v, dtype=np.float64) for g, modes in table.items() for m, v in modes.items()}


# Two matched groups, three dims. Hand calculation:
#   E-R: g1 [2, 2, -2], g2 [2, -1, -1]  -> mean [2, 0.5, -1.5]
#   E-S: g1 [3, 1, -1], g2 [3,  1, -2]  -> mean [3, 1.0, -1.5]
#   R-S: g1 [1, -1, 1], g2 [1,  2, -1]  -> mean [1, 0.5,  0.0]
TWO_GROUPS = {
    "g1": {R: [1, 0, 2], S: [0, 1, 1], E: [3, 2, 0]},
    "g2": {R: [2, 2, 2], S: [1, 0, 3], E: [4, 1, 1]},
}
V_ER = np.array([2.0, 0.5, -1.5])
V_ES = np.array([3.0, 1.0, -1.5])
V_RS = np.array([1.0, 0.5, 0.0])

# (context, mode, polarity) -> vector, a full 2x2 (enactment/roleplay x positive/neutral) for c1, c2.
#   c1: (E,pos - E,neu) = [3, 0]; (R,pos - R,neu) = [1, 0]; interaction [2, 0]
#   c2: (E,pos - E,neu) = [0, 3]; (R,pos - R,neu) = [0, 1]; interaction [0, 2]
#   c3 lacks the roleplay/neutral cell and must be excluded.
TWO_BY_TWO = {
    ("c1", E, "positive"): [4, 1],
    ("c1", E, "neutral"): [1, 1],
    ("c1", R, "positive"): [2, 0],
    ("c1", R, "neutral"): [1, 0],
    ("c2", E, "positive"): [0, 5],
    ("c2", E, "neutral"): [0, 2],
    ("c2", R, "positive"): [0, 1],
    ("c2", R, "neutral"): [0, 0],
    ("c3", E, "positive"): [7, 7],
    ("c3", E, "neutral"): [0, 0],
    ("c3", R, "positive"): [7, 7],
}


def _polarity_acts(table):
    return {k: np.array(v, dtype=np.float64) for k, v in table.items()}


class TestPairedModeContrasts:
    def test_hand_calculated_means(self):
        cv = paired_mode_contrasts(_acts(TWO_GROUPS))
        assert set(cv) == {"ER", "ES", "RS"}
        assert isinstance(cv["ER"], ContrastVector)
        np.testing.assert_array_equal(cv["ER"].vector, V_ER)
        np.testing.assert_array_equal(cv["ES"].vector, V_ES)
        np.testing.assert_array_equal(cv["RS"].vector, V_RS)
        for cid, c in cv.items():
            assert c.contrast_id == cid
            assert c.n_groups == 2
            assert c.group_ids == ["g1", "g2"]
            assert c.near_zero is False
        assert cv["ER"].norm == pytest.approx(math.sqrt(4 + 0.25 + 2.25))
        assert cv["ES"].norm == 3.5  # sqrt(9 + 1 + 2.25) = sqrt(12.25), exact
        assert cv["RS"].norm == pytest.approx(math.sqrt(1.25))

    def test_per_group_diffs_kept_for_resampling(self):
        cv = paired_mode_contrasts(_acts(TWO_GROUPS))
        np.testing.assert_array_equal(cv["ER"].per_group_diffs, [[2, 2, -2], [2, -1, -1]])
        np.testing.assert_array_equal(cv["RS"].per_group_diffs, [[1, -1, 1], [1, 2, -1]])
        np.testing.assert_array_equal(cv["ER"].per_group_diffs.mean(axis=0), cv["ER"].vector)

    def test_raw_identity_holds_on_identical_group_sets(self):
        np.testing.assert_array_equal(V_ER + V_RS, V_ES)  # the hand values satisfy it exactly
        cv = paired_mode_contrasts(_acts(TWO_GROUPS))
        ok, err = check_raw_identity(cv)
        assert ok is True
        assert err == pytest.approx(0.0, abs=1e-12)

    def test_group_missing_a_mode_is_excluded_only_where_needed(self):
        table = dict(TWO_GROUPS)
        table["g3"] = {R: [0, 0, 0], S: [1, 1, 1]}  # no enactment activation for g3
        cv = paired_mode_contrasts(_acts(table))
        assert cv["ER"].group_ids == ["g1", "g2"]
        assert cv["ES"].group_ids == ["g1", "g2"]
        assert cv["RS"].group_ids == ["g1", "g2", "g3"]
        assert cv["RS"].n_groups == 3
        np.testing.assert_array_equal(cv["ER"].vector, V_ER)
        # R-S for g3 is [-1, -1, -1]; mean over three groups = ([1,-1,1] + [1,2,-1] + [-1,-1,-1]) / 3
        np.testing.assert_allclose(cv["RS"].vector, [1 / 3, 0.0, -1 / 3])
        ok, err = check_raw_identity(cv)
        assert ok is False
        assert math.isnan(err)

    def test_no_matched_group_raises(self):
        acts = {("g1", R): np.zeros(3), ("g2", E): np.zeros(3)}  # never both modes in one group
        with pytest.raises(ValueError, match="no matched groups for contrast ER"):
            paired_mode_contrasts(acts)

    def test_non_vector_activation_raises(self):
        acts = _acts(TWO_GROUPS)
        acts[("g1", R)] = np.zeros((2, 3))
        with pytest.raises(ValueError, match="1-D"):
            paired_mode_contrasts(acts)

    def test_swapping_modes_reverses_sign_exactly(self):
        cv = paired_mode_contrasts(_acts(TWO_GROUPS), contrasts={"ER": (E, R), "RE": (R, E), "SE": (S, E)})
        np.testing.assert_array_equal(cv["RE"].vector, -cv["ER"].vector)
        np.testing.assert_array_equal(cv["RE"].vector, -V_ER)
        np.testing.assert_array_equal(cv["SE"].vector, -V_ES)
        assert cv["RE"].norm == cv["ER"].norm
        assert cv["RE"].group_ids == cv["ER"].group_ids
        np.testing.assert_array_equal(cv["RE"].per_group_diffs, -cv["ER"].per_group_diffs)

    def test_equal_weighting_each_group_contributes_one_over_n(self):
        n = 4
        table = {f"small{i}": {R: [0, 0, 0], E: [0, 1, 0]} for i in range(n - 1)}
        table["huge"] = {R: [0, 0, 0], E: [1e6, 0, 0]}
        acts = _acts(table)
        cv = paired_mode_contrasts(acts, contrasts={"ER": (E, R)})
        assert cv["ER"].n_groups == n
        diffs = np.stack([acts[(g, E)] - acts[(g, R)] for g in sorted(table)])
        np.testing.assert_array_equal(cv["ER"].vector, diffs.mean(axis=0))
        # Explicit 1/n weights: the huge group moves dim 0 by exactly 1e6/4 ...
        assert cv["ER"].vector[0] == 1e6 / n
        # ... and the small groups keep their full 3/4 share of dim 1, which a norm-weighted
        # average would have crushed towards zero.
        assert cv["ER"].vector[1] == (n - 1) / n
        assert cv["ER"].vector[2] == 0.0
        huge_diff = acts[("huge", E)] - acts[("huge", R)]
        np.testing.assert_array_equal(cv["ER"].vector - huge_diff / n, [0.0, 0.75, 0.0])

    def test_identical_activations_across_modes_give_flagged_zero_vector(self):
        table = {g: {R: v, S: v, E: v} for g, v in {"g1": [1, 2, 3], "g2": [-4, 0, 9]}.items()}
        cv = paired_mode_contrasts(_acts(table))
        for c in cv.values():
            np.testing.assert_array_equal(c.vector, np.zeros(3))
            assert c.norm == 0.0
            assert c.near_zero is True
            u, ok = unit_vector(c.vector)
            assert ok is False
            np.testing.assert_array_equal(u, np.zeros(3))
            assert not np.isnan(u).any()

    def test_cancelling_groups_are_flagged_not_normalized(self):
        table = {"g1": {R: [0, 0], E: [1, 0]}, "g2": {R: [0, 0], E: [-1, 0]}}
        cv = paired_mode_contrasts(_acts(table), contrasts={"ER": (E, R)})
        assert cv["ER"].norm == 0.0
        assert cv["ER"].near_zero is True  # norm 0 against a per-group difference scale of 1

    def test_to_record_fields(self):
        cv = paired_mode_contrasts(_acts(TWO_GROUPS))
        assert cv["ES"].to_record() == {
            "contrast_id": "ES",
            "norm": 3.5,
            "n_groups": 2,
            "near_zero": False,
            "dim": 3,
        }


class TestLabelsNeverEnterFeatures:
    """Loader contract: activations are keyed by (group_id, mode) only. A per-record ``label``
    (judge label, observed-mode label, polarity annotation) must never influence the arrays."""

    @staticmethod
    def _activations_from_records(records):
        # Several samples per (group, mode) are averaged; the ``label`` field is never read.
        sums: dict[tuple[str, str], np.ndarray] = {}
        counts: dict[tuple[str, str], int] = {}
        for r in records:
            key = (r["group_id"], r["mode"])
            sums[key] = sums.get(key, 0.0) + np.asarray(r["activation"], dtype=np.float64)
            counts[key] = counts.get(key, 0) + 1
        return {k: sums[k] / counts[k] for k in sums}

    @staticmethod
    def _records():
        rng = random.Random(0)
        records = []
        for g in ("g1", "g2", "g3"):
            for m in MODES:
                for sample in (0, 1):  # two samples per key, stored consecutively
                    records.append(
                        {
                            "group_id": g,
                            "mode": m,
                            "label": sample,
                            "activation": [rng.uniform(-1, 1) for _ in range(3)],
                        }
                    )
        return records

    def test_permuting_labels_leaves_vectors_unchanged(self):
        records = self._records()
        base = paired_mode_contrasts(self._activations_from_records(records))

        # labels permuted among the two records sharing each (group, mode) key ...
        within = [dict(r) for r in records]
        for i in range(0, len(within), 2):
            a, b = within[i], within[i + 1]
            assert (a["group_id"], a["mode"]) == (b["group_id"], b["mode"])
            a["label"], b["label"] = b["label"], a["label"]
        assert [r["label"] for r in within] != [r["label"] for r in records]

        # ... and across all records
        labels = [r["label"] for r in records]
        random.Random(1).shuffle(labels)
        assert labels != [r["label"] for r in records]
        across = [{**r, "label": lab} for r, lab in zip(records, labels, strict=True)]

        for variant in (within, across):
            cv = paired_mode_contrasts(self._activations_from_records(variant))
            for cid in CONTRASTS:
                np.testing.assert_array_equal(cv[cid].vector, base[cid].vector)
                assert cv[cid].group_ids == base[cid].group_ids

    def test_label_is_not_a_feature_dimension(self):
        cv = paired_mode_contrasts(self._activations_from_records(self._records()))
        assert all(c.vector.shape == (3,) for c in cv.values())


class TestUnitVectorAndScales:
    def test_unit_vector_normalizes(self):
        u, ok = unit_vector(np.array([3.0, 4.0]))
        assert ok is True
        np.testing.assert_allclose(u, [0.6, 0.8])
        assert np.linalg.norm(u) == pytest.approx(1.0)

    def test_unit_vector_zero_is_flagged_not_nan(self):
        u, ok = unit_vector(np.zeros(4))
        assert ok is False
        np.testing.assert_array_equal(u, np.zeros(4))
        assert np.isfinite(u).all()

    def test_unit_vector_threshold_is_relative_to_reference_scale(self):
        v = np.array([1e-7, 0.0])
        assert unit_vector(v)[1] is False  # default scale 1.0: 1e-7 <= 1e-6
        assert unit_vector(v, reference_scale=1.0)[1] is False
        assert unit_vector(v, reference_scale=1e-3)[1] is True  # 1e-7 > 1e-6 * 1e-3

    def test_projection_scale_is_population_std_of_projections(self):
        H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        e1 = np.array([1.0, 0.0])
        # projections onto e1: [1, 0, 1, 0] -> mean 0.5, deviations +-0.5 -> std 0.5
        assert projection_scale(H, e1) == 0.5
        u = np.array([1.0, 1.0]) / math.sqrt(2)
        # projections [1, 1, 2, 0]/sqrt2 -> deviations [0, 0, 1, -1]/sqrt2 -> variance 0.25 -> std 0.5
        assert projection_scale(H, u) == pytest.approx(0.5)
        assert projection_scale(H, u) == np.std(H @ u, ddof=0)
        assert projection_scale(H, e1) != np.std(H @ e1, ddof=1)  # population, not sample, std

    def test_projection_scale_requires_matrix(self):
        with pytest.raises(ValueError):
            projection_scale(np.array([1.0, 2.0]), np.array([1.0, 0.0]))

    def test_intervention_displacement_norm_and_sign(self):
        u = np.array([0.6, 0.8])  # unit
        d = intervention_displacement(2.0, 1.5, u)
        np.testing.assert_allclose(d, [1.8, 2.4])
        assert np.linalg.norm(d) == pytest.approx(2.0 * 1.5)
        d_neg = intervention_displacement(-2.0, 1.5, u)
        np.testing.assert_array_equal(d_neg, -d)
        assert np.linalg.norm(d_neg) == pytest.approx(abs(-2.0) * 1.5)
        np.testing.assert_array_equal(intervention_displacement(0.0, 1.5, u), np.zeros(2))


class TestRandomDirections:
    def test_rows_have_requested_norm(self):
        rd = random_directions(dim=5, n=7, seed=0, norm=2.5)
        assert rd.shape == (7, 5)
        np.testing.assert_allclose(np.linalg.norm(rd, axis=1), 2.5, rtol=0, atol=1e-9)

    def test_seed_determinism(self):
        a = random_directions(6, 4, seed=123, norm=1.0)
        b = random_directions(6, 4, seed=123, norm=1.0)
        c = random_directions(6, 4, seed=124, norm=1.0)
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a, c)
        assert not np.allclose(a[0], a[1])  # distinct draws, not one row repeated

    def test_norm_only_rescales(self):
        a = random_directions(6, 4, seed=5, norm=1.0)
        b = random_directions(6, 4, seed=5, norm=3.0)
        np.testing.assert_allclose(b, 3.0 * a)


class TestPolarityContrasts:
    def test_within_mode_polarity_contrast_hand_values(self):
        acts = _polarity_acts(
            {
                ("c1", R, "positive"): [1, 2],
                ("c1", R, "negative"): [0, 1],  # diff [1, 1]
                ("c2", R, "positive"): [3, 0],
                ("c2", R, "negative"): [1, 2],  # diff [2, -2]
                ("c3", R, "positive"): [9, 9],  # no negative partner -> excluded
                ("c1", E, "positive"): [100, 100],
                ("c1", E, "negative"): [0, 0],  # other mode -> ignored
            }
        )
        c = within_mode_polarity_contrast(acts, R, "positive", "negative")
        np.testing.assert_array_equal(c.vector, [1.5, -0.5])
        assert c.group_ids == ["c1", "c2"]
        assert c.n_groups == 2
        assert c.contrast_id == "R_positive_minus_negative"
        assert c.near_zero is False
        np.testing.assert_array_equal(c.per_group_diffs, [[1, 1], [2, -2]])
        reversed_c = within_mode_polarity_contrast(acts, R, "negative", "positive")
        np.testing.assert_array_equal(reversed_c.vector, -c.vector)

    def test_within_mode_contrast_requires_matched_polarities(self):
        acts = {("c1", R, "positive"): np.ones(2)}
        with pytest.raises(ValueError, match="no contexts with both polarities"):
            within_mode_polarity_contrast(acts, R, "positive", "negative")
        with pytest.raises(ValueError):
            within_mode_polarity_contrast(acts, S, "positive", "negative")

    def test_interaction_hand_values_on_2x2(self):
        c = mode_by_polarity_interaction(_polarity_acts(TWO_BY_TWO))  # enactment vs roleplay, pos vs neu
        np.testing.assert_array_equal(c.vector, [1.0, 1.0])
        assert c.group_ids == ["c1", "c2"]
        assert c.n_groups == 2
        assert c.contrast_id == "ExR_positive_neutral_interaction"
        assert c.near_zero is False
        np.testing.assert_array_equal(c.per_group_diffs, [[2, 0], [0, 2]])

    def test_interaction_swapping_modes_reverses_sign(self):
        acts = _polarity_acts(TWO_BY_TWO)
        c_er = mode_by_polarity_interaction(acts, mode_a=E, mode_b=R)
        c_re = mode_by_polarity_interaction(acts, mode_a=R, mode_b=E)
        np.testing.assert_array_equal(c_re.vector, -c_er.vector)
        assert c_re.contrast_id == "RxE_positive_neutral_interaction"

    def test_interaction_is_zero_when_polarity_effect_is_the_same_in_both_modes(self):
        acts = _polarity_acts(
            {
                ("c1", E, "positive"): [2, 0],
                ("c1", E, "neutral"): [1, 0],  # effect [1, 0]
                ("c1", R, "positive"): [5, 5],
                ("c1", R, "neutral"): [4, 5],  # effect [1, 0]
            }
        )
        c = mode_by_polarity_interaction(acts)
        np.testing.assert_array_equal(c.vector, [0.0, 0.0])
        assert c.near_zero is True

    def test_interaction_requires_full_2x2(self):
        with pytest.raises(ValueError, match="full 2x2"):
            mode_by_polarity_interaction({("c1", E, "positive"): np.ones(2)})


class TestReadouts:
    def test_auc_perfect_reversed_and_tied(self):
        assert mann_whitney_auc([3, 4, 5], [0, 1, 2]) == 1.0
        assert mann_whitney_auc([0, 1, 2], [3, 4, 5]) == 0.0
        assert mann_whitney_auc([1, 1, 1], [1, 1, 1]) == 0.5
        assert mann_whitney_auc([7.0], [7.0]) == 0.5

    def test_auc_partial_overlap_hand_counts(self):
        # pairs (pos, neg): (1,2) no, (1,0) yes, (3,2) yes, (3,0) yes -> 3/4
        assert mann_whitney_auc([1, 3], [2, 0]) == 0.75
        # ties count one half: (1,1) tie, (1,0) yes, (2,1) yes, (2,0) yes -> 3.5/4
        assert mann_whitney_auc([1, 2], [1, 0]) == 0.875
        # swapping the roles gives 1 - AUC
        assert mann_whitney_auc([2, 0], [1, 3]) == 0.25

    def test_auc_empty_is_nan(self):
        assert math.isnan(mann_whitney_auc([], [1.0]))
        assert math.isnan(mann_whitney_auc([1.0], []))

    def test_projection_discrimination_keys_and_values(self):
        H_a = np.array([[2.0, 0.0], [3.0, 5.0]])
        H_b = np.array([[0.0, 9.0], [1.0, -1.0]])
        u = np.array([1.0, 0.0])
        out = projection_discrimination(H_a, H_b, u)
        assert set(out) == {"auc", "mean_diff", "n_a", "n_b"}
        assert out["auc"] == 1.0
        assert out["mean_diff"] == 2.0  # mean([2, 3]) - mean([0, 1])
        assert out["n_a"] == 2
        assert out["n_b"] == 2
        # the second dimension is orthogonal to u and never enters; flipping u flips the readout
        flipped = projection_discrimination(H_a, H_b, -u)
        assert flipped["auc"] == 0.0
        assert flipped["mean_diff"] == -2.0

    def test_cosine(self):
        assert cosine([1, 0], [0, 1]) == 0.0
        assert cosine([1, 1], [2, 2]) == pytest.approx(1.0)
        assert cosine([1, 0], [-3, 0]) == pytest.approx(-1.0)
        assert math.isnan(cosine([0, 0], [1, 0]))


class TestStandardizer:
    H_C = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    # deviations dim0: [-2, 0, 2] -> variance 8/3; dim1: [-10, 0, 10] -> variance 200/3
    STD_C = np.array([math.sqrt(8 / 3), math.sqrt(200 / 3)])

    def test_mean_and_population_std(self):
        mean, std = per_dimension_standardizer(self.H_C, eps=0.0)
        np.testing.assert_array_equal(mean, [3.0, 20.0])
        np.testing.assert_allclose(std, self.STD_C)
        assert not np.allclose(std, self.H_C.std(axis=0, ddof=1))  # population, not sample, std

    def test_eps_added_and_constant_dimension_safe(self):
        _, std = per_dimension_standardizer(self.H_C)
        np.testing.assert_allclose(std, self.STD_C + 1e-8, rtol=0, atol=1e-15)
        _, std_const = per_dimension_standardizer(np.array([[2.0], [2.0]]), eps=1e-8)
        assert std_const[0] == 1e-8

    def test_fitted_on_the_construction_matrix_only(self):
        H_test = np.array([[1000.0, -1000.0]])
        mean, std = per_dimension_standardizer(self.H_C)
        mean_pooled, std_pooled = per_dimension_standardizer(np.vstack([self.H_C, H_test]))
        # the standardizer is a pure function of what it is handed: held-out rows never enter
        np.testing.assert_array_equal(mean, [3.0, 20.0])
        assert not np.allclose(mean, mean_pooled)
        assert not np.allclose(std, std_pooled)
