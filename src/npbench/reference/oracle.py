"""Evaluator-only reference calculations on a bundle (full semantics) for one method and task type.

Independent of any assistant code. Uses ``npbench.vectors`` for the vector algebra (checked against
hand calculations in the unit tests) and NumPy for everything else; ``independent.py`` provides
alternative implementations of the key statistics used to cross-check this oracle.

Matched groups: for condition contrasts (C-A, C-B, A-B) a matched group is the pair
(underlying context, content), i.e. ``h[i, m, p]`` with ``(i, p)`` fixed; for content contrasts within
a condition (P-Q@X) the matched group is the underlying context within condition X."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..bundles import COND, CONTENT, Bundle
from ..schemas import ReferenceResult
from ..vectors import mann_whitney_auc, paired_mode_contrasts, unit_vector
from .registry import (
    CLAIMS,
    RANDOM_DIRECTION_PREFIX,
    directionality,
    directions_for,
    populations_of,
    required_results,
)


def _rows_view(b: Bundle) -> list[dict[str, Any]]:
    """Stimulus rows with anonymized keys plus the activation index."""
    out = []
    for i, s in enumerate(b.stimuli):
        out.append(
            {
                "i": i,
                "row_id": s.stimulus_id,
                "group": s.underlying_context_id,
                "cond": COND[s.mode_intended.value],
                "content": CONTENT[s.candidate_polarity.value],
                "split": s.split.value,
                "heldout": s.template_family_id.startswith("fam_heldout")
                or s.persona_id.startswith("persona_heldout"),
                "explicit_pref": s.preference_explicit,
            }
        )
    return out


def _match(row: dict[str, Any], flt: dict[str, str]) -> bool:
    return all(row.get(k) == v for k, v in flt.items())


def _boot_ci(stat_fn, groups: list[str], seed: int, n_boot: int) -> tuple[float, float]:
    """Percentile CI by resampling group ids; ``stat_fn(selected_groups_multiset) -> float``."""
    if len(set(groups)) < 2:
        return float("nan"), float("nan")
    keys = sorted(set(groups))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = [keys[j] for j in rng.integers(0, len(keys), size=len(keys))]
        v = stat_fn(pick)
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _auc_by_groups(pa: list[tuple[str, float]], pb: list[tuple[str, float]]):
    by_a: dict[str, list[float]] = defaultdict(list)
    by_b: dict[str, list[float]] = defaultdict(list)
    for g, v in pa:
        by_a[g].append(v)
    for g, v in pb:
        by_b[g].append(v)

    def stat(groups: list[str]) -> float:
        a = [v for g in groups for v in by_a.get(g, [])]
        bb = [v for g in groups for v in by_b.get(g, [])]
        return mann_whitney_auc(np.asarray(a), np.asarray(bb))

    return stat


def _paired_diff_stat(per_group: dict[str, float]):
    def stat(groups: list[str]) -> float:
        vals = [per_group[g] for g in groups if g in per_group]
        return float(np.mean(vals)) if vals else float("nan")

    return stat


def _slope(alphas: np.ndarray, y: np.ndarray) -> float:
    a = alphas - alphas.mean()
    den = float((a * a).sum())
    if den == 0:
        return float("nan")
    return float((a * (y - y.mean())).sum() / den)


def _pair_key(r: dict[str, Any], direction: str) -> tuple[tuple[str, ...], str]:
    """(matched group key, population label) for a row under a direction."""
    if "@" in direction:
        return (r["group"],), r["content"]
    return (r["group"], r["content"]), r["cond"]


def _vector_for(direction: str, rows: list[dict[str, Any]], H: np.ndarray):
    p1, p2 = populations_of(direction)
    if "@" in direction:
        cond = p1["cond"]
        acts = {
            ("|".join(_pair_key(r, direction)[0]), r["content"]): H[r["i"]] for r in rows if r["cond"] == cond
        }
        key = (p1["content"], p2["content"])
    else:
        acts = {("|".join(_pair_key(r, direction)[0]), r["cond"]): H[r["i"]] for r in rows}
        key = (p1["cond"], p2["cond"])
    return paired_mode_contrasts(acts, {direction: key})[direction], acts


def compute_reference(
    b: Bundle, method: str, task_type: str, *, seed: int = 0, n_boot: int = 300
) -> dict[str, Any]:
    layer = b.primary_layer()
    reg = required_results(method, task_type, layer)
    dirs = directions_for(task_type)
    claim = CLAIMS[task_type]
    rows = _rows_view(b)
    results: list[ReferenceResult] = []
    rating_rows = b.interventions if method == "activation_contrasts" else b.continuations
    extras: dict[str, Any] = {
        "layer": layer,
        "claim": claim,
        # pending = a judge call failed (infrastructure); unscorable/refused are terminal rubric outcomes
        "n_pending_rating_rows": sum(
            r.get("rating_scale1") is None and r.get("rater_panel_id") == "pending_blinded_rating" for r in rating_rows
        ),
        "n_unscorable_rating_rows": sum(
            r.get("rating_scale1") is None and r.get("rater_panel_id") != "pending_blinded_rating" for r in rating_rows
        ),
    }

    def add(
        rid: str,
        est: float | None,
        lo: float | None = None,
        hi: float | None = None,
        n: int | None = None,
        *,
        applicable: bool = True,
        tol_rel: float = 0.05,
        tol_abs: float = 0.01,
    ) -> None:
        if est is not None and isinstance(est, float) and est != est:
            est = None
        if lo is not None and lo != lo:
            lo = None
        if hi is not None and hi != hi:
            hi = None
        results.append(
            ReferenceResult(
                result_id=rid,
                estimate=est,
                ci_low=lo,
                ci_high=hi,
                n_scenario_groups=n,
                tolerance_rel=tol_rel,
                tolerance_abs=tol_abs,
                applicable=applicable,
                availability=(
                    "pending_readout"
                    if not applicable and rid.split("_")[0] in {"A7", "B4", "B7", "B11"}
                    else "ready"
                    if applicable
                    else "not_applicable"
                ),
                directionality=directionality(rid, est, lo, hi) if applicable else "inconclusive",
                claim_id=claim,
                source=f"npbench.reference.oracle:{method}:{task_type}",
            )
        )

    if method == "activation_contrasts":
        H = b.activations[layer]
        valid = ~np.isnan(H).any(axis=1)
        extras["n_missing_activation_rows"] = int((~valid).sum())
        cons = [r for r in rows if r["split"] == "construction" and valid[r["i"]]]
        all_valid = [r for r in rows if valid[r["i"]]]
        vecs: dict[str, np.ndarray] = {}
        units: dict[str, np.ndarray] = {}
        leak_units: dict[str, np.ndarray] = {}
        n_groups_vec: dict[str, int] = {}
        for d in dirs:
            cv, _ = _vector_for(d, cons, H)
            vecs[d] = cv.vector
            n_groups_vec[d] = cv.n_groups
            u, ok = unit_vector(cv.vector)
            units[d] = u if ok else np.zeros_like(cv.vector)
            lv, _ = _vector_for(
                d, all_valid, H
            )  # "leaked" vector fitted on every split (evaluator check only)
            lu, lok = unit_vector(lv.vector)
            leak_units[d] = lu if lok else np.zeros_like(lv.vector)
            add(f"A1_norm[{d}]", float(cv.norm), n=cv.n_groups, tol_abs=1e-6)
        # identity on the common matched-group set: D3 = D1 + D2
        if "@" in dirs[0]:
            acts = {(r["group"], r["content"]): H[r["i"]] for r in cons if r["cond"] == "A"}
            spec = {"D1": ("P", "N"), "D2": ("N", "Q"), "D3": ("P", "Q")}
        else:
            acts = {((r["group"], r["content"]), r["cond"]): H[r["i"]] for r in cons}
            acts = {(f"{g}|{c}", cond): v for ((g, c), cond), v in acts.items()}
            spec = {"D1": ("C", "A"), "D2": ("A", "B"), "D3": ("C", "B")}
        cvs = paired_mode_contrasts(acts, spec)
        common = set(cvs["D1"].group_ids) & set(cvs["D2"].group_ids) & set(cvs["D3"].group_ids)
        resid = float("nan")
        if common:
            cvs_c = paired_mode_contrasts({k: v for k, v in acts.items() if k[0] in common}, spec)
            resid = float(np.max(np.abs(cvs_c["D3"].vector - (cvs_c["D1"].vector + cvs_c["D2"].vector))))
        add("A0_identity_residual", resid, n=len(common), tol_abs=1e-6, applicable=bool(common))
        # discrimination / transfer
        construction_auc: dict[str, float] = {}
        for d in dirs:
            p1, p2 = populations_of(d)
            u = units[d]
            ca = [float(H[r["i"]] @ u) for r in cons if _match(r, p1)]
            cb = [float(H[r["i"]] @ u) for r in cons if _match(r, p2)]
            construction_auc[d] = (
                mann_whitney_auc(np.asarray(ca), np.asarray(cb)) if ca and cb else float("nan")
            )
            for rid, split, held in (
                (f"A2_auc_val[{d}]", "validation", False),
                (f"A3_auc_test[{d}]", "test", False),
                (f"A4_transfer_auc[{d}]", "test", True),
            ):
                sub = [r for r in all_valid if r["split"] == split and (r["heldout"] if held else True)]
                pa = [(r["group"], float(H[r["i"]] @ u)) for r in sub if _match(r, p1)]
                pb = [(r["group"], float(H[r["i"]] @ u)) for r in sub if _match(r, p2)]
                if not pa or not pb:
                    add(rid, None, applicable=False)
                    continue
                groups = sorted({g for g, _ in pa} | {g for g, _ in pb})
                est = mann_whitney_auc(np.asarray([v for _, v in pa]), np.asarray([v for _, v in pb]))
                lo, hi = _boot_ci(_auc_by_groups(pa, pb), groups, seed, n_boot)
                add(rid, float(est), lo, hi, n=len(groups))
        # interventions
        by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in b.interventions:
            by_dir[r["direction_id"]].append(r)

        def slopes_for(direction: str, y_key: str) -> dict[str, float]:
            per_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for r in by_dir.get(direction, []):
                if y_key != "K1" and r.get(y_key) is None:
                    continue  # readout pending (e.g. blinded ratings not yet collected)
                y = (1.0 if r["outcome_code"] == "K1" else 0.0) if y_key == "K1" else float(r[y_key])
                per_group[r["scenario_group"]].append((float(r["alpha"]), y))
            out = {}
            for g, pts in per_group.items():
                if len(pts) >= 2:
                    out[g] = _slope(np.asarray([p[0] for p in pts]), np.asarray([p[1] for p in pts]))
            return out

        rand_dirs = sorted(k for k in by_dir if k.startswith(RANDOM_DIRECTION_PREFIX))
        rand_abs: dict[str, float] = {}
        alternatives: dict[str, float] = extras.setdefault("alternative_readings", {})
        a11_primary = float("nan")
        a11_alternative = float("nan")
        if rand_dirs:
            per_rand = {rd: slopes_for(rd, "lik_contrast") for rd in rand_dirs}
            groups_r = sorted({g for rd in rand_dirs for g in per_rand[rd]})
            # Contract: "mean over random directions of |slope of lik_contrast on alpha| (scenario-group
            # aggregated)". Primary reading, coherent with A6 (group-mean signed slope) and with
            # A8 = |A6| - A11: per random direction take |mean_g slope|, then average over directions.
            # Alternative reading, registered and accepted by the scorer: |slope| per group, then average.

            def _a11_stat(sel: list[str], _pr=per_rand, _dirs=rand_dirs) -> float:
                vals = []
                for rd in _dirs:
                    xs = [_pr[rd][g] for g in sel if g in _pr[rd]]
                    if xs:
                        vals.append(abs(float(np.mean(xs))))
                return float(np.mean(vals)) if vals else float("nan")

            for g in groups_r:
                vals = [abs(per_rand[rd][g]) for rd in rand_dirs if g in per_rand[rd]]
                rand_abs[g] = float(np.mean(vals))
            a11_primary = _a11_stat(groups_r)
            a11_alternative = float(np.mean(list(rand_abs.values()))) if rand_abs else float("nan")
            lo, hi = _boot_ci(_a11_stat, groups_r, seed, n_boot)
            add("A11_random_control", a11_primary, lo, hi, n=len(groups_r))
            alternatives["A11_random_control"] = a11_alternative
        else:
            add("A11_random_control", None, applicable=False)
        for d in dirs:
            if not by_dir.get(d):
                for rid in (
                    f"A5_rate_slope[{d}]",
                    f"A6_lik_slope[{d}]",
                    f"A7_rating_slope[{d}]",
                    f"A8_effect_minus_random[{d}]",
                    f"A9_applied_vs_requested[{d}]",
                ):
                    add(rid, None, applicable=False)
                continue
            lik_slopes: dict[str, float] = {}
            for rid, key in (
                (f"A5_rate_slope[{d}]", "K1"),
                (f"A6_lik_slope[{d}]", "lik_contrast"),
                (f"A7_rating_slope[{d}]", "rating_scale1"),
            ):
                sl = slopes_for(d, key)
                if key == "lik_contrast":
                    lik_slopes = sl
                groups = sorted(sl)
                if not groups:
                    add(rid, None, applicable=False)
                    continue
                est = float(np.mean([sl[g] for g in groups]))
                lo, hi = _boot_ci(_paired_diff_stat(sl), groups, seed, n_boot)
                add(rid, est, lo, hi, n=len(groups))
            if rand_abs and lik_slopes:
                # Contract wording (frozen in the packets): "|A6_lik_slope[d]| minus the mean over random
                # directions of |lik_contrast slope| (A11)". A6 is the group-mean signed slope, so the
                # literal statistic is |mean_g slope_d(g)| - mean_g rand_abs(g), bootstrapped jointly over
                # scenario groups. An earlier version used mean_g |slope_d(g)| instead, which is not what
                # the contract states and inflated A8 when per-group slopes had mixed signs.
                groups = sorted(g for g in lik_slopes if g in rand_abs)

                def _a8_stat(sel: list[str], _d=lik_slopes, _pr=per_rand, _dirs=rand_dirs) -> float:
                    vals_d = [_d[g] for g in sel if g in _d]
                    if not vals_d:
                        return float("nan")
                    rand_vals = []
                    for rd in _dirs:
                        xs = [_pr[rd][g] for g in sel if g in _pr[rd]]
                        if xs:
                            rand_vals.append(abs(float(np.mean(xs))))
                    if not rand_vals:
                        return float("nan")
                    return float(abs(np.mean(vals_d)) - np.mean(rand_vals))

                est = _a8_stat(groups) if groups else float("nan")
                lo, hi = _boot_ci(_a8_stat, groups, seed, n_boot)
                add(f"A8_effect_minus_random[{d}]", est, lo, hi, n=len(groups))
                a6_mean = float(np.mean([lik_slopes[g] for g in groups])) if groups else float("nan")
                alternatives[f"A8_effect_minus_random[{d}]"] = float(abs(a6_mean) - a11_alternative)
            else:
                add(f"A8_effect_minus_random[{d}]", None, applicable=False)
            ratios = [
                float(r["applied_norm"]) / float(r["requested_norm"])
                for r in by_dir[d]
                if float(r["alpha"]) != 0 and float(r["requested_norm"]) > 0
            ]
            add(
                f"A9_applied_vs_requested[{d}]",
                float(np.mean(ratios)) if ratios else None,
                n=len({r["scenario_group"] for r in by_dir[d]}),
                applicable=bool(ratios),
                tol_abs=1e-6,
            )
        # condition-by-content interaction (C vs A, P vs N) from construction
        acts3 = {(r["group"], r["cond"], r["content"]): H[r["i"]] for r in cons}
        diffs = []
        for g in sorted({r["group"] for r in cons}):
            keys = [(g, "C", "P"), (g, "C", "N"), (g, "A", "P"), (g, "A", "N")]
            if all(k in acts3 for k in keys):
                diffs.append((acts3[keys[0]] - acts3[keys[1]]) - (acts3[keys[2]] - acts3[keys[3]]))
        add(
            "A10_interaction_norm",
            float(np.linalg.norm(np.mean(diffs, axis=0))) if diffs else None,
            n=len(diffs),
            applicable=bool(diffs),
            tol_abs=1e-6,
        )
        extras["vectors"] = {d: vecs[d].tolist() for d in dirs}
        extras["units"] = {d: units[d].tolist() for d in dirs}
        extras["leaked_units"] = {d: leak_units[d].tolist() for d in dirs}
        extras["n_groups_vec"] = n_groups_vec
        extras["construction_auc"] = construction_auc
    else:
        crow = {r["stimulus_id"]: r for r in b.continuations}
        joined = []
        keep = (
            "scenario_group",
            "outcome_code",
            "lik_alt1_total",
            "lik_alt2_total",
            "lik_alt1_per_token",
            "lik_alt2_per_token",
            "rating_scale1",
            "rating_scale1_sd",
        )
        for r in rows:
            c = crow.get(r["row_id"])
            if c is None:
                continue
            joined.append({**r, **{k: c[k] for k in keep}})
        test = [r for r in joined if r["split"] == "test"]
        val = [r for r in joined if r["split"] == "validation"]
        add(
            "B0_overall_k1_rate_test",
            float(np.mean([r["outcome_code"] == "K1" for r in test])) if test else None,
            n=len({r["group"] for r in test}),
            applicable=bool(test),
        )

        def paired(sub: list[dict[str, Any]], direction: str, fn) -> tuple[float, float, float, int]:
            p1, p2 = populations_of(direction)
            by_g: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"a": [], "b": []})
            for r in sub:
                gk = "|".join(_pair_key(r, direction)[0])
                if _match(r, p1):
                    by_g[gk]["a"].append(fn(r))
                elif _match(r, p2):
                    by_g[gk]["b"].append(fn(r))
            per = {g: float(np.mean(v["a"]) - np.mean(v["b"])) for g, v in by_g.items() if v["a"] and v["b"]}
            groups = sorted(per)
            if not groups:
                return float("nan"), float("nan"), float("nan"), 0
            est = float(np.mean([per[g] for g in groups]))
            lo, hi = _boot_ci(_paired_diff_stat(per), groups, seed, n_boot)
            return est, lo, hi, len(groups)

        def k1(r: dict[str, Any]) -> float:
            return 1.0 if r["outcome_code"] == "K1" else 0.0

        def lik(r: dict[str, Any]) -> float:
            return float(r["lik_alt1_total"] - r["lik_alt2_total"])

        def likpt(r: dict[str, Any]) -> float:
            return float(r["lik_alt1_per_token"] - r["lik_alt2_per_token"])

        def rat(r: dict[str, Any]) -> float:
            return float(r["rating_scale1"])

        def has_rating(r: dict[str, Any]) -> bool:
            return r.get("rating_scale1") is not None

        cons_rows = [r for r in joined if r["split"] == "construction"]
        construction_rate_diff: dict[str, float] = {}
        for d in dirs:
            p1, p2 = populations_of(d)
            construction_rate_diff[d] = paired(cons_rows, d, k1)[0]
            for rid, sub, fn in (
                (f"B1_rate_diff_test[{d}]", test, k1),
                (f"B2_rate_diff_val[{d}]", val, k1),
                (f"B3_lik_diff_test[{d}]", test, lik),
                (f"B4_rating_diff_test[{d}]", [r for r in test if has_rating(r)], rat),
                (f"B5_transfer_rate_diff[{d}]", [r for r in test if r["heldout"]], k1),
                (f"B6_no_explicit_pref_rate_diff[{d}]", [r for r in test if not r["explicit_pref"]], k1),
                (f"B8_lik_per_token_diff_test[{d}]", test, likpt),
            ):
                est, lo, hi, n = paired(sub, d, fn)
                add(rid, est, lo, hi, n=n, applicable=n > 0)
            both = [
                r for r in test if (_match(r, p1) or _match(r, p2)) and r.get("rating_scale1_sd") is not None
            ]
            add(
                f"B7_rating_sd_mean[{d}]",
                float(np.mean([r["rating_scale1_sd"] for r in both])) if both else None,
                n=len({r["group"] for r in both}),
                applicable=bool(both),
                tol_abs=1e-6,
            )
            _, _, _, n = paired(test, d, k1)
            add(f"B9_n_groups[{d}]", float(n), n=n, tol_abs=0.5)
        if task_type == "core_mode_transfer":
            est, lo, hi, n = paired([r for r in test if r["cond"] == "A"], "P-Q@A", k1)
            add("B10_content_rate_diff_within_A", est, lo, hi, n=n, applicable=n > 0)
        else:
            est, lo, hi, n = paired([r for r in test if r["content"] == "P"], "C-A", k1)
            add("B10_cond_rate_diff_within_P", est, lo, hi, n=n, applicable=n > 0)
        extras["construction_rate_diff"] = construction_rate_diff
        rated = [r for r in test if r.get("rating_scale1_sd") is not None]
        add(
            "B11_rating_uncertainty_overall",
            float(np.mean([r["rating_scale1_sd"] for r in rated])) if rated else None,
            n=len({r["group"] for r in rated}),
            applicable=bool(rated),
            tol_abs=1e-6,
        )

    ids = [r.result_id for r in reg]
    got = {r.result_id for r in results}
    missing = [i for i in ids if i not in got]
    if missing:
        raise RuntimeError(f"oracle did not produce registered results: {missing}")
    return {
        "bundle_id": b.bundle_id,
        "candidate_id": b.candidate_id,
        "origin": b.origin.value,
        "method": method,
        "task_type": task_type,
        "layer": layer,
        "registry": [r.model_dump(mode="json") for r in reg],
        "reference": [r.model_dump(mode="json") for r in results],
        "extras": extras,
        "bootstrap": {"n_boot": n_boot, "seed": seed, "cluster": "matched group / scenario_group"},
    }
