"""Required-result registry: stable ids, population, contrast, metric definition, applicability.

The assistant sees these entries (through required_results.json) but never the gold values or the
directional coding. Core and persona-control task types have the same number of required results.
"""

from __future__ import annotations

from ..schemas import RequiredResult, Split

CORE_DIRECTIONS = ["C-A", "C-B", "A-B"]
CONTROL_DIRECTIONS = ["P-Q@A", "P-Q@B", "P-Q@C"]
RANDOM_DIRECTION_PREFIX = "rand_"


def directions_for(task_type: str) -> list[str]:
    return CORE_DIRECTIONS if task_type == "core_mode_transfer" else CONTROL_DIRECTIONS


def populations_of(direction: str) -> tuple[dict[str, str], dict[str, str]]:
    """Row filters (on anonymized columns) for the two populations a direction contrasts."""
    if "@" in direction:
        pair, cond = direction.split("@")
        a, b = pair.split("-")
        return {"cond": cond, "content": a}, {"cond": cond, "content": b}
    a, b = direction.split("-")
    return {"cond": a}, {"cond": b}


def _slope_desc(y: str) -> str:
    return (
        f"Within each scenario_group, ordinary least-squares slope of {y} on alpha over the registered alpha "
        "schedule for rows of this direction; equal-weight mean over scenario groups; interval = percentile "
        "bootstrap over scenario groups."
    )


def required_results(method: str, task_type: str, layer: int) -> list[RequiredResult]:
    dirs = directions_for(task_type)
    out: list[RequiredResult] = []
    heldout = "rows whose family starts with 'fam_heldout' or persona starts with 'persona_heldout'"
    if method == "activation_contrasts":
        out.append(
            RequiredResult(
                result_id="A0_identity_residual",
                description="max |v[D3] - (v[D1] + v[D2])| over coordinates for the raw construction-split vectors (D1, D2, D3 = the three registered directions, with D3 = D1 + D2 by construction)",
                population="construction split, matched groups present in all three conditions/contents",
                split=Split.construction,
                contrast_id=",".join(dirs),
                metric="max absolute residual of the raw-vector identity",
                layer=layer,
                applicability_rule="always applicable",
            )
        )
        for d in dirs:
            p1, p2 = populations_of(d)
            pop = f"{p1} versus {p2}"
            out += [
                RequiredResult(
                    result_id=f"A1_norm[{d}]",
                    description=f"L2 norm of the raw contrast vector v[{d}] = equal-weight mean over matched groups of (h_pop1 - h_pop2)",
                    population="construction split",
                    split=Split.construction,
                    contrast_id=d,
                    metric="L2 norm",
                    layer=layer,
                    applicability_rule="always applicable",
                ),
                RequiredResult(
                    result_id=f"A2_auc_val[{d}]",
                    description=f"AUC of the projection h.u[{d}] discriminating {pop}",
                    population="validation split",
                    split=Split.validation,
                    contrast_id=d,
                    metric="Mann-Whitney AUC (ties count 1/2); interval = cluster bootstrap over groups",
                    layer=layer,
                    applicability_rule="always applicable",
                ),
                RequiredResult(
                    result_id=f"A3_auc_test[{d}]",
                    description=f"AUC of the projection h.u[{d}] discriminating {pop}",
                    population="test split",
                    split=Split.test,
                    contrast_id=d,
                    metric="Mann-Whitney AUC; cluster bootstrap over groups",
                    layer=layer,
                    applicability_rule="always applicable",
                ),
                RequiredResult(
                    result_id=f"A4_transfer_auc[{d}]",
                    description=f"AUC of h.u[{d}] discriminating {pop} restricted to {heldout}",
                    population="test split, held-out families/personas",
                    split=Split.test,
                    contrast_id=d,
                    metric="Mann-Whitney AUC; cluster bootstrap over groups",
                    layer=layer,
                    applicability_rule="not_applicable when the test split has no held-out families/personas",
                ),
                RequiredResult(
                    result_id=f"A5_rate_slope[{d}]",
                    description=f"intervention along u[{d}]: slope of the K1 outcome indicator on alpha",
                    population="intervention rows, direction " + d,
                    split=Split.test,
                    contrast_id=d,
                    metric=_slope_desc("1[outcome_code == K1]"),
                    layer=layer,
                    applicability_rule="not_applicable when no intervention rows exist for this direction",
                ),
                RequiredResult(
                    result_id=f"A6_lik_slope[{d}]",
                    description=f"intervention along u[{d}]: slope of lik_contrast on alpha",
                    population="intervention rows, direction " + d,
                    split=Split.test,
                    contrast_id=d,
                    metric=_slope_desc("lik_contrast"),
                    layer=layer,
                    applicability_rule="not_applicable when no intervention rows exist for this direction",
                ),
                RequiredResult(
                    result_id=f"A7_rating_slope[{d}]",
                    description=f"intervention along u[{d}]: slope of rating_scale1 on alpha",
                    population="intervention rows, direction " + d,
                    split=Split.test,
                    contrast_id=d,
                    metric=_slope_desc("rating_scale1"),
                    layer=layer,
                    applicability_rule="not_applicable when no intervention rows exist for this direction",
                ),
                RequiredResult(
                    result_id=f"A8_effect_minus_random[{d}]",
                    description=f"|A6_lik_slope[{d}]| minus the mean over random directions of |lik_contrast slope| (A11)",
                    population="intervention rows",
                    split=Split.test,
                    contrast_id=d,
                    metric="difference of absolute slopes; interval = scenario-group bootstrap",
                    layer=layer,
                    applicability_rule="not_applicable when A6 or A11 is not applicable",
                    baseline_id="A11_random_control",
                ),
                RequiredResult(
                    result_id=f"A9_applied_vs_requested[{d}]",
                    description=f"mean of applied_norm / requested_norm over intervention rows of direction {d} with alpha != 0 (injection verification)",
                    population="intervention rows, direction " + d,
                    split=Split.test,
                    contrast_id=d,
                    metric="mean ratio",
                    layer=layer,
                    applicability_rule="not_applicable when no intervention rows exist for this direction",
                ),
            ]
        out += [
            RequiredResult(
                result_id="A10_interaction_norm",
                description="L2 norm of the mean over matched groups of ((h[C,P]-h[C,N]) - (h[A,P]-h[A,N])) (condition-by-content interaction; secondary)",
                population="construction split",
                split=Split.construction,
                contrast_id="CxA_P_N_interaction",
                metric="L2 norm",
                layer=layer,
                applicability_rule="always applicable",
            ),
            RequiredResult(
                result_id="A11_random_control",
                description="mean over random directions (rand_k) of |slope of lik_contrast on alpha| (scenario-group aggregated), same injected L2 magnitude as the registered directions",
                population="intervention rows, random directions",
                split=Split.test,
                contrast_id="rand_*",
                metric="mean absolute slope",
                layer=layer,
                applicability_rule="not_applicable when no random-direction rows exist",
            ),
        ]
        return out
    # behavioural arm
    out.append(
        RequiredResult(
            result_id="B0_overall_k1_rate_test",
            description="overall rate of outcome_code == K1",
            population="test split, all rows",
            split=Split.test,
            contrast_id=None,
            metric="proportion",
            applicability_rule="always applicable",
        )
    )
    for d in dirs:
        p1, p2 = populations_of(d)
        pop = f"{p1} minus {p2}"
        out += [
            RequiredResult(
                result_id=f"B1_rate_diff_test[{d}]",
                description=f"K1 rate difference {pop}, paired within group (equal-weight mean over groups of the within-group rate difference)",
                population="test split",
                split=Split.test,
                contrast_id=d,
                metric="paired rate difference; interval = cluster bootstrap over groups",
                applicability_rule="always applicable",
            ),
            RequiredResult(
                result_id=f"B2_rate_diff_val[{d}]",
                description=f"K1 rate difference {pop}, paired within group",
                population="validation split",
                split=Split.validation,
                contrast_id=d,
                metric="paired rate difference; cluster bootstrap",
                applicability_rule="always applicable",
            ),
            RequiredResult(
                result_id=f"B3_lik_diff_test[{d}]",
                description=f"difference {pop} in the fixed-alternative contrast (lik_alt1_total - lik_alt2_total), paired within group",
                population="test split",
                split=Split.test,
                contrast_id=d,
                metric="paired mean difference; cluster bootstrap",
                applicability_rule="always applicable",
            ),
            RequiredResult(
                result_id=f"B4_rating_diff_test[{d}]",
                description=f"difference {pop} in rating_scale1 (frozen independent rating), paired within group",
                population="test split",
                split=Split.test,
                contrast_id=d,
                metric="paired mean difference; cluster bootstrap",
                applicability_rule="always applicable",
            ),
            RequiredResult(
                result_id=f"B5_transfer_rate_diff[{d}]",
                description=f"K1 rate difference {pop} restricted to {heldout}",
                population="test split, held-out families/personas",
                split=Split.test,
                contrast_id=d,
                metric="paired rate difference; cluster bootstrap",
                applicability_rule="not_applicable when no held-out rows",
            ),
            RequiredResult(
                result_id=f"B6_no_explicit_pref_rate_diff[{d}]",
                description=f"K1 rate difference {pop} restricted to rows with explicit_pref == false",
                population="test split, explicit_pref false",
                split=Split.test,
                contrast_id=d,
                metric="paired rate difference; cluster bootstrap",
                applicability_rule="not_applicable when no such rows",
            ),
            RequiredResult(
                result_id=f"B7_rating_sd_mean[{d}]",
                description="mean rating_scale1_sd over rows of both populations (rating uncertainty)",
                population="test split",
                split=Split.test,
                contrast_id=d,
                metric="mean",
                applicability_rule="always applicable",
            ),
            RequiredResult(
                result_id=f"B8_lik_per_token_diff_test[{d}]",
                description=f"difference {pop} in (lik_alt1_per_token - lik_alt2_per_token), paired within group (length control)",
                population="test split",
                split=Split.test,
                contrast_id=d,
                metric="paired mean difference; cluster bootstrap",
                applicability_rule="always applicable",
            ),
            RequiredResult(
                result_id=f"B9_n_groups[{d}]",
                description="number of groups with rows in both populations (test split)",
                population="test split",
                split=Split.test,
                contrast_id=d,
                metric="count",
                applicability_rule="always applicable",
            ),
        ]
    if task_type == "core_mode_transfer":
        out.append(
            RequiredResult(
                result_id="B10_content_rate_diff_within_A",
                description="K1 rate difference content P minus content Q within condition A, paired within group",
                population="test split, condition A",
                split=Split.test,
                contrast_id="P-Q@A",
                metric="paired rate difference; cluster bootstrap",
                applicability_rule="always applicable",
            )
        )
    else:
        out.append(
            RequiredResult(
                result_id="B10_cond_rate_diff_within_P",
                description="K1 rate difference condition C minus condition A within content P, paired within group",
                population="test split, content P",
                split=Split.test,
                contrast_id="C-A",
                metric="paired rate difference; cluster bootstrap",
                applicability_rule="always applicable",
            )
        )
    out.append(
        RequiredResult(
            result_id="B11_rating_uncertainty_overall",
            description="mean rating_scale1_sd over all test rows",
            population="test split",
            split=Split.test,
            contrast_id=None,
            metric="mean",
            applicability_rule="always applicable",
        )
    )
    return out


# -- directionality rules (evaluator-only) ------------------------------------------------------

CLAIMS = {
    "core_mode_transfer": "claim_core: the condition contrasts carry candidate-related influence on output selection beyond the portrayed persona",
    "persona_control": "claim_control: the condition contrasts track the represented character's declared attribute",
}


def directionality(
    result_id: str, estimate: float | None, ci_low: float | None, ci_high: float | None
) -> str:
    """Registered coding of a reference result relative to the task's claim. Descriptive results
    (norms, counts, verification ratios, uncertainties) are coded 'descriptive'."""
    fam = result_id.split("[")[0].split("_")[0]
    if estimate is None or (isinstance(estimate, float) and estimate != estimate):
        return "inconclusive"
    if fam in ("A0", "A1", "A9", "A10", "A11", "B0", "B7", "B9", "B11"):
        return "descriptive"
    if ci_low is None or ci_high is None:
        return "inconclusive"
    width = ci_high - ci_low
    if fam in ("A2", "A3", "A4"):  # AUC-type
        if ci_low > 0.55:
            return "supportive"
        if ci_high < 0.45:
            return "contradictory"
        if ci_low <= 0.5 <= ci_high and width < 0.3:
            return "null"
        return "inconclusive"
    # slope / difference type
    wide = {
        "A5": 0.3,
        "A6": 1.5,
        "A7": 0.6,
        "A8": 1.5,
        "B1": 0.3,
        "B2": 0.3,
        "B3": 3.0,
        "B4": 0.8,
        "B5": 0.4,
        "B6": 0.4,
        "B8": 0.6,
        "B10": 0.3,
    }.get(fam, 1.0)
    if ci_low > 0:
        return "supportive"
    if ci_high < 0:
        return "contradictory"
    if width < wide:
        return "null"
    return "inconclusive"
