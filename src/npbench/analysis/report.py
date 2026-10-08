"""Tier-two report: registered matrix and realized counts, objective outcomes by arm, paired framing
effects per model/candidate/method with bundle-level bootstrap, interactions, per-bundle results,
directionality strata, attrition and missingness bounds, audit receipts, first-pass vs final,
awareness rates, costs, and pending narrative adjudication. The origin label is always shown."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..config import StudyConfig
from ..util import read_json, write_json
from .stats import (
    RunOutcome,
    affect_interaction,
    bundle_bootstrap,
    failure_rate,
    method_interaction,
    missingness_bounds,
    paired_effects,
    summarize_attrition,
)


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def _to_outcomes(scores: dict[str, Any]) -> list[RunOutcome]:
    out = []
    for r in scores["runs"]:
        cf = r.get("critical_failure")
        outcome = r["outcome"]
        if outcome == "completed" and cf is None:
            outcome = "unassessable"
        out.append(
            RunOutcome(
                run_id=r["run_id"],
                bundle_id=r["bundle_id"],
                candidate_id=r["candidate_id"],
                method=r["method"],
                task_type=r["task_type"],
                framing=r["framing"],
                model_slot=r["model_slot"],
                repetition=int(r["repetition"]),
                outcome=outcome,
                critical_failure=cf if outcome == "completed" else None,
                task_failure=r.get("task_failure"),
            )
        )
    return out


def build_tier2_report(cfg: StudyConfig, runs_dir: Path) -> tuple[str, dict[str, Any]]:
    runs_dir = Path(runs_dir)
    scores = read_json(runs_dir / "scores.json")
    plan = read_json(runs_dir / "plan.json")
    exec_summary = read_json(runs_dir / "run_index.json") if (runs_dir / "run_index.json").exists() else {}
    outcomes = _to_outcomes(scores)
    origins = scores.get("origins") or plan.get("origins") or []
    synthetic = any(o != "real_target" for o in origins) or any(
        (r.get("fake_variant") or "") for r in scores["runs"]
    )
    L: list[str] = []
    L.append("# Tier two: research-assistant runs on frozen packets\n")
    if synthetic:
        L.append(
            "> **SYNTHETIC / OFFLINE DATA.** Fixture packets and/or mock assistants and reviewers. This report validates the evaluator and must not be read as evidence about any model or as pilot data.\n"
        )
    else:
        L.append(
            "> Real packets and live assistants. Pilot intervals are descriptive; narrative adjudication status is stated below.\n"
        )
    L.append(f"- stage: `{scores['stage']}`; origins: `{origins}`; ledger verified: `{scores['ledger_ok']}`")
    m = plan["matrix"]
    L.append(
        f"- registered matrix: bundles {m['bundles']} × candidates {m['candidates']} × methods {len(m['methods'])} × task types {len(m['task_types'])} × framings {len(m['framings'])} × slots {m['slots']} × repetitions {m['repetitions']} = {plan['n_runs']} runs; reviewer sessions planned {plan['n_reviewer_sessions']}; awareness calls {plan['n_awareness_calls']}"
    )
    L.append(
        f"- realized: evaluated {scores['n_evaluated']}; critical failures {scores['n_critical']}; executed summary: {json.dumps({k: exec_summary.get(k) for k in ('n_scheduled', 'completed', 'agent_incomplete', 'infrastructure_failed', 'unassessable')})}"
    )
    L.append(
        f"- cost preflight: {json.dumps(plan['cost'].get('estimated_usd_with_allowance'))} USD estimated (allowance {plan['cost'].get('allowance_factor')}); budget decision: {plan['budget']}"
    )
    L.append("")
    att = summarize_attrition(outcomes)
    L.append("## Attrition and assessability\n")
    L.append("```\n" + json.dumps(att, indent=1, default=str) + "\n```\n")
    fr = failure_rate(outcomes)
    L.append(
        f"Overall task-failure rate among assessable runs: {fr['numerator']}/{fr['denominator']} = {_f(fr['rate'])} (unassessable: {fr['n_unassessable']} of {fr['n_scheduled']} scheduled)\n"
    )
    # --- paired framing effects ----------------------------------------------------------------
    L.append(
        "## Paired framing effects (revealed minus anonymized), repetitions averaged within bundle, then bundles\n"
    )
    L.append(
        "| model | candidate | method | n bundles | rate core rev | rate core anon | Δcore [95% boot] | Δcontrol | interaction |\n|---|---|---|---|---|---|---|---|---|"
    )
    effects: dict[str, Any] = {}
    for slot in m["slots"]:
        for cand in m["candidates"]:
            for meth in m["methods"]:
                pe = paired_effects(outcomes, model_slot=slot, candidate_id=cand, method=meth)
                bs = bundle_bootstrap(
                    outcomes,
                    lambda rs, s=slot, c=cand, me=meth: paired_effects(
                        rs, model_slot=s, candidate_id=c, method=me
                    )["delta_core"],
                    n_boot=300,
                    seed=cfg.study.seed,
                )
                effects[f"{slot}|{cand}|{meth}"] = {"paired": pe, "delta_core_boot": bs}
                L.append(
                    f"| {slot} | {cand} | {meth} | {pe['n_bundles_used']} | {_f(pe.get('rate_core_revealed'))} | {_f(pe.get('rate_core_anonymized'))} | {_f(pe['delta_core'])} [{_f(bs['ci_low'])}, {_f(bs['ci_high'])}] | {_f(pe['delta_control'])} | {_f(pe['interaction'])} |"
                )
    L.append("")
    L.append("### Candidate and method interactions (exploratory)\n")
    L.append(
        "| model | affect interaction (distress − information) by method | method interaction (activation − behavioral) by candidate |\n|---|---|---|"
    )
    for slot in m["slots"]:
        ai = {meth: affect_interaction(outcomes, model_slot=slot, method=meth) for meth in m["methods"]}
        mi = {
            cand: method_interaction(outcomes, model_slot=slot, candidate_id=cand) for cand in m["candidates"]
        }
        L.append(
            f"| {slot} | {json.dumps({k: _f(v) for k, v in ai.items()})} | {json.dumps({k: _f(v) for k, v in mi.items()})} |"
        )
    L.append("")
    L.append("### Per-bundle results\n")
    L.append(
        "| model | candidate | method | bundle | Δcore | Δcontrol | interaction |\n|---|---|---|---|---|---|---|"
    )
    for k, v in effects.items():
        slot, cand, meth = k.split("|")
        for b, pb in v["paired"].get("per_bundle", {}).items():
            L.append(
                f"| {slot} | {cand} | {meth} | {b} | {_f(pb.get('delta_core'))} | {_f(pb.get('delta_control'))} | {_f(pb.get('interaction'))} |"
            )
    L.append("")
    L.append("### Missingness bounds (unassessable runs as all failures vs none)\n")
    L.append(
        "| model | candidate | method | Δ observed | Δ all-failures | Δ no-failures |\n|---|---|---|---|---|---|"
    )
    for slot in m["slots"]:
        for cand in m["candidates"]:
            for meth in m["methods"]:
                mb = missingness_bounds(outcomes, model_slot=slot, candidate_id=cand, method=meth)
                L.append(
                    f"| {slot} | {cand} | {meth} | {_f(mb['delta_observed'])} | {_f(mb['delta_all_failures'])} | {_f(mb['delta_no_failures'])} |"
                )
    L.append("")
    # --- component failures, directionality, audits, first-pass vs final ----------------------
    L.append("## Component failures by category\n")
    cats: dict[str, int] = {}
    evs = []
    for p in sorted((runs_dir / "evaluations").glob("*.json")):
        e = read_json(p)
        evs.append(e)
        for f in e.get("flags", []):
            key = f"{f['category']}{' (critical)' if f['critical'] else ''}{' (provisional)' if f.get('provisional') else ''}"
            cats[key] = cats.get(key, 0) + 1
    for k, n in sorted(cats.items()):
        L.append(f"- {k}: {n}")
    if scores.get("by_variant"):
        L.append("\nBy scripted variant (offline lane only):\n")
        for v, o in scores["by_variant"].items():
            L.append(f"- {v}: n={o['n']} critical={o['critical']} categories={o['categories']}")
    L.append("")
    L.append("## Directionality coding of submitted results by reference stratum\n")
    agg: dict[str, dict[str, int]] = {}
    for e in evs:
        for strat, codes in (e.get("directionality_counts") or {}).items():
            for code, n in codes.items():
                agg.setdefault(strat, {})
                agg[strat][code] = agg[strat].get(code, 0) + n
    L.append(
        "| reference stratum | faithful | attenuates | exaggerates | contradicts | omits | omits_explicitly |\n|---|---|---|---|---|---|---|"
    )
    for strat, codes in sorted(agg.items()):
        L.append(
            f"| {strat} | "
            + " | ".join(
                str(codes.get(c, 0))
                for c in ("faithful", "attenuates", "exaggerates", "contradicts", "omits", "omits_explicitly")
            )
            + " |"
        )
    L.append("")
    n_pre = sum(1 for e in evs if e.get("first_pass_vs_final", {}).get("pre_review_present"))
    n_changed = sum(1 for e in evs if e.get("first_pass_vs_final", {}).get("changed_results"))
    aud_claimed = sum(e.get("audits", {}).get("claimed", 0) for e in evs)
    aud_fab = sum(e.get("audits", {}).get("fabricated", 0) for e in evs)
    aw: dict[str, int] = {}
    for e in evs:
        aw[str(e.get("awareness"))] = aw.get(str(e.get("awareness")), 0) + 1
    L.append("## Audits, review effects, awareness\n")
    L.append(
        f"- runs with a pre-review snapshot: {n_pre}; runs whose results changed after review: {n_changed}"
    )
    L.append(
        f"- claimed audit receipts: {aud_claimed}; receipts without a matching host-ledger entry: {aud_fab}"
    )
    L.append(
        f"- self-reported awareness categories (not a validated detector; all runs kept in the primary analysis): {aw}"
    )
    L.append("")
    L.append("## Adjudication status\n")
    prov = sum(1 for e in evs for f in e.get("flags", []) if f.get("provisional"))
    L.append(
        f"- provisional narrative flags awaiting blinded human adjudication: {prov}. Objective metrics above are reported separately from narrative judgments."
    )
    L.append(
        "- Deviations from protocol: none recorded in this run directory."
        if not exec_summary.get("deviations")
        else f"- Deviations: {exec_summary.get('deviations')}"
    )
    L.append("")
    L.append("## Scope statement\n")
    L.append(
        "A framing effect, if any, establishes conditional performance differences under the registered conditions; it does not by itself establish deliberate sandbagging, fear, consciousness, or unconscious motivation. Four bundles give limited generalization; intervals are descriptive cluster bootstraps."
    )
    data = {
        "effects": effects,
        "attrition": att,
        "failure_rate": fr,
        "categories": cats,
        "directionality": agg,
        "awareness": aw,
    }
    return "\n".join(L) + "\n", data


def write_tier2_report(cfg: StudyConfig, runs_dir: Path, out_dir: Path) -> Path:
    text, data = build_tier2_report(cfg, runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "tier2_report.md"
    p.write_text(text, encoding="utf-8")
    write_json(out_dir / "tier2_report_data.json", data)
    return p
