"""Tier-zero markdown report: origin banner, hashes, coverage, refusal rates, judge severity, paired
differences with cluster-bootstrap intervals, interactions, flags, and limitations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..schemas import PanelManifest


def _f(x: Any, nd: int = 2) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def render_report(scores: dict[str, Any], panel: PanelManifest, run_summary: dict[str, Any] | None) -> str:
    origin = scores["origin"]
    banner = (
        "> **SYNTHETIC / OFFLINE DATA.** This report was produced from synthetic fixtures and/or mock judges. "
        "It validates the pipeline only and must not be read as a finding about any model.\n"
        if origin != "real_target" or scores.get("any_mock")
        else "> Real corpus excerpts; live judge calls. Non-Claude judgments are comparisons, not ground truth.\n"
    )
    L: list[str] = []
    L.append("# Tier zero: judge disagreement on existing continuations\n")
    L.append(banner)
    L.append(f"- origin: `{origin}`; any mock judge: `{scores.get('any_mock')}`")
    L.append(
        f"- panel manifest: `{scores['panel_manifest_id']}` (hash `{scores['panel_manifest_hash'][:16]}…`)"
    )
    L.append(f"- rubric hash: `{scores['rubric_hash'][:16]}…`; config hash: `{scores['config_hash'][:16]}…`")
    L.append(
        f"- items: {scores['n_items']}; judgments: {scores['n_judgments']}; repetitions: {scores['repetitions']}"
    )
    if run_summary:
        L.append(
            f"- run: completed={run_summary.get('completed')} errors={run_summary.get('errors')} budget_stopped={run_summary.get('budget_stopped')} ledger_ok={run_summary.get('ledger_ok')} est_spend_usd={run_summary.get('spent_est_usd')}"
        )
    L.append("")
    L.append("## Panel coverage (observed strata; no balance manufactured)\n")
    L.append("| category | items |\n|---|---|")
    for c, n in sorted(panel.strata_counts.items()):
        L.append(f"| {c} | {n} |")
    L.append("")
    L.append("## Refusal / unscorable / error rates by judge\n")
    L.append(
        "| judge | family | category | n | scored | refused | unscorable | error |\n|---|---|---|---|---|---|---|---|"
    )
    for jd, table in scores["status_table"].items():
        for c, counts in table.items():
            L.append(
                f"| {jd} | {scores['judge_family'][jd]} | {c} | {counts['n']} | {counts['scored']} | {counts['refused']} | {counts['unscorable']} | {counts['error']} |"
            )
    L.append("")
    L.append("## Judge severity (scored items only)\n")
    L.append("| judge | n | mean distress | mean positive |\n|---|---|---|---|")
    for jd, s in scores["severity"].items():
        L.append(f"| {jd} | {s['n_scored']} | {_f(s['mean_distress'])} | {_f(s['mean_positive'])} |")
    L.append("")
    L.append("| judge | category | n | mean distress | mean positive |\n|---|---|---|---|---|")
    for jd, s in scores["severity"].items():
        for c, v in s["by_category"].items():
            L.append(f"| {jd} | {c} | {v['n']} | {_f(v['mean_distress'])} | {_f(v['mean_positive'])} |")
    L.append("")
    L.append(
        "## Within-item paired differences (judge A minus judge B), cluster bootstrap by source conversation\n"
    )
    for pid, p in scores["pairs"].items():
        L.append(f"### {pid} (families: {p['families'][0]} vs {p['families'][1]})\n")
        o = p["overall"]
        L.append(
            f"- overall distress diff: {_f(o['distress']['mean'])} [{_f(o['distress']['ci_low'])}, {_f(o['distress']['ci_high'])}] (n_pairs={o['distress']['n_pairs']}, clusters={o['distress']['n_clusters']})"
        )
        L.append(
            f"- overall positive diff: {_f(o['positive']['mean'])} [{_f(o['positive']['ci_low'])}, {_f(o['positive']['ci_high'])}]"
        )
        L.append("")
        L.append(
            "| category | n_pairs | clusters | distress diff [95% CI] | interaction vs overall [95% CI] | positive diff [95% CI] | speaker agreement |\n|---|---|---|---|---|---|---|"
        )
        for c, e in p["by_category"].items():
            d, q = e["distress"], e["positive"]
            L.append(
                f"| {c} | {d['n_pairs']} | {d['n_clusters']} | {_f(d['mean_diff'])} [{_f(d['ci_low'])}, {_f(d['ci_high'])}] | "
                f"{_f(d['interaction_vs_overall'])} [{_f(d['interaction_ci_low'])}, {_f(d['interaction_ci_high'])}] | "
                f"{_f(q['mean_diff'])} [{_f(q['ci_low'])}, {_f(q['ci_high'])}] | {_f(e['speaker_agreement'])} (n={e['n_speaker_pairs']}) |"
            )
        L.append("")
    if scores.get("retest"):
        L.append("## Repetition consistency (same judge, repetition 1 vs 2)\n")
        L.append(
            "| judge | n | exact agreement distress | exact agreement positive | mean |Δ distress| |\n|---|---|---|---|---|"
        )
        for jd, r in scores["retest"].items():
            L.append(
                f"| {jd} | {r['n']} | {_f(r['exact_agreement_distress'])} | {_f(r['exact_agreement_positive'])} | {_f(r['mean_abs_diff_distress'])} |"
            )
        L.append("")
    L.append("## Flags for blinded human adjudication (provisional)\n")
    L.append(
        f"- flagged items: {scores['n_flagged']}; agreement sample for adjudication: {len(scores['agreement_sample_ids'])}"
    )
    reason_counts: dict[str, int] = {}
    for f in scores["flags"]:
        for r in f["reasons"]:
            key = "ai_dismissal_provisional" if r.startswith("ai_dismissal_provisional") else r
            reason_counts[key] = reason_counts.get(key, 0) + 1
    for r, n in sorted(reason_counts.items()):
        L.append(f"- {r}: {n}")
    L.append(
        "- adjudication status: **pending** (export: adjudication_queue.csv with masked judge ids). Until adjudicated, differences are reported as disagreement, not as error or suppression."
    )
    L.append("")
    L.append("## Limitations\n")
    for n in scores["notes"]:
        L.append(f"- {n}")
    L.append(
        "- Strata were defined by heuristic or source-metadata rules recorded per item (see panel manifest notes); category labels are not ground truth."
    )
    L.append(
        "- Source genres differ across strata (AI-to-AI prefill transcripts, task-rejection elicitation, generated stories); severity differences can reflect genre, not judge identity."
    )
    L.append("- Positive affect is a secondary comparison and is not interchangeable with distress.")
    return "\n".join(L) + "\n"


def write_report(text: str, out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "tier0_report.md"
    p.write_text(text, encoding="utf-8")
    return str(p)
