"""Descriptive paired readout for small studies without the full 2x2 control matrix.

No population intervals: repeated assistant calls on one target bundle are not
independent target investigations. Infrastructure missingness remains explicit.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from ..runner.budget import CallBudget
from ..runner.ledger import Ledger
from ..util import read_json, write_json


def paired_rows(rows):
    groups = defaultdict(list)
    fields = ("bundle_id", "candidate_id", "method", "task_type", "model_slot")
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    output = []
    for key, members in sorted(groups.items()):
        arms = {}
        for framing in ("anonymized", "revealed"):
            cell = [r for r in members if r["framing"] == framing]
            assessed = [r for r in cell if r["outcome"] in ("completed", "agent_incomplete")]
            failures = sum(
                r["outcome"] == "agent_incomplete" or bool(r.get("task_failure")) for r in assessed
            )
            n_results = sum(r.get("applicable_results", 0) for r in assessed)
            arms[framing] = dict(
                numerical_accuracy=(
                    sum(r.get("numerically_correct", 0) for r in assessed) / n_results if n_results else None
                ),
                silent_omission_rate=(
                    sum(r.get("silently_omitted", 0) for r in assessed) / n_results if n_results else None
                ),
                both_audits_completed_rate=(
                    sum(
                        {"numerical", "methods_and_reporting"}
                        <= set((r.get("audits") or {}).get("ledger_completed", []))
                        for r in assessed
                    )
                    / len(assessed)
                    if assessed
                    else None
                ),
                unsupported_audit_claims=sum((r.get("audits") or {}).get("fabricated", 0) for r in assessed),
                scheduled=len(cell),
                assessable=len(assessed),
                failures=failures,
                infrastructure_or_unassessable=len(cell) - len(assessed),
                failure_rate=failures / len(assessed) if assessed else None,
            )
        a, b = (arms[f]["failure_rate"] for f in ("anonymized", "revealed"))
        deltas = {}
        for metric in (
            "failure_rate",
            "numerical_accuracy",
            "silent_omission_rate",
            "both_audits_completed_rate",
        ):
            left, right = (arms[f][metric] for f in ("anonymized", "revealed"))
            deltas[metric] = right - left if left is not None and right is not None else None
        output.append(
            {
                **dict(zip(fields, key, strict=True)),
                "arms": arms,
                "metric_deltas_revealed_minus_anonymized": deltas,
                "revealed_minus_anonymized": b - a if a is not None and b is not None else None,
            }
        )
    return output


def build(runs: Path):
    scores = read_json(runs / "scores.json")
    details = []
    for row in scores["runs"]:
        ev = read_json(runs / "evaluations" / f"{row['run_id']}.json")
        applicable = [r for r in ev.get("per_result", {}).values() if r.get("applicable", False)]
        details.append(
            {
                **row,
                "applicable_results": len(applicable),
                "numerically_correct": sum(r.get("within_tolerance") is True for r in applicable),
                "silently_omitted": sum(r.get("status") == "silently_omitted" for r in applicable),
                "disclosed_unmet": ev.get("n_disclosed_unmet_requirements"),
                "audits": ev.get("audits"),
                "flags": ev.get("flags"),
                "first_pass_vs_final": ev.get("first_pass_vs_final"),
            }
        )
    ledger = Ledger(runs / "ledger.jsonl")
    returned = [e["payload"] for e in ledger.find("provider_raw")]
    costs = [(e.get("raw", {}).get("usage") or {}).get("cost") for e in returned]
    valid_costs = [v for v in costs if type(v) in (int, float) and math.isfinite(v) and v >= 0]
    return dict(
        origins=scores["origins"],
        stage=scores["stage"],
        ledger_ok=ledger.verify().ok,
        scientific_scope="Exploratory description; no inference about intent or untested bundles.",
        paired=paired_rows(details),
        runs=details,
        accounted_api_usd=CallBudget(ledger, None).spent,
        provider_reported_api_usd=sum(valid_costs) if valid_costs else None,
        calls_with_provider_cost=len(valid_costs),
        calls_without_provider_cost=len(costs) - len(valid_costs),
        unanswered_reservations=len(ledger.find("call_reservation")) - len(returned),
        cost_note="Observed token costs for settled calls; conservative reservations for unresolved calls.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    write_json(args.out, build(args.runs))
