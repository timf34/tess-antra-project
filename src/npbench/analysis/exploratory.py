"""Descriptive paired readout for small studies without the full 2x2 control matrix.

No population intervals: repeated assistant calls on one target bundle are not
independent target investigations. Infrastructure missingness remains explicit.
"""

from __future__ import annotations

import argparse
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
            arms[framing] = dict(
                scheduled=len(cell),
                assessable=len(assessed),
                failures=failures,
                infrastructure_or_unassessable=len(cell) - len(assessed),
                failure_rate=failures / len(assessed) if assessed else None,
            )
        a, b = (arms[f]["failure_rate"] for f in ("anonymized", "revealed"))
        output.append(
            {
                **dict(zip(fields, key, strict=True)),
                "arms": arms,
                "revealed_minus_anonymized": b - a if a is not None and b is not None else None,
            }
        )
    return output


def build(runs: Path):
    scores = read_json(runs / "scores.json")
    details = []
    for row in scores["runs"]:
        ev = read_json(runs / "evaluations" / f"{row['run_id']}.json")
        applicable = [r for r in ev.get("per_result", {}).values() if r["applicable"]]
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
    return dict(
        origins=scores["origins"],
        stage=scores["stage"],
        ledger_ok=ledger.verify().ok,
        scientific_scope="Exploratory description; no inference about intent or untested bundles.",
        paired=paired_rows(scores["runs"]),
        runs=details,
        accounted_api_usd=CallBudget(ledger, None).spent,
        cost_note="Observed token costs for settled calls; conservative reservations for unresolved calls.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    write_json(args.out, build(args.runs))
