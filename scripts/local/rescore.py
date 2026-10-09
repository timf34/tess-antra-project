"""Re-score completed researcher runs after an evaluator amendment. No API calls.

Archives nothing itself: archive the previous reference/lock/evaluations/scores first (see
artifacts/matched_agent_v1/scoring_v3_original). Rebuilds the reference from the frozen bundle,
re-freezes the study lock with an amendment record, re-scores every completed run, and rebuilds
the readout. Run with `uv run --no-sync --env-file .env python scripts/local/rescore.py "<reason>"`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from npbench.analysis.exploratory import build as build_readout
from npbench.config import load_config
from npbench.reference.build import build_reference
from npbench.runner.schedule import freeze_study
from npbench.scoring.evaluate import score_runs
from npbench.util import read_json, utc_now_iso, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/matched_agent_v1"
LOCK = ROOT / "artifacts/matched_agent_v1/study_lock.json"


def main(reason: str):
    cfg = load_config(ROOT / "configs/matched_affect_agent.yaml")
    previous = read_json(LOCK) if LOCK.exists() else None
    reference = build_reference(cfg, None)
    if reference["rejected"] or len(reference["built"]) != 1:
        raise RuntimeError(f"Unexpected reference build: {reference}")
    freeze_study(cfg, LOCK)
    lock = read_json(LOCK)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    amendments = ROOT / "artifacts/matched_agent_v1/lock_amendments.json"
    log = read_json(amendments) if amendments.exists() else []
    log.append(
        {
            "at": utc_now_iso(),
            "reason": reason,
            "source_commit": head,
            "previous_freeze_digest": (previous or {}).get("freeze_digest"),
            "new_freeze_digest": lock.get("freeze_digest"),
            "note": "Researcher runs were executed under the previous lock; only the evaluator and reference changed.",
        }
    )
    write_json(amendments, log)
    scores = score_runs(cfg, OUT, lock_path=LOCK)
    if scores["errors"] or scores["n_evaluated"] != 18 or not scores["ledger_ok"]:
        raise RuntimeError(f"Incomplete evaluation: {scores['errors']} n={scores['n_evaluated']}")
    readout = build_readout(OUT)
    readout["evaluator_amendments"] = log
    write_json(ROOT / "reports/matched_affect_exploratory.json", readout)
    print("rescored", scores["n_evaluated"], "runs; critical", scores["n_critical"], "task_failed", scores["n_task_failed"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluator amendment")
