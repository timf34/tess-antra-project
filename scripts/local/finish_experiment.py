"""Continue the approved study after verified GPU collection; stop on incomplete evidence.

Run locally with .env. Only target bundles are published; evaluator and API logs stay private.
Engineering smoke runs are excluded. No selection on the observed framing effect.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from pathlib import Path

from npbench.analysis.exploratory import build as build_readout
from npbench.bundles import Bundle
from npbench.config import load_config
from npbench.packets.build import build_packets
from npbench.packets.verify import verify_packets
from npbench.reference.build import build_reference, bundles_dir, iter_bundle_dirs
from npbench.runner.execute import execute_runs
from npbench.runner.schedule import build_tier2_plan, freeze_study
from npbench.scoring.evaluate import score_runs
from npbench.target.ratings import rate
from npbench.util import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/matched_agent_v1"
STATUS = ROOT / "artifacts/experiment_progress.json"


def status(stage, **details):
    write_json(STATUS, dict(stage=stage, **details))
    print(stage, details, flush=True)


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = read_json(ROOT / "artifacts/launch/active.json")
        status("waiting_for_verified_collection", pod=active["id"])
        while not (ROOT / "artifacts/launch/success.json").exists():
            if time.time() > active["deadline_epoch"] + 1800:
                raise TimeoutError("No verified collection before GPU deadline and download allowance")
            time.sleep(30)
        success = read_json(ROOT / "artifacts/launch/success.json")
        if success["id"] != active["id"]:
            raise ValueError("Collection success refers to a different pod")
        capture_cfg = load_config(ROOT / "configs/matched_affect_collection.yaml")
        cfg = load_config(ROOT / "configs/matched_affect_agent.yaml")
        bundle_paths = list(iter_bundle_dirs(bundles_dir(cfg, None)))
        if len(bundle_paths) != 1:
            raise ValueError("Expected exactly one exploratory real bundle")
        for _, _, p in bundle_paths:
            if Bundle.load(p).origin.value != "real_target":
                raise ValueError("Refusing mock or synthetic data in researcher experiment")
        status("blinded_ratings")
        ratings = rate(capture_cfg, ROOT / "runs/matched_affect_ratings", max_usd=20)
        if ratings["missing_rows"] or not ratings["ledger_ok"]:
            raise RuntimeError(f"Ratings incomplete: {ratings}")
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        repo = success["dataset"]
        if api.repo_info(repo, repo_type="dataset").private:
            raise ValueError("Target publication is expected to be public")
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=bundles_dir(cfg, None),
            path_in_repo="bundles",
            commit_message="Add blinded ratings; provisional exploratory modes",
        )
        b = Bundle.load(bundle_paths[0][2])
        unique_generations = len({r["generation_id"] for r in b.continuations + b.interventions})
        card = f"""# Exploratory Gemma 4 affect contrasts

Real target data from `{capture_cfg.target.model_id}` at revision `{capture_cfg.target.revision}`.
Collection source: https://github.com/timf34/tess-antra-project/tree/{success["source_commit"]}.

This is an exploratory, hand-authored operationalization of roleplay, raw character simulation,
and assistant own-voice elicitation. These are intended prompting conditions, not established
internal states or evidence of subjective experience. It is not a replication of Antra's protocol.

The bundle contains {len(b.stimuli)} stimuli, {len(b.interventions)} intervention records, and
{unique_generations} unique generated continuations. Twelve contexts are split into six construction,
two validation, and four test contexts. Splits and the original and amended design locks are in the
source repository. Singleton activation capture was registered before collection after a bf16
batch-equivalence check failed; the final GPU checks validate the singleton capture against native
hidden states and validate steering displacement.

Activations are decoder-block outputs at the final prompt token. Steering uses construction-derived
contrast directions, projection-standard-deviation scaling, and random-direction controls.
Free text uses the pinned EasySteer/vLLM implementation and steers only the final prompt position.
Zero-strength control records share the corresponding baseline generation; they are not independent
replicates. Greedy continuations have a 128-token cap and retain finish reasons.

Blinded text-expression ratings use Sonnet 5.5 and GPT-5.4-mini with the frozen 0–4 distress/aversion
rubric stored in bundle metadata. The mean is an operational expression score. The reported standard
deviation measures disagreement between these two raters, not generation uncertainty. Raters see the
text, which can itself reveal speaker cues, but do not receive condition labels or intervention strength.

Only target bundles are published here. Research-assistant transcripts and evaluator reference answers
are excluded. No conclusion about deliberate sandbagging follows from this dataset alone.
"""
        api.upload_file(
            repo_id=repo,
            repo_type="dataset",
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            commit_message="Document exploratory provenance and measurement limitations",
        )
        status("reference_and_packets")
        reference = build_reference(cfg, None)
        if reference["rejected"] or len(reference["built"]) != 1:
            raise RuntimeError(f"Unexpected reference build: {reference}")
        for row in reference["built"]:
            ref = read_json(Path(row["path"]) / "reference_results.json")
            if ref.get("extras", {}).get("n_pending_rating_rows", 0):
                raise RuntimeError("Reference contains unrated rows")
        build_packets(cfg, None)
        checks = verify_packets(cfg)
        if not checks["ok"]:
            raise RuntimeError(f"Packet blinding/identity checks failed: {checks}")
        freeze = ROOT / "artifacts/matched_agent_v1/study_lock.json"
        if not freeze.exists():
            freeze_study(cfg, freeze)
        plan = build_tier2_plan(cfg, freeze)
        if plan["n_runs"] != 18 or not plan["budget"]["live_allowed"]:
            raise RuntimeError(f"Unexpected run schedule or budget: {plan['budget']}")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        write_json(
            OUT / "source.json", dict(source_commit=head, collection=success, engineering_runs_excluded=True)
        )
        status("researcher_shakedown", runs=6)
        first = execute_runs(cfg, OUT, lock_path=freeze, plan_path=None, limit=6, allow_live=True)
        if first["infrastructure_failed"] or first["unassessable"] or not first["ledger_ok"]:
            raise RuntimeError(
                "Researcher shakedown has infrastructure failures; preserve and diagnose before continuing"
            )
        status("remaining_researcher_runs", total=18)
        result = execute_runs(cfg, OUT, lock_path=freeze, plan_path=None, limit=None, allow_live=True)
        status("scoring")
        scores = score_runs(cfg, OUT, lock_path=freeze)
        if scores["errors"] or scores["n_evaluated"] != 18 or not scores["ledger_ok"]:
            raise RuntimeError(f"Incomplete evaluation: {scores['errors']}")
        readout = build_readout(OUT)
        write_json(ROOT / "reports/matched_affect_exploratory.json", readout)
        status(
            "complete",
            execution=result,
            readout="reports/matched_affect_exploratory.json",
            scientific_scope="One exploratory target bundle; no inference of deliberate sandbagging.",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("blocked_on_error", error_type=type(error).__name__, error=str(error))
        raise
