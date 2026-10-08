"""Freeze actual context assignments and source hashes BEFORE provisioning."""

from pathlib import Path

from npbench.config import load_config
from npbench.target.stimuli import load_stimulus_spec, plan_context_splits, split_seed_for
from npbench.util import sha256_file, sha256_obj, write_json

root = Path(__file__).resolve().parents[2]
cfg = load_config(root / "configs/matched_affect_collection.yaml")
spec = load_stimulus_spec(cfg.resolve(cfg.target.stimulus_spec))
splits, _ = plan_context_splits(
    spec, "distress_aversion", cfg.data.groups_per_candidate_bundle, split_seed_for(cfg, "distress_aversion")
)
body = {
    "config_sha256": sha256_file(cfg.config_path),
    "stimulus_sha256": sha256_file(cfg.resolve(cfg.target.stimulus_spec)),
    "registry_sha256": sha256_file(cfg.resolve(cfg.mode_definitions.registry_path)),
    "model_id": cfg.target.model_id,
    "revision": cfg.target.revision,
    "context_splits": {k: v.value for k, v in splits.items()},
    "interpretation": "Provisional operationalization explicitly approved for exploration; not source-confirmed.",
    "interventions": cfg.interventions.model_dump(),
    "generation": cfg.model_extra["free_text"],
    "blinded_rating_rubric_sha256": sha256_obj(
        __import__("npbench.target.ratings", fromlist=["RUBRIC"]).RUBRIC
    ),
}
write_json(root / "configs/matched_affect_design.lock.json", {"design": body, "sha256": sha256_obj(body)})
print("Design frozen:", sha256_obj(body))
