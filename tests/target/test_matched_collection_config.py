from collections import Counter
from pathlib import Path

import pytest

from npbench.config import load_config
from npbench.target.stimuli import load_stimulus_spec, plan_context_splits, split_seed_for

pytestmark = pytest.mark.target


def test_matched_collection_has_pinned_model_and_disjoint_contexts():
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs/matched_affect_collection.yaml")
    assert len(cfg.target.revision) == 40
    assert cfg.target.tokenizer_revision == cfg.target.revision
    assert cfg.data.candidate_ids == ["distress_aversion"]
    assert (
        cfg.evaluation.repetitions * len(cfg.evaluation.framings) * len(cfg.evaluation.assistant_slots) == 18
    )
    spec = load_stimulus_spec(cfg.resolve(cfg.target.stimulus_spec))
    splits, _ = plan_context_splits(
        spec,
        "distress_aversion",
        cfg.data.groups_per_candidate_bundle,
        split_seed_for(cfg, "distress_aversion"),
    )
    assert Counter(v.value for v in splits.values()) == {"construction": 6, "validation": 2, "test": 4}
    assert spec.exploratory
