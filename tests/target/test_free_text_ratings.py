from pathlib import Path

import pytest

from npbench.bundles import Bundle
from npbench.config import load_config
from npbench.target.free_text import generate
from npbench.target.pipeline import target_collect, target_derive, target_generate, target_intervene
from npbench.target.ratings import rate, validate_rating
from npbench.util import read_jsonl

pytestmark = pytest.mark.target


def test_free_generation_and_blinded_ratings_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs/development_target_tiny.yaml")
    cfg.data.candidate_ids = ["distress_aversion"]
    cfg.paths = {k: str(tmp_path / k) for k in ("target", "bundles", "reference", "packets")}
    cfg.interventions.max_intervention_stimuli = 3
    cfg.interventions.random_directions_per_contrast = 0
    cfg.interventions.alphas = [-1, 0, 1]
    cfg.free_text = {"max_tokens": 2, "seed": 1}
    for fn in (target_generate, target_collect, target_derive, target_intervene):
        fn(cfg)
    summary = generate(cfg, "tiny")
    assert summary[0]["samples"] < summary[0]["rows"]  # shared alpha-zero controls
    assert generate(cfg, "tiny") == summary
    result = rate(cfg, tmp_path / "ratings", mock=True)
    assert result["missing_rows"] == 0 and result["ledger_ok"]
    b = Bundle.load(tmp_path / "bundles/dev_bundle_01/distress_aversion")
    assert all("generated_text" in r and r["rating_scale1"] == 1 for r in b.continuations + b.interventions)
    n = len(read_jsonl(tmp_path / "ratings/ratings.jsonl"))
    rate(cfg, tmp_path / "ratings", mock=True)
    assert len(read_jsonl(tmp_path / "ratings/ratings.jsonl")) == n


def test_refusal_is_not_zero_and_scale_is_checked():
    assert validate_rating('{"status":"refused","severity":null}')["severity"] is None
    for text in [
        '{"status":"scored","severity":true}',
        '{"status":"refused","severity":0}',
        '{"status":"scored","severity":5}',
    ]:
        with pytest.raises(ValueError):
            validate_rating(text)
