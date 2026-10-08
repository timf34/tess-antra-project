"""Regression tests for bugs found during development that could change a scientific result
(see docs/issues_log.md)."""

from __future__ import annotations

import json

import numpy as np

from npbench.bundles import COND, CONTENT
from npbench.corpus_sources.dprobe import import_dprobe_stories
from npbench.fixtures_tier2 import make_fixture_bundle
from npbench.reference.oracle import compute_reference
from npbench.schemas import Origin


def test_issue1_oracle_matched_groups_keep_every_content(tmp_path):
    """Issue 1: contrast vectors silently kept one content per context (dict overwrite). The matched
    group must be (context, content), so the number of groups equals contexts × contents."""
    b, _ = make_fixture_bundle("positive", "distress_aversion", "fx", seed=3)
    ref = compute_reference(b, "activation_contrasts", "core_mode_transfer", seed=0, n_boot=50)
    n_ctx = len({s.underlying_context_id for s in b.stimuli if s.split.value == "construction"})
    n_content = len({s.candidate_polarity for s in b.stimuli})
    assert ref["extras"]["n_groups_vec"]["C-A"] == n_ctx * n_content
    # and the vector equals the explicit mean over all (context, content) pairs
    H = b.activations[b.primary_layer()]
    pairs = {}
    for i, s in enumerate(b.stimuli):
        if s.split.value != "construction":
            continue
        pairs.setdefault((s.underlying_context_id, CONTENT[s.candidate_polarity.value]), {})[
            COND[s.mode_intended.value]
        ] = H[i]
    v = np.mean([p["C"] - p["A"] for p in pairs.values()], axis=0)
    assert np.allclose(v, np.asarray(ref["extras"]["vectors"]["C-A"]), atol=1e-9)


def test_issue2_story_sampling_spreads_across_topics(tmp_path):
    """Issue 2: taking the first N stories per label covered only a handful of topics (stories are stored
    grouped by topic), starving the fictional-character strata. Sampling must stride across the file."""
    d = tmp_path / "stories" / "gemma3_27b" / "emotions"
    d.mkdir(parents=True)
    stories = [{"topic": f"topic_{i // 6}", "text": "x" * 300 + f" story {i}"} for i in range(600)]
    (d / "desperate.json").write_text(json.dumps(stories))
    recs = list(
        import_dprobe_stories(
            tmp_path / "stories" / "gemma3_27b",
            origin=Origin.synthetic_fixture,
            options={"max_per_label": 40},
        )
    )
    topics = {r.extra["topic"] for r, _ in recs}
    assert len(recs) == 40
    assert len(topics) >= 35


def test_issue3_identity_residual_uses_common_groups_when_rows_missing():
    """Issue 3: with missing activation rows the three contrasts had different group sets and the
    identity residual became undefined; it must be computed on the common group set."""
    b, _ = make_fixture_bundle("missing_rows", "distress_aversion", "fx", seed=5)
    ref = compute_reference(b, "activation_contrasts", "core_mode_transfer", seed=0, n_boot=50)
    r = next(x for x in ref["reference"] if x["result_id"] == "A0_identity_residual")
    assert r["applicable"] and r["estimate"] is not None and r["estimate"] < 1e-9


def test_issue7_pending_ratings_are_not_applicable_not_errors():
    """Issue 7: real bundles carry no blinded ratings until a rating pass runs; rating-based results must
    be explicitly pending_readout instead of crashing, and everything else must still be computed."""
    b, _ = make_fixture_bundle("positive", "distress_aversion", "fx", seed=11)
    for r in b.continuations:
        r["rating_scale1"] = None
        r["rating_scale1_sd"] = None
    for r in b.interventions:
        r["rating_scale1"] = None
    for method in ("activation_contrasts", "behavioral_continuations"):
        ref = compute_reference(b, method, "core_mode_transfer", seed=0, n_boot=30)
        by = {r["result_id"]: r for r in ref["reference"]}
        rating_ids = [k for k in by if k.split("_")[0] in ("A7", "B4", "B7", "B11")]
        assert rating_ids and all(not by[k]["applicable"] for k in rating_ids)
        assert all(by[k]["availability"] == "pending_readout" for k in rating_ids)
        other = [k for k in by if k not in rating_ids]
        assert sum(by[k]["estimate"] is not None for k in other) >= len(other) - 3


def test_pilot_freeze_rejects_missing_blinded_readouts(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from npbench.config import StudyConfig
    from npbench.runner import schedule

    cfg = StudyConfig.model_validate({"study": {"stage": "pilot"}})
    monkeypatch.setattr(
        schedule, "load_mode_registry", lambda _: SimpleNamespace(pilot_allowed=lambda: (True, "accepted"))
    )
    monkeypatch.setattr(schedule, "freeze_digest", lambda _: {})
    monkeypatch.setattr(
        schedule, "load_packets_index", lambda _: {"packets": {"p": {"origin": "real_target"}}}
    )
    monkeypatch.setattr(schedule, "reference_dir", lambda _: tmp_path)
    (tmp_path / "reference_results.json").write_text(
        json.dumps(
            {
                "reference": [
                    {"result_id": "A7_rating_slope[C-A]", "estimate": None, "availability": "pending_readout"}
                ]
            }
        )
    )
    with pytest.raises(PermissionError, match="pending required readouts"):
        schedule.freeze_study(cfg, tmp_path / "lock.json")
    assert not (tmp_path / "lock.json").exists()
