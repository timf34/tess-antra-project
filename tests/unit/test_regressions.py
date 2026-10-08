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
