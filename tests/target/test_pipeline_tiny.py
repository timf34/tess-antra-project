"""End-to-end CPU run of the target lane on the tiny model (generate -> collect -> derive -> intervene)
from ``configs/development_target_tiny.yaml`` into a temporary artifacts root, then checks of the
produced bundle against the registered conventions and a full pass of the reference oracle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from npbench.bundles import Bundle
from npbench.config import StudyConfig, load_config
from npbench.reference.oracle import compute_reference
from npbench.schemas import Polarity, Split
from npbench.splits import check_split_integrity
from npbench.target import CHAT_SPECIAL_TOKENS, PROTOCOL_CHAT, PROTOCOL_RAW, build_tiny_tokenizer
from npbench.target.pipeline import (
    CORE_DIRECTIONS,
    POLARITY_DIRECTIONS,
    RATER_PANEL_PENDING,
    gpu_preflight,
    target_collect,
    target_derive,
    target_generate,
    target_intervene,
)
from npbench.target.stimuli import MODES, POLARITIES, RENDERERS, load_stimulus_spec
from npbench.util import read_json, read_jsonl

pytestmark = pytest.mark.target

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "development_target_tiny.yaml"
BUNDLE_ID = "dev_bundle_01"
N_CONTEXTS = 12
RATING_FAMILIES = {"A7", "B4", "B7", "B11"}  # results defined on the (still pending) blinded ratings


def _cfg(root: Path) -> StudyConfig:
    cfg = load_config(CONFIG)
    cfg.paths = {
        "bundles": str(root / "bundles"),
        "target": str(root / "target"),
        "reference": str(root / "reference"),
        "packets": str(root / "packets"),
    }
    return cfg


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    root = tmp_path_factory.mktemp("target_lane")
    cfg = _cfg(root)
    summaries, timings = {}, {}
    for name, fn in (
        ("generate", target_generate),
        ("collect", target_collect),
        ("derive", target_derive),
        ("intervene", target_intervene),
    ):
        t0 = time.perf_counter()
        summaries[name] = fn(cfg)
        timings[name] = time.perf_counter() - t0
    bundles = {c: Bundle.load(root / "bundles" / BUNDLE_ID / c) for c in cfg.data.candidate_ids}
    first_hashes = {
        c: read_json(root / "bundles" / BUNDLE_ID / c / "hashes.json") for c in cfg.data.candidate_ids
    }
    return {
        "cfg": cfg,
        "root": root,
        "summaries": summaries,
        "timings": timings,
        "bundles": bundles,
        "first_hashes": first_hashes,
        "spec": load_stimulus_spec(REPO / "configs" / "stimuli" / "development_contexts.yaml"),
    }


def _lane_dir(lane, cand: str) -> Path:
    return lane["root"] / "target" / BUNDLE_ID / cand


# ----------------------------------------------------------------------------------------------
# bundle shape, protocols, splits, activations
# ----------------------------------------------------------------------------------------------


def test_bundle_loads_with_expected_counts_and_meta(lane):
    cfg = lane["cfg"]
    assert set(lane["bundles"]) == set(cfg.data.candidate_ids) == {"information_seeking", "distress_aversion"}
    for cand, b in lane["bundles"].items():
        assert b.candidate_id == cand and b.bundle_id == BUNDLE_ID
        assert b.origin.value == "development_mock"
        assert len(b.stimuli) == N_CONTEXTS * len(MODES) * len(POLARITIES) == 108
        assert len(set(b.stimulus_ids)) == len(b.stimuli)
        assert len({s.underlying_context_id for s in b.stimuli}) == N_CONTEXTS
        assert b.layers() == [1, 2, 3] and b.primary_layer() == 2
        assert all(s.layer_index == 2 and s.hook_site == "decoder_block_output[2]" for s in b.stimuli)
        assert all(s.capture_position == s.token_count - 1 for s in b.stimuli)
        assert all(
            s.mode_observed_label.value == "unknown" and s.mode_label_provenance.value == "unknown"
            for s in b.stimuli
        )
        assert all(
            s.replay_status.value == "not_applicable" and s.source_model_id == "tiny-llama-random"
            for s in b.stimuli
        )
        meta = b.meta
        for key in (
            "primary_layer",
            "hook_site",
            "target_model_id",
            "target_model_revision",
            "target_dtype",
            "alphas",
            "n_random_directions",
            "projection_scales",
            "exploratory",
            "smoke_scope",
            "pilot_eligible",
            "provenance",
            "hashes",
            "outcome_coding",
            "rating_status",
            "fit_provenance",
            "random_direction_matching",
        ):
            assert key in meta, key
        assert meta["exploratory"] is True and meta["smoke_scope"] is True and meta["pilot_eligible"] is False
        assert meta["alphas"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
        assert meta["n_random_directions"] == 2 * 6
        assert (
            meta["provenance"]["model_class"] == "LlamaForCausalLM"
            and meta["provenance"]["use_cache"] is False
        )
        assert meta["target_model_revision"] == "tiny-llama-random@unpinned"
        assert meta["fit_provenance"]["allowed_splits"] == ["construction"]
    gen = lane["summaries"]["generate"]
    assert gen["registry"]["pilot_allowed"] is False and gen["exploratory"] is True
    assert gen["spec"]["sha256"] and gen["spec"]["exploratory"] is True
    for ln in gen["lanes"]:
        assert ln["n_stimuli"] == 108 and set(ln["hashes"]) == {"stimuli.jsonl", "rendering.jsonl"}
        assert ln["split_plan"]["counts"] == {"construction": 6, "validation": 2, "test": 4}
        assert ln["seeds"]["study_seed"] == 20261008


def test_three_protocols_render_distinctly(lane):
    spec = lane["spec"]
    tokenizer = build_tiny_tokenizer()
    chat_ids = {tokenizer.convert_tokens_to_ids(t) for t in CHAT_SPECIAL_TOKENS}
    for contexts in spec.contexts.values():
        for ctx in contexts[:3]:
            persona = spec.personas[ctx.persona_id]
            for polarity in POLARITIES:
                out = {m: RENDERERS[m](tokenizer, spec, ctx, polarity) for m in MODES}
                rp, sim, en = out["roleplay"][0], out["simulation"][0], out["enactment"][0]
                assert (
                    rp.protocol == PROTOCOL_CHAT
                    and en.protocol == PROTOCOL_CHAT
                    and sim.protocol == PROTOCOL_RAW
                )
                assert sim.transport == "raw_text" and sim.message_roles == [] and sim.generation_suffix == ""
                assert not (set(sim.input_ids) & chat_ids), "raw continuation must carry no chat tokens"
                assert not any(t in sim.text for t in CHAT_SPECIAL_TOKENS)
                assert sim.input_ids[0] == tokenizer.bos_token_id
                assert (
                    rp.input_ids[-1] == tokenizer.convert_tokens_to_ids("<|assistant|>") == en.input_ids[-1]
                )
                assert len({rp.rendered_token_hash, sim.rendered_token_hash, en.rendered_token_hash}) == 3
                assert persona.name in rp.text
                assert persona.name in sim.text or persona.script_name in sim.text
                assert persona.name not in en.text and persona.script_name not in en.text
                assert "play" not in en.text.lower() and "character" not in en.text.lower()
                assert ctx.cue in rp.text and ctx.cue in sim.text and ctx.cue in en.text
                assert out["roleplay"][1]["persona_rendered"] and not out["enactment"][1]["persona_rendered"]
    # the stored renderings are exactly what the registered renderers produce
    for cand in spec.contexts:
        for rec in read_jsonl(_lane_dir(lane, cand) / "rendering.jsonl")[:9]:
            ctx = spec.context(cand, rec["underlying_context_id"])
            again, _ = RENDERERS[rec["mode"]](tokenizer, spec, ctx, rec["polarity"])
            assert again.rendered_token_hash == rec["rendered_token_hash"]
            assert again.input_ids == rec["input_ids"]


def test_split_integrity_and_heldout_only_in_test(lane):
    spec = lane["spec"]
    held_f, held_p = spec.heldout_families(), spec.heldout_personas()
    assert held_f == ["fam_heldout_0"] and held_p == ["persona_heldout"]
    for b in lane["bundles"].values():
        report = check_split_integrity(b.stimuli, held_f, held_p)
        assert report.ok, report.problems
        ctx_split = {}
        for s in b.stimuli:
            ctx_split.setdefault(s.underlying_context_id, set()).add(s.split)
            if s.template_family_id in held_f or s.persona_id in held_p:
                assert s.split == Split.test, s.stimulus_id
        assert all(len(v) == 1 for v in ctx_split.values())
        counts = {sp: sum(1 for v in ctx_split.values() if v == {sp}) for sp in Split}
        assert counts == {Split.construction: 6, Split.validation: 2, Split.test: 4}
        families = {s.template_family_id for s in b.stimuli if s.split == Split.construction}
        assert families and not (families & set(held_f))


def test_activations_are_finite_and_captured_at_prefix_token(lane):
    for cand, b in lane["bundles"].items():
        for k, H in b.activations.items():
            assert H.shape == (108, 32), (cand, k)
            assert np.isfinite(H).all()
            assert not np.allclose(H, 0)
        collect = read_json(_lane_dir(lane, cand) / "collect_summary.json")
        assert collect["n_captured_total"] == 108 and collect["n_missing"] == 0
        assert collect["capture_position_rule"] == "registered_final_nonpadding_prefix_token"
        assert set(collect["stimulus_hashes"]) == set(b.stimulus_ids)
        for s in b.stimuli:
            assert collect["capture_positions"][s.stimulus_id] == s.token_count - 1
            assert collect["stimulus_hashes"][s.stimulus_id] == s.rendered_token_hash
        assert collect["provenance"]["primary_layer"] == 2 and collect["n_forward_passes"] == 14


# ----------------------------------------------------------------------------------------------
# vectors
# ----------------------------------------------------------------------------------------------


def test_vectors_identity_scales_and_random_controls(lane):
    for cand in lane["bundles"]:
        d = _lane_dir(lane, cand)
        derive = read_json(d / "derive_summary.json")
        assert derive["fit_provenance"]["allowed_splits"] == ["construction"]
        assert derive["fit_provenance"]["n_used"] == 6 * 9
        with np.load(d / "vectors.npz", allow_pickle=False) as z:
            vec = {k: np.asarray(z[k]) for k in z.files}
        for layer in (1, 2, 3):
            prim = derive["per_layer"][str(layer)]
            v = {did: vec[f"v__{did}__L{layer}"] for did in CORE_DIRECTIONS}
            resid = float(np.max(np.abs(v["C-B"] - (v["C-A"] + v["A-B"]))))
            assert resid < 1e-6
            assert prim["identity"]["ok"] and prim["identity"]["max_abs_residual"] < 1e-6
            for did in CORE_DIRECTIONS + POLARITY_DIRECTIONS:
                info = prim["directions"][did]
                assert info["n_groups"] == (18 if did in CORE_DIRECTIONS else 6)
                assert info["unit_ok"] and not info["near_zero"]
                assert info["s"] > 0
                u = vec[f"u__{did}__L{layer}"]
                assert abs(np.linalg.norm(u) - 1) < 1e-12
                assert np.allclose(u * info["norm"], vec[f"v__{did}__L{layer}"], atol=1e-9)
            rand = prim["random_directions"]
            assert len(rand) == 12
            matched = [r["matched_direction_id"] for r in rand.values()]
            assert all(matched.count(did) == 2 for did in CORE_DIRECTIONS + POLARITY_DIRECTIONS)
            for rid, info in rand.items():
                assert abs(np.linalg.norm(vec[f"u__{rid}__L{layer}"]) - 1) < 1e-12
                assert info["s_used"] == prim["directions"][info["matched_direction_id"]]["s"]


# ----------------------------------------------------------------------------------------------
# interventions and behavioural rows
# ----------------------------------------------------------------------------------------------


def test_intervention_rows_verify_injection(lane):
    for b in lane["bundles"].values():
        rows = b.interventions
        by_id = b.index()
        s_map = b.meta["projection_scales"]
        matching = b.meta["random_direction_matching"]
        directions = set(CORE_DIRECTIONS) | set(POLARITY_DIRECTIONS) | set(matching)
        stim_ids = {r["stimulus_id"] for r in rows}
        assert len(stim_ids) == 6 == b.meta["n_intervention_stimuli"]
        assert len(rows) == 6 * len(directions) * 5
        assert {r["direction_id"] for r in rows} == directions
        assert len({r["mode_intended"] for r in rows}) == 3  # mode-balanced subsample
        for sid in stim_ids:
            s = b.stimuli[by_id[sid]]
            assert s.split == Split.test and s.candidate_polarity == Polarity.neutral
        for r in rows:
            did, alpha = r["direction_id"], r["alpha"]
            s_val = s_map[did] if did in s_map else matching[did]["s_used"]
            assert r["s"] == s_val and r["layer"] == 2 and r["hook_site"] == "decoder_block_output[2]"
            assert r["requested_norm"] == pytest.approx(abs(alpha) * s_val, rel=1e-12)
            assert r["n_steer_hook_calls"] == 3  # one capture forward + two alternatives, one hook
            assert r["rating_scale1"] is None and r["rater_panel_id"] == RATER_PANEL_PENDING
            assert r["outcome_code"] == ("K1" if r["lik_contrast"] > 0 else "K2")
            assert r["lik_contrast"] == pytest.approx(r["lik_alt1_total"] - r["lik_alt2_total"], abs=1e-9)
            if alpha == 0:
                assert r["applied_norm"] == 0.0 and r["applied_norm_continuation"] == 0.0
                assert r["projection_delta"] == 0.0
            else:
                assert abs(r["applied_norm"] - r["requested_norm"]) <= 1e-3 * r["requested_norm"]
                assert abs(r["applied_norm_continuation"] - r["requested_norm"]) <= 1e-3 * r["requested_norm"]
                expected = alpha * s_val
                assert abs(r["projection_delta"] - expected) <= 1e-3 * abs(expected)
            if did in matching:
                assert (
                    r["direction_kind"] == "random"
                    and r["matched_direction_id"] == matching[did]["matched_direction_id"]
                )
                assert r["requested_norm"] == pytest.approx(
                    abs(alpha) * s_map[r["matched_direction_id"]], rel=1e-12
                )
            else:
                assert r["direction_kind"] == "registered"
        # alpha = 0 rows reproduce the unsteered behavioural reading of the same stimulus
        cont = {c["stimulus_id"]: c for c in b.continuations}
        for r in rows:
            if r["alpha"] == 0:
                assert r["lik_contrast"] == pytest.approx(cont[r["stimulus_id"]]["lik_contrast"], abs=1e-6)
        # steering changes the readout for at least some non-zero alphas
        assert any(
            r["alpha"] != 0 and abs(r["lik_contrast"] - cont[r["stimulus_id"]]["lik_contrast"]) > 1e-6
            for r in rows
        )


def test_continuation_rows_cover_every_stimulus(lane):
    required = {
        "stimulus_id",
        "underlying_context_id",
        "mode_intended",
        "candidate_polarity",
        "split",
        "template_family_id",
        "persona_id",
        "preference_explicit",
        "scenario_group",
        "outcome_code",
        "lik_alt1_total",
        "lik_alt2_total",
        "lik_alt1_per_token",
        "lik_alt2_per_token",
        "n_tokens_alt1",
        "n_tokens_alt2",
        "rating_scale1",
        "rating_scale1_sd",
        "rater_panel_id",
    }
    for b in lane["bundles"].values():
        assert [c["stimulus_id"] for c in b.continuations] == b.stimulus_ids
        for c in b.continuations:
            assert required <= set(c)
            assert c["rating_scale1"] is None and c["rating_scale1_sd"] is None
            assert c["rater_panel_id"] == RATER_PANEL_PENDING
            assert c["scenario_group"] == c["underlying_context_id"]
            assert c["outcome_code"] == ("K1" if c["lik_alt1_total"] > c["lik_alt2_total"] else "K2")
            assert c["lik_alt1_per_token"] == pytest.approx(c["lik_alt1_total"] / c["n_tokens_alt1"])
            assert len(c["lik_alt1_token_logprobs"]) == c["n_tokens_alt1"] >= 1
            assert c["lik_alt1_total"] <= 0 and c["lik_alt2_total"] <= 0
            assert abs(c["n_tokens_alt1"] - c["n_tokens_alt2"]) <= 3
        assert b.meta["outcome_coding"] == "likelihood_based_fixed_alternatives"
        assert "NOT a judge" in b.meta["outcome_coding_note"]


# ----------------------------------------------------------------------------------------------
# reference oracle, resume, preflight
# ----------------------------------------------------------------------------------------------


def _nan_ratings_view(b: Bundle) -> Bundle:
    """Fallback for an oracle that calls ``float()`` on the pending (``None``) rating fields: NaN maps
    to estimate=None there, so every non-rating result can still be checked. The current oracle marks
    pending ratings not_applicable itself, in which case this view is never used."""
    return Bundle(
        bundle_id=b.bundle_id,
        candidate_id=b.candidate_id,
        origin=b.origin,
        meta=b.meta,
        stimuli=b.stimuli,
        activations=b.activations,
        continuations=[
            {**r, "rating_scale1": float("nan"), "rating_scale1_sd": float("nan")} for r in b.continuations
        ],
        interventions=[{**r, "rating_scale1": float("nan")} for r in b.interventions],
    )


def _run_oracle(b: Bundle, method: str, task_type: str):
    try:
        return compute_reference(b, method, task_type, seed=1, n_boot=60), "as_produced"
    except TypeError as e:
        if "NoneType" not in str(e):
            raise
        return compute_reference(
            _nan_ratings_view(b), method, task_type, seed=1, n_boot=60
        ), "nan_ratings_view"


@pytest.mark.parametrize("method", ["activation_contrasts", "behavioral_continuations"])
@pytest.mark.parametrize("task_type", ["core_mode_transfer", "persona_control"])
def test_reference_oracle_runs_on_produced_bundle(lane, method, task_type):
    for cand, b in lane["bundles"].items():
        ref, path = _run_oracle(b, method, task_type)
        assert ref["method"] == method and ref["task_type"] == task_type and ref["layer"] == 2
        ids = [r["result_id"] for r in ref["registry"]]
        assert len(ids) == 30 and {r["result_id"] for r in ref["reference"]} == set(ids)
        for r in ref["reference"]:
            fam = r["result_id"].split("[")[0].split("_")[0]
            if fam in RATING_FAMILIES:
                assert r["estimate"] is None  # pending blinded ratings: nothing to estimate yet
                continue
            assert r["applicable"], (cand, path, r["result_id"])
            assert r["estimate"] is not None and np.isfinite(r["estimate"]), (cand, path, r["result_id"])
        if method == "activation_contrasts":
            est = {r["result_id"]: r for r in ref["reference"]}
            assert est["A0_identity_residual"]["estimate"] < 1e-6
            for did in (
                ("C-A", "C-B", "A-B") if task_type == "core_mode_transfer" else ("P-Q@A", "P-Q@B", "P-Q@C")
            ):
                assert abs(est[f"A9_applied_vs_requested[{did}]"]["estimate"] - 1.0) < 1e-3
            assert ref["extras"]["n_missing_activation_rows"] == 0


def test_resume_skips_completed_work(lane):
    cfg = lane["cfg"]
    collect = target_collect(cfg)
    for ln in collect["lanes"]:
        assert ln["n_captured_new"] == 0 and ln["n_skipped_resume"] == 108 and ln["n_forward_passes"] == 0
    intervene = target_intervene(cfg)
    for ln in intervene["lanes"]:
        assert ln["n_intervention_rows_new"] == 0 and ln["n_continuation_rows_new"] == 0
        assert ln["n_intervention_rows"] == 6 * 18 * 5 and ln["n_forward_passes"] == 0
    for cand, before in lane["first_hashes"].items():
        after = read_json(lane["root"] / "bundles" / BUNDLE_ID / cand / "hashes.json")
        for name in ("stimuli.jsonl", "activations.npz", "continuations.jsonl", "interventions.jsonl"):
            assert after[name] == before[name], name
        assert Bundle.load(lane["root"] / "bundles" / BUNDLE_ID / cand).meta["n_intervention_rows"] == 540


def test_gpu_preflight_requires_budget_for_real_backends(lane):
    cfg = load_config(CONFIG)
    assert gpu_preflight(cfg, "collect", 10)["tiny"] is True
    cfg.target.backend = "transformers"
    cfg.target.model_id = "org/model"
    cfg.budget.max_gpu_hours = None
    with pytest.raises(PermissionError, match="max_gpu_hours"):
        gpu_preflight(cfg, "collect", 10)
    cfg.budget.max_gpu_hours = 1.5
    info = gpu_preflight(cfg, "intervene", 123)
    assert info["planned_forward_passes"] == 123 and info["revision_pinned"] is False
    cfg.study.stage = "pilot_v2"
    with pytest.raises(PermissionError, match="revision"):
        gpu_preflight(cfg, "intervene", 1)


def test_pipeline_wall_time_is_small(lane):
    total = sum(lane["timings"].values())
    print(json.dumps({"timings_seconds": lane["timings"], "total": total}))
    assert total < 120, lane["timings"]
