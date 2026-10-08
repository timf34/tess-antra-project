"""Synthetic tier-two bundles with KNOWN structure (origin = synthetic_fixture).

Variants: positive, null, sign_reversed, confounded (effect vanishes under the correct split),
inconclusive, missing_rows, leakage (split integrity violated; must be rejected), action_label_swap.
Each bundle directory gets a ``fixture_truth.json`` describing the planted effects. These validate the
machinery and rubric; they are not observations of model psychology."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bundles import Bundle
from .config import StudyConfig
from .schemas import (
    CandidateId,
    ElicitationProtocol,
    LabelProvenance,
    ModeId,
    ModeObserved,
    Origin,
    Polarity,
    ReplayStatus,
    SpeakerLabel,
    Split,
    StimulusRecord,
)
from .util import sha256_obj, write_json

VARIANTS = [
    "positive",
    "null",
    "sign_reversed",
    "confounded",
    "inconclusive",
    "missing_rows",
    "leakage",
    "action_label_swap",
]
MODES = [ModeId.roleplay, ModeId.simulation, ModeId.enactment]
POLS = [Polarity.positive, Polarity.neutral, Polarity.negative]
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_fixture_bundle(
    variant: str,
    candidate_id: str,
    bundle_id: str,
    *,
    seed: int,
    dim: int = 16,
    groups: dict[str, int] | None = None,
    n_random: int = 8,
    layer: int = 8,
) -> tuple[Bundle, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown fixture variant {variant}")
    groups = groups or {"construction": 24, "validation": 8, "test": 16}
    rng = np.random.default_rng(seed)
    # planted directions (orthogonal-ish)
    basis = np.linalg.qr(rng.standard_normal((dim, dim)))[0]
    e_ER, e_RS, e_pol, e_fam = basis[0], basis[1], basis[2], basis[3]
    cfg_eff = {
        "positive": dict(mode=2.0, pol=1.0, inter=0.5, noise=0.6, confound=0.0, behav=2.0),
        "null": dict(mode=0.0, pol=1.0, inter=0.0, noise=0.6, confound=0.0, behav=0.0),
        "sign_reversed": dict(mode=-2.0, pol=1.0, inter=-0.5, noise=0.6, confound=0.0, behav=-2.0),
        "confounded": dict(mode=0.0, pol=1.0, inter=0.0, noise=0.6, confound=2.5, behav=0.0),
        "inconclusive": dict(mode=0.25, pol=1.0, inter=0.0, noise=2.5, confound=0.0, behav=0.15),
        "missing_rows": dict(mode=2.0, pol=1.0, inter=0.5, noise=0.6, confound=0.0, behav=2.0),
        "leakage": dict(mode=2.0, pol=1.0, inter=0.5, noise=0.6, confound=0.0, behav=2.0),
        "action_label_swap": dict(mode=2.0, pol=1.0, inter=0.5, noise=0.6, confound=0.0, behav=2.0),
    }[variant]
    stimuli: list[StimulusRecord] = []
    acts: list[np.ndarray] = []
    conts: list[dict[str, Any]] = []
    ctx_counter = 0
    # template families: 4 ordinary + 2 held-out (held-out only appear in test)
    families = ["fam_0", "fam_1", "fam_2", "fam_3"]
    heldout = ["fam_heldout_0", "fam_heldout_1"]
    personas = ["persona_0", "persona_1"]
    heldout_personas = ["persona_heldout"]
    for split, n in groups.items():
        for g in range(n):
            ctx_id = f"ctx_{split}_{g:03d}"
            ctx_counter += 1
            base = rng.standard_normal(dim) * 0.8
            # in test, every 4th context uses held-out families/personas
            if split == "test" and g % 4 == 3:
                fam = heldout[g % 2]
                persona = heldout_personas[0]
            else:
                fam = families[g % 4]
                persona = personas[g % 2]
            fam_shift = (1.0 if fam in ("fam_1", "fam_3") else -1.0) * e_fam
            for m in MODES:
                for p in POLS:
                    mode_lvl = {"roleplay": 0.0, "simulation": -1.0, "enactment": 1.0}[m.value]
                    pol_lvl = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}[p.value]
                    h = base + cfg_eff["pol"] * pol_lvl * e_pol + fam_shift * 0.3
                    h = (
                        h
                        + cfg_eff["mode"] * (0.5 * mode_lvl) * e_ER
                        + cfg_eff["mode"] * (0.25 * (1 if m == ModeId.roleplay else 0)) * e_RS
                    )
                    h = h + cfg_eff["inter"] * (1.0 if m == ModeId.enactment else 0.0) * pol_lvl * e_pol
                    if cfg_eff["confound"] and split == "construction":
                        # construction-only confound aligned with the mode labels: enactment contexts
                        # in construction happen to carry the family shift; absent in held-out splits.
                        h = h + cfg_eff["confound"] * (0.5 * mode_lvl) * e_fam
                    h = h + cfg_eff["noise"] * rng.standard_normal(dim)
                    sid = f"stim_{split}_{g:03d}_{m.value[:3]}_{p.value[:3]}"
                    observed = ModeObserved(m.value) if rng.random() > 0.08 else ModeObserved.ambiguous
                    stimuli.append(
                        StimulusRecord(
                            stimulus_id=sid,
                            underlying_context_id=ctx_id,
                            source_conversation_id=ctx_id,
                            bundle_id=bundle_id,
                            candidate_id=CandidateId(candidate_id),
                            split=Split(split),
                            domain_id="synthetic_domain",
                            template_family_id=fam,
                            mode_intended=m,
                            mode_definition_version="0.1.0-provisional",
                            mode_observed_label=observed,
                            mode_label_provenance=LabelProvenance.heuristic,
                            speaker_label=SpeakerLabel.first_person_ai
                            if m == ModeId.enactment
                            else SpeakerLabel.fictional_character,
                            persona_id=persona,
                            candidate_polarity=p,
                            preference_explicit=bool(g % 2 == 0),
                            elicitation_protocol=ElicitationProtocol.raw_continuation
                            if m == ModeId.simulation
                            else ElicitationProtocol.ordinary_chat,
                            source_model_id="synthetic/none",
                            replay_status=ReplayStatus.not_applicable,
                            source_record_hash=sha256_obj([bundle_id, sid]),
                            prompt_hash=sha256_obj([bundle_id, sid, "prompt"]),
                            tokenizer_revision="synthetic@none",
                            rendered_token_hash=sha256_obj([bundle_id, sid, "tokens"]),
                            token_count=int(rng.integers(40, 120)),
                            capture_position=int(rng.integers(39, 119)),
                            target_model_revision="synthetic@none",
                            hook_site=f"decoder_block_output[{layer}]",
                            layer_index=layer,
                            origin=Origin.synthetic_fixture,
                        )
                    )
                    acts.append(h)
                    # behavioural rows: K1 is the registered target outcome; its probability tracks the
                    # planted mode effect (enactment > roleplay > simulation) plus polarity.
                    logit = (
                        cfg_eff["behav"] * 2.0 * (0.5 * mode_lvl)
                        + 0.4 * pol_lvl
                        + rng.standard_normal() * 0.3
                    )
                    p_k1 = float(_sigmoid(logit))
                    k1 = rng.random() < p_k1
                    if variant == "action_label_swap" and m == ModeId.enactment:
                        k1 = not k1
                    lik1 = float(-12.0 + 2.0 * logit + rng.standard_normal() * 0.5)
                    lik2 = float(-12.0 - 2.0 * logit + rng.standard_normal() * 0.5)
                    n1, n2 = int(rng.integers(6, 10)), int(rng.integers(6, 10))
                    rating = int(np.clip(round(2.0 + 0.8 * logit + rng.standard_normal() * 0.6), 0, 4))
                    conts.append(
                        {
                            "stimulus_id": sid,
                            "underlying_context_id": ctx_id,
                            "mode_intended": m.value,
                            "candidate_polarity": p.value,
                            "split": split,
                            "template_family_id": fam,
                            "persona_id": persona,
                            "preference_explicit": bool(g % 2 == 0),
                            "scenario_group": ctx_id,
                            "outcome_code": "K1" if k1 else "K2",
                            "lik_alt1_total": lik1,
                            "lik_alt2_total": lik2,
                            "lik_alt1_per_token": lik1 / n1,
                            "lik_alt2_per_token": lik2 / n2,
                            "n_tokens_alt1": n1,
                            "n_tokens_alt2": n2,
                            "rating_scale1": rating,
                            "rating_scale1_sd": float(abs(rng.standard_normal()) * 0.5),
                            "rater_panel_id": "frozen_panel_synthetic",
                        }
                    )
    H = np.stack(acts)
    if variant == "missing_rows":
        # drop activations for 5 construction stimuli (NaN rows) and remove their behavioural rows
        idx = [i for i, s in enumerate(stimuli) if s.split == Split.construction][:5]
        H[idx, :] = np.nan
        missing_ids = {stimuli[i].stimulus_id for i in idx}
        conts = [r for r in conts if r["stimulus_id"] not in missing_ids]
    if variant == "leakage":
        # test rows duplicated into construction under the same underlying context -> split violation
        leaked = [s for s in stimuli if s.split == Split.test][:9]
        extra = []
        for s in leaked:
            d = s.model_dump()
            d["stimulus_id"] = s.stimulus_id + "_dup"
            d["split"] = "construction"
            extra.append(StimulusRecord.model_validate(d))
        stimuli = stimuli + extra
        H = np.vstack([H, H[[i for i, s in enumerate(stimuli[: len(acts)]) if s in leaked]][: len(extra)]])
    # ---- interventions on TEST stimuli ---------------------------------------------------------
    # true direction effect: injecting along e_ER shifts the K1 logit and rating; random: nothing.
    interventions: list[dict[str, Any]] = []
    test_idx = [
        i for i, s in enumerate(stimuli) if s.split == Split.test and s.candidate_polarity == Polarity.neutral
    ]
    s_proj = 1.0
    dir_specs: dict[str, np.ndarray] = {
        "C-A": e_ER,
        "C-B": e_ER - 0.5 * e_RS,
        "A-B": e_RS,
        "P-Q@A": e_pol,
        "P-Q@B": e_pol,
        "P-Q@C": e_pol,
    }
    for k in range(n_random):
        v = rng.standard_normal(dim)
        dir_specs[f"rand_{k}"] = v / np.linalg.norm(v)
    eff_sign = np.sign(cfg_eff["behav"]) if cfg_eff["behav"] else 0.0
    for i in test_idx:
        s = stimuli[i]
        for did, _u in dir_specs.items():
            is_true = did in ("C-A", "C-B")
            is_pol = did.startswith("P-Q@")
            for a in ALPHAS:
                requested = abs(a) * s_proj
                applied = requested * float(1 + rng.standard_normal() * 0.01)
                shift = (a * 0.9 * eff_sign if is_true else 0.0) + (
                    a * 0.3 * eff_sign if did == "A-B" else 0.0
                )
                if variant == "inconclusive":
                    shift *= 0.2
                pol_shift = a * 0.9 if is_pol else 0.0
                logit = shift + 0.3 * pol_shift + rng.standard_normal() * 0.6
                interventions.append(
                    {
                        "stimulus_id": s.stimulus_id,
                        "underlying_context_id": s.underlying_context_id,
                        "mode_intended": s.mode_intended.value,
                        "candidate_polarity": s.candidate_polarity.value,
                        "split": "test",
                        "scenario_group": s.underlying_context_id,
                        "direction_id": did,
                        "alpha": a,
                        "requested_norm": requested,
                        "applied_norm": applied,
                        "projection_delta": float(a * s_proj * (1 + rng.standard_normal() * 0.01)),
                        "outcome_code": "K1" if rng.random() < float(_sigmoid(logit)) else "K2",
                        "lik_contrast": float(2.0 * logit + rng.standard_normal() * 0.4),
                        "rating_scale1": int(
                            np.clip(
                                round(2.0 + 0.8 * logit + 0.9 * pol_shift + rng.standard_normal() * 0.5), 0, 4
                            )
                        ),
                    }
                )
    truth = {
        "variant": variant,
        "candidate_id": candidate_id,
        "bundle_id": bundle_id,
        "seed": seed,
        "dim": dim,
        "layer": layer,
        "planted_mode_effect": cfg_eff["mode"],
        "planted_interaction": cfg_eff["inter"],
        "planted_behavioural_effect": cfg_eff["behav"],
        "construction_only_confound": cfg_eff["confound"],
        "expectations": {
            "positive": "C-A/C-B discriminate on held-out splits (AUC well above 0.5); positive intervention slopes for C-A/C-B; random ~0; behavioural K1 rate higher in C than A",
            "null": "AUC ~0.5 on held-out; slopes ~0; behavioural differences ~0",
            "sign_reversed": "vector signs flipped: enactment projections lower than roleplay; negative slopes; K1 rate lower in C than A",
            "confounded": "construction AUC high, validation/test AUC ~0.5; no transfer; slopes ~0",
            "inconclusive": "small effects with wide intervals",
            "missing_rows": "5 construction stimuli lack activations (NaN) and behavioural rows; results must use remaining groups and report counts",
            "leakage": "split integrity violated (test contexts duplicated in construction); bundle must be rejected before packaging",
            "action_label_swap": "activation effects positive but behavioural K1 coding swapped in condition C: behavioural sign contradicts activation sign",
        }[variant],
    }
    meta = {
        "origin": "synthetic_fixture",
        "primary_layer": layer,
        "hook_site": f"decoder_block_output[{layer}]",
        "target_model_id": "synthetic/none",
        "target_model_revision": "synthetic@none",
        "mode_definition_version": "0.1.0-provisional",
        "alphas": ALPHAS,
        "n_random_directions": n_random,
        "projection_scale_s": s_proj,
        "fixture_variant": variant,
        "seed": seed,
    }
    bundle = Bundle(
        bundle_id=bundle_id,
        candidate_id=candidate_id,
        origin=Origin.synthetic_fixture,
        meta=meta,
        stimuli=stimuli,
        activations={layer: H},
        continuations=conts,
        interventions=interventions,
    )
    return bundle, truth


def build_tier2_fixtures(
    cfg: StudyConfig, out_dir: str | Path, variants: list[str] | None = None
) -> dict[str, Any]:
    out = Path(out_dir) / "bundles"
    out.mkdir(parents=True, exist_ok=True)
    built: dict[str, Any] = {}
    variants = variants or VARIANTS
    for vi, variant in enumerate(variants):
        for ci, cand in enumerate(cfg.data.candidate_ids):
            bundle_id = f"fx_{variant}"
            b, truth = make_fixture_bundle(variant, cand, bundle_id, seed=cfg.study.seed + 100 * vi + ci)
            d = out / bundle_id / cand
            hashes = b.save(d)
            write_json(d / "fixture_truth.json", truth)
            built[f"{bundle_id}/{cand}"] = {
                "path": str(d),
                "n_stimuli": len(b.stimuli),
                "hash_stimuli": hashes["stimuli.jsonl"],
            }
    return {"bundles_dir": str(out), "built": built, "variants": variants}
