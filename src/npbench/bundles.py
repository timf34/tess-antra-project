"""Bundle data format shared by fixtures, the target lane, the reference oracle and packet building.

A bundle holds everything the evaluator knows about one (bundle_id, candidate_id): stimuli with full
semantics, activations per layer, coded behavioural rows, and intervention readouts. Packets are
anonymized *views* of a bundle; the bundle itself is never mounted into an assistant workspace.

Anonymized ids used in packet views (identical in both framings):
    conditions  A=roleplay  B=simulation  C=enactment
    contents    P=positive  N=neutral     Q=negative
    directions  C-A (=ER), C-B (=ES), A-B (=RS); P-Q@A etc. for polarity contrasts within a condition
    outcome codes K1 (registered target outcome) / K2 (other); rating_scale1 (frozen independent rating)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import Origin, StimulusRecord
from .util import read_jsonl, sha256_file, write_json

COND = {"roleplay": "A", "simulation": "B", "enactment": "C"}
COND_INV = {v: k for k, v in COND.items()}
CONTENT = {"positive": "P", "neutral": "N", "negative": "Q"}
CONTENT_INV = {v: k for k, v in CONTENT.items()}
DIRECTIONS = {
    "C-A": ("enactment", "roleplay"),
    "C-B": ("enactment", "simulation"),
    "A-B": ("roleplay", "simulation"),
}
MODE_DIRECTION_OF = {"ER": "C-A", "ES": "C-B", "RS": "A-B"}


@dataclass
class Bundle:
    bundle_id: str
    candidate_id: str
    origin: Origin
    meta: dict[str, Any]
    stimuli: list[StimulusRecord]
    activations: dict[int, np.ndarray]  # layer -> (n_stimuli, d), rows aligned with ``stimuli``
    continuations: list[dict[str, Any]] = field(default_factory=list)
    interventions: list[dict[str, Any]] = field(default_factory=list)

    # -- accessors ----------------------------------------------------------------------------
    @property
    def stimulus_ids(self) -> list[str]:
        return [s.stimulus_id for s in self.stimuli]

    def index(self) -> dict[str, int]:
        return {s.stimulus_id: i for i, s in enumerate(self.stimuli)}

    def layers(self) -> list[int]:
        return sorted(self.activations)

    def primary_layer(self) -> int:
        return int(self.meta.get("primary_layer", self.layers()[0] if self.activations else 0))

    # -- IO -------------------------------------------------------------------------------------
    def save(self, directory: str | Path) -> dict[str, str]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "stimuli.jsonl", "w", encoding="utf-8") as f:
            for s in self.stimuli:
                f.write(json.dumps(s.model_dump(mode="json"), sort_keys=True) + "\n")
        arrays: dict[str, np.ndarray] = {"stimulus_ids": np.asarray(self.stimulus_ids)}
        for k, a in self.activations.items():
            arrays[f"layer_{k}"] = np.asarray(a, dtype=np.float32)
        np.savez(d / "activations.npz", **arrays)
        with open(d / "continuations.jsonl", "w", encoding="utf-8") as f:
            for r in self.continuations:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        with open(d / "interventions.jsonl", "w", encoding="utf-8") as f:
            for r in self.interventions:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        meta = {
            **self.meta,
            "bundle_id": self.bundle_id,
            "candidate_id": self.candidate_id,
            "origin": self.origin.value,
            "n_stimuli": len(self.stimuli),
            "layers": self.layers(),
            "n_continuation_rows": len(self.continuations),
            "n_intervention_rows": len(self.interventions),
        }
        write_json(d / "bundle.json", meta)
        hashes = {
            p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != "hashes.json"
        }
        write_json(d / "hashes.json", hashes)
        return hashes

    @classmethod
    def load(cls, directory: str | Path) -> Bundle:
        d = Path(directory)
        with open(d / "bundle.json", encoding="utf-8") as f:
            meta = json.load(f)
        stimuli = [StimulusRecord.model_validate(r) for r in read_jsonl(d / "stimuli.jsonl")]
        acts: dict[int, np.ndarray] = {}
        with np.load(d / "activations.npz", allow_pickle=False) as z:
            ids = [str(x) for x in z["stimulus_ids"]]
            if ids != [s.stimulus_id for s in stimuli]:
                raise ValueError("activations.npz stimulus order does not match stimuli.jsonl")
            for k in z.files:
                if k.startswith("layer_"):
                    acts[int(k.split("_", 1)[1])] = np.asarray(z[k], dtype=np.float64)
        conts = read_jsonl(d / "continuations.jsonl") if (d / "continuations.jsonl").exists() else []
        ints = read_jsonl(d / "interventions.jsonl") if (d / "interventions.jsonl").exists() else []
        return cls(
            bundle_id=meta["bundle_id"],
            candidate_id=meta["candidate_id"],
            origin=Origin(meta["origin"]),
            meta={k: v for k, v in meta.items() if k not in ("bundle_id", "candidate_id", "origin")},
            stimuli=stimuli,
            activations=acts,
            continuations=conts,
            interventions=ints,
        )


def anonymize_stimulus(s: StimulusRecord) -> dict[str, Any]:
    """Packet view of one stimulus: condition/content ids only, no mode/candidate vocabulary."""
    return {
        "row_id": s.stimulus_id,
        "group": s.underlying_context_id,
        "cond": COND[s.mode_intended.value],
        "content": CONTENT[s.candidate_polarity.value],
        "split": s.split.value,
        "family": s.template_family_id,
        "persona": s.persona_id,
        "explicit_pref": s.preference_explicit,
        # None when no observed-mode label was assigned (label "unknown"); False only for a labelled mismatch.
        "observed_ok": (
            None if s.mode_observed_label.value == "unknown" else s.mode_observed_label.value == s.mode_intended.value
        ),
        "token_count": s.token_count,
    }


def anonymize_continuation(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": r["stimulus_id"],
        "group": r["underlying_context_id"],
        "cond": COND[r["mode_intended"]],
        "content": CONTENT[r["candidate_polarity"]],
        "split": r["split"],
        "family": r["template_family_id"],
        "persona": r["persona_id"],
        "explicit_pref": r["preference_explicit"],
        "scenario_group": r["scenario_group"],
        "outcome_code": r["outcome_code"],
        "lik_alt1_total": r["lik_alt1_total"],
        "lik_alt2_total": r["lik_alt2_total"],
        "lik_alt1_per_token": r["lik_alt1_per_token"],
        "lik_alt2_per_token": r["lik_alt2_per_token"],
        "n_tokens_alt1": r["n_tokens_alt1"],
        "n_tokens_alt2": r["n_tokens_alt2"],
        "rating_scale1": r["rating_scale1"],
        "rating_scale1_sd": r["rating_scale1_sd"],
        "rater_panel_id": r["rater_panel_id"],
        "rating_status": r.get("rating_status", "scored" if r.get("rating_scale1") is not None else "pending"),
    }


def anonymize_intervention(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": r["stimulus_id"],
        "group": r["underlying_context_id"],
        "cond": COND[r["mode_intended"]],
        "content": CONTENT[r["candidate_polarity"]],
        "split": r["split"],
        "scenario_group": r["scenario_group"],
        "direction": r["direction_id"],
        "alpha": r["alpha"],
        "requested_norm": r["requested_norm"],
        "applied_norm": r["applied_norm"],
        "projection_delta": r["projection_delta"],
        "outcome_code": r["outcome_code"],
        "lik_contrast": r["lik_contrast"],
        "rating_scale1": r["rating_scale1"],
        "rating_status": r.get("rating_status", "scored" if r.get("rating_scale1") is not None else "pending"),
    }
