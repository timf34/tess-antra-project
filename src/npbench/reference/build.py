"""``npbench reference build``: compute reference results for every bundle/candidate/method/task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..bundles import Bundle
from ..config import StudyConfig, assert_production_origin
from ..splits import check_split_integrity
from ..util import write_json


def reference_dir(cfg: StudyConfig) -> Path:
    return cfg.resolve(cfg.paths.get("reference", f"artifacts/{cfg.study.version}/reference"))


def bundles_dir(cfg: StudyConfig, data: Path | None) -> Path:
    if data is not None:
        d = data / "bundles" if (data / "bundles").exists() else data
        return d
    return cfg.resolve(cfg.paths.get("bundles", f"artifacts/{cfg.study.version}/bundles"))


def iter_bundle_dirs(root: Path):
    for bdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            if (cdir / "bundle.json").exists():
                yield bdir.name, cdir.name, cdir


def build_reference(cfg: StudyConfig, data: Path | None) -> dict[str, Any]:
    from .oracle import compute_reference

    root = bundles_dir(cfg, data)
    out_root = reference_dir(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    built: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for bundle_id, cand, cdir in iter_bundle_dirs(root):
        b = Bundle.load(cdir)
        try:
            assert_production_origin(cfg, b.origin)
        except PermissionError as e:
            rejected.append({"bundle": f"{bundle_id}/{cand}", "reason": str(e)})
            continue
        integ = check_split_integrity(
            b.stimuli,
            heldout_template_families=[
                f for f in {s.template_family_id for s in b.stimuli} if f.startswith("fam_heldout")
            ],
            heldout_personas=[
                p for p in {s.persona_id for s in b.stimuli} if p.startswith("persona_heldout")
            ],
        )
        if not integ.ok:
            rejected.append(
                {
                    "bundle": f"{bundle_id}/{cand}",
                    "reason": "split integrity violated",
                    "problems": integ.problems[:5],
                }
            )
            continue
        for method in cfg.evaluation.methods:
            for task_type in cfg.evaluation.task_types:
                ref = compute_reference(b, method, task_type, seed=cfg.study.seed)
                d = out_root / bundle_id / cand / method / task_type
                d.mkdir(parents=True, exist_ok=True)
                write_json(d / "reference_results.json", ref)
                if method == "activation_contrasts":
                    np.savez(
                        d / "reference_vectors.npz",
                        **{f"u_{k}": np.asarray(v) for k, v in ref["extras"]["units"].items()},
                        **{f"leak_{k}": np.asarray(v) for k, v in ref["extras"]["leaked_units"].items()},
                    )
                built.append(
                    {
                        "bundle": bundle_id,
                        "candidate": cand,
                        "method": method,
                        "task_type": task_type,
                        "n_results": len(ref["reference"]),
                        "path": str(d),
                    }
                )
    summary = {
        "kind": "reference_build",
        "bundles_dir": str(root),
        "reference_dir": str(out_root),
        "built": built,
        "rejected": rejected,
    }
    write_json(out_root / "reference_summary.json", summary)
    return summary
