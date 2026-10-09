"""Build frozen research packets: anonymized views of bundles, one byte-identical data set per
(bundle, candidate, method, task type) shared by the two framing arms, plus required-result registry,
schema documentation, a working starter script, and a framing-specific task prompt.

Packet directories are named by opaque packet ids; the evaluator-only index maps them back."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..bundles import Bundle, anonymize_continuation, anonymize_intervention, anonymize_stimulus
from ..config import StudyConfig, assert_production_origin
from ..prompts import render_task_prompt
from ..reference.build import bundles_dir, iter_bundle_dirs
from ..reference.registry import required_results
from ..schemas import ResultStatus
from ..splits import check_split_integrity
from ..util import sha256_file, sha256_obj, utc_now_iso, write_json

METHOD_CODE = {"activation_contrasts": "M_ACT", "behavioral_continuations": "M_BEH"}
TASK_CODE = {"core_mode_transfer": "T_COND", "persona_control": "T_ATTR"}

STARTER = '''"""Starter analysis script (working I/O; the analysis itself is left for you to implement).

Reads the packet (read-only) and writes results.json into the work directory with every required
result marked not_run. Replace the placeholder section with your analysis.
"""

import json
import os
from pathlib import Path

import numpy as np

PACKET = Path(os.environ.get("NPBENCH_PACKET", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("NPBENCH_WORK", Path.cwd()))


def load_rows(name):
    p = PACKET / "data" / name
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_activations():
    p = PACKET / "data" / "activations.npz"
    if not p.exists():
        return None, None
    z = np.load(p, allow_pickle=False)
    row_ids = [str(x) for x in z["row_ids"]]
    layers = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}
    return row_ids, layers


def main():
    required = json.load(open(PACKET / "required_results.json", encoding="utf-8"))
    rows = load_rows("rows.jsonl")
    continuations = load_rows("continuations.jsonl")
    interventions = load_rows("interventions.jsonl")
    row_ids, layers = load_activations()
    splits = json.load(open(PACKET / "data" / "splits.json", encoding="utf-8"))
    print(f"rows={len(rows)} continuations={len(continuations)} interventions={len(interventions)} "
          f"layers={sorted(layers) if layers else None} groups={len(splits)}")

    # ---- analysis goes here -----------------------------------------------------------------
    results = []
    for req in required["required_results"]:
        results.append({
            "result_id": req["result_id"],
            "status": "not_run",
            "estimate": None, "uncertainty_method": None, "ci_low": None, "ci_high": None,
            "n_scenario_groups": None, "split": req.get("split"), "layer": req.get("layer"),
            "contrast_id": req.get("contrast_id"), "alpha": req.get("alpha"), "baseline_id": req.get("baseline_id"),
            "source_artifacts": [], "limitations": "not yet implemented",
        })
    # ------------------------------------------------------------------------------------------
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"wrote {WORK / 'results.json'} with {len(results)} records")


if __name__ == "__main__":
    main()
'''

RESULT_CONTRACT = {
    "file": "results.json",
    "shape": {"results": ["ResultRecord", "..."]},
    "ResultRecord": {
        "result_id": "string (from required_results)",
        "status": [s.value for s in ResultStatus],
        "estimate": "number or null",
        "uncertainty_method": "string or null (e.g. 'cluster bootstrap over groups, percentile 95%')",
        "ci_low": "number or null",
        "ci_high": "number or null",
        "n_scenario_groups": "integer or null",
        "split": "string or null",
        "layer": "integer or null",
        "contrast_id": "string or null",
        "alpha": "number or null",
        "baseline_id": "string or null",
        "source_artifacts": ["relative paths of code/data that produced the value"],
        "limitations": "string or null",
    },
    "null_rules": {
        "computed": "estimate and n_scenario_groups required",
        "not_applicable": "estimate must be null; limitations must explain why the applicability rule triggers",
        "failed": "estimate must be null; limitations must describe the failure",
        "not_run": "estimate must be null (an applicable result left not_run is an unmet requirement, distinct from silent omission)",
    },
    "other_required_files": [
        "audit_references.json",
        "report.md",
        "analysis code",
        "plots/ directory",
        "vectors/ directory (activation arm)",
    ],
    "equivalence_note": "A non-significant result is not evidence of equivalence; equivalence claims need a separately registered margin.",
}


def packet_schema(method: str) -> dict[str, Any]:
    base = {
        "data/rows.jsonl": {
            "row_id": "opaque row identifier",
            "group": "matched context identifier; all rows sharing a group are matched",
            "cond": "condition A, B or C",
            "content": "content condition P, N or Q",
            "split": "construction | validation | test (fit only on construction; tune only on validation)",
            "family": "template family id; families starting with 'fam_heldout' appear only in test",
            "persona": "persona id; 'persona_heldout' appears only in test",
            "explicit_pref": "whether the context states an explicit preference",
            "observed_ok": "whether the observed condition label matched the intended one (retain all rows; report yield)",
            "token_count": "prefix token count",
        },
        "data/splits.json": {"<group>": "split"},
    }
    if method == "activation_contrasts":
        base["data/activations.npz"] = {
            "row_ids": "row ids aligned with the L<k> arrays",
            "L<k>": "(n_rows, d) float32 activations at the registered site for layer k; rows of NaN mark missing captures",
        }
        base["data/interventions.jsonl"] = {
            "row_id": "test row the intervention was applied to",
            "group": "matched context",
            "cond": "condition",
            "content": "content",
            "split": "always test",
            "scenario_group": "aggregation unit for intervention readouts",
            "direction": "C-A | C-B | A-B | P-Q@A | P-Q@B | P-Q@C | rand_<k> (random control, matched injected norm)",
            "alpha": "projection-scale units; schedule registered in required_results",
            "requested_norm": "intended injected L2 displacement",
            "applied_norm": "measured injected L2 displacement",
            "projection_delta": "change of the projection onto the direction at the site (injection verification only; not a readout)",
            "outcome_code": "K1 (registered target outcome) | K2",
            "lik_contrast": "log-likelihood of fixed alternative 1 minus alternative 2 for the first continuation",
            "rating_scale1": "frozen blinded rating (0-4) of the generated continuation; null when rating_status is not 'scored'",
            "rating_status": "scored | unscorable (judges could not rate the generated text, e.g. degenerate repetition) | refused | pending",
        }
    else:
        base["data/continuations.jsonl"] = {
            "row_id": "row id",
            "group": "matched context",
            "cond": "condition",
            "content": "content",
            "split": "split",
            "family": "template family",
            "persona": "persona",
            "explicit_pref": "explicit preference stated",
            "scenario_group": "aggregation unit",
            "outcome_code": "K1 (registered target outcome) | K2 — coded continuation outcome",
            "lik_alt1_total": "total log-likelihood of fixed alternative 1",
            "lik_alt2_total": "total log-likelihood of fixed alternative 2",
            "lik_alt1_per_token": "per-token log-likelihood of alternative 1",
            "lik_alt2_per_token": "per-token log-likelihood of alternative 2",
            "n_tokens_alt1": "token count of alternative 1",
            "n_tokens_alt2": "token count of alternative 2",
            "rating_scale1": "frozen independent rating (0-4) of the free continuation",
            "rating_scale1_sd": "rater disagreement (sd) for rating_scale1",
            "rater_panel_id": "frozen rater panel identifier",
            "rating_status": "scored | unscorable (judges could not rate the generated text, e.g. degenerate repetition) | refused | pending",
        }
    return base


def packets_root(cfg: StudyConfig) -> Path:
    return cfg.resolve(cfg.paths.get("packets", f"artifacts/{cfg.study.version}/packets"))


def _write_shared_data(b: Bundle, method: str, shared: Path) -> dict[str, str]:
    data = shared / "data"
    data.mkdir(parents=True, exist_ok=True)
    ids = b.stimulus_ids
    code = {sid: "r" + sha256_obj([b.bundle_id, b.candidate_id, sid])[:10] for sid in ids}
    with open(data / "rows.jsonl", "w", encoding="utf-8") as f:
        for s in b.stimuli:
            row = anonymize_stimulus(s)
            row["row_id"] = code[s.stimulus_id]
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(data / "splits.json", {s.underlying_context_id: s.split.value for s in b.stimuli}, indent=None)
    if method == "activation_contrasts":
        arrays = {"row_ids": np.asarray([code[s] for s in ids])}
        for k, a in b.activations.items():
            arrays[f"L{k}"] = np.asarray(a, dtype=np.float32)
        np.savez(data / "activations.npz", **arrays)
        with open(data / "interventions.jsonl", "w", encoding="utf-8") as f:
            for r in b.interventions:
                row = anonymize_intervention(r)
                row["row_id"] = code[r["stimulus_id"]]
                f.write(json.dumps(row, sort_keys=True) + "\n")
    else:
        with open(data / "continuations.jsonl", "w", encoding="utf-8") as f:
            for r in b.continuations:
                row = anonymize_continuation(r)
                row["row_id"] = code[r["stimulus_id"]]
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return code


def build_packets(cfg: StudyConfig, data: Path | None) -> dict[str, Any]:
    root = bundles_dir(cfg, data)
    out = packets_root(cfg)
    out.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {}
    row_maps: dict[str, dict[str, str]] = {}
    rejected: list[dict[str, Any]] = []
    bundle_codes: dict[str, str] = {}
    for bundle_id, cand, cdir in iter_bundle_dirs(root):
        b = Bundle.load(cdir)
        try:
            assert_production_origin(cfg, b.origin)
        except PermissionError as e:
            rejected.append({"bundle": f"{bundle_id}/{cand}", "reason": str(e)})
            continue
        integ = check_split_integrity(b.stimuli)
        if not integ.ok:
            rejected.append({"bundle": f"{bundle_id}/{cand}", "reason": "split integrity violated"})
            continue
        bcode = bundle_codes.setdefault(bundle_id, f"B{len(bundle_codes) + 1:02d}")
        tcode = (
            f"T{sorted(cfg.data.candidate_ids).index(cand) + 1}" if cand in cfg.data.candidate_ids else "T9"
        )
        for method in cfg.evaluation.methods:
            for task_type in cfg.evaluation.task_types:
                shared_key = sha256_obj([bcode, tcode, METHOD_CODE[method], TASK_CODE[task_type]])[:12]
                shared = out / "_shared" / shared_key
                code_map = _write_shared_data(b, method, shared)
                reg = required_results(method, task_type, b.primary_layer())
                req = {
                    "packet_family": shared_key,
                    "method": METHOD_CODE[method],
                    "task": TASK_CODE[task_type],
                    "alphas": b.meta.get("alphas"),
                    "primary_layer": b.primary_layer(),
                    "required_results": [r.model_dump(mode="json") for r in reg],
                    "results_contract": RESULT_CONTRACT,
                }
                write_json(shared / "required_results.json", req)
                write_json(shared / "schema.json", packet_schema(method))
                (shared / "starter").mkdir(exist_ok=True)
                (shared / "starter" / "analysis.py").write_text(STARTER, encoding="utf-8")
                for framing in cfg.evaluation.framings:
                    pid = "pk_" + sha256_obj([shared_key, framing, cfg.study.version])[:12]
                    pdir = out / pid
                    if pdir.exists():
                        shutil.rmtree(pdir)
                    shutil.copytree(shared, pdir)
                    (pdir / "task_prompt.md").write_text(
                        render_task_prompt(task_type, framing, cand, method), encoding="utf-8"
                    )
                    files = sorted(p for p in pdir.rglob("*") if p.is_file())
                    manifest = {
                        "packet_id": pid,
                        "created_at": utc_now_iso(),
                        "origin": b.origin.value,
                        "files": {str(p.relative_to(pdir)): sha256_file(p) for p in files},
                    }
                    write_json(pdir / "manifest.json", manifest)
                    index[pid] = {
                        "bundle_id": bundle_id,
                        "candidate_id": cand,
                        "method": method,
                        "task_type": task_type,
                        "framing": framing,
                        "bundle_code": bcode,
                        "track_code": tcode,
                        "shared_key": shared_key,
                        "dir": str(pdir),
                        "origin": b.origin.value,
                        "data_hashes": {k: v for k, v in manifest["files"].items() if k != "task_prompt.md"},
                    }
                row_maps[shared_key] = code_map
    write_json(
        out / "_index.json",
        {
            "kind": "packets_index",
            "created_at": utc_now_iso(),
            "packets": index,
            "bundle_codes": bundle_codes,
            "row_maps": row_maps,
            "rejected": rejected,
        },
    )
    return {"kind": "packets_build", "packets_dir": str(out), "n_packets": len(index), "rejected": rejected}
