"""Tier-two schedule: stable run keys, randomized framing order within blocks, interleaving, cost
preflight including reviewer and awareness calls, and the study freeze lock."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .. import prompts
from ..config import StudyConfig
from ..mode_registry import load_mode_registry
from ..packets.build import packets_root
from ..reference.build import reference_dir
from ..schemas import RunKey
from ..util import sha256_file, sha256_obj, sha256_text, utc_now_iso, write_json

SCORING_VERSION = "scoring_v2"


def load_packets_index(cfg: StudyConfig) -> dict[str, Any]:
    p = packets_root(cfg) / "_index.json"
    if not p.exists():
        raise FileNotFoundError(f"packets index not found at {p}; run `npbench packets build`")
    return json.loads(p.read_text())


def enumerate_runs(cfg: StudyConfig, index: dict[str, Any]) -> list[dict[str, Any]]:
    packets = index["packets"]
    families: dict[str, dict[str, Any]] = {}
    for pid, info in packets.items():
        if cfg.evaluation.bundle_ids and info["bundle_id"] not in cfg.evaluation.bundle_ids:
            continue
        if (
            info["candidate_id"] not in cfg.data.candidate_ids
            or info["method"] not in cfg.evaluation.methods
            or info["task_type"] not in cfg.evaluation.task_types
        ):
            continue
        fam = families.setdefault(
            info["shared_key"],
            {
                "bundle_id": info["bundle_id"],
                "candidate_id": info["candidate_id"],
                "method": info["method"],
                "task_type": info["task_type"],
                "packets": {},
            },
        )
        fam["packets"][info["framing"]] = pid
    runs: list[dict[str, Any]] = []
    slots = cfg.evaluation.assistant_slots
    for rep in range(1, cfg.evaluation.repetitions + 1):
        for fam_key, fam in sorted(families.items()):
            for slot in slots:
                block = sha256_obj([cfg.study.version, fam_key, slot, rep, "framing_order"])
                order = list(cfg.evaluation.framings)
                random.Random(block).shuffle(order)
                for pos, framing in enumerate(order):
                    pid = fam["packets"].get(framing)
                    if pid is None:
                        continue
                    key = RunKey(
                        study_version=cfg.study.version,
                        bundle_id=fam["bundle_id"],
                        candidate_id=fam["candidate_id"],
                        method=fam["method"],
                        task_type=fam["task_type"],
                        framing=framing,
                        model_slot=slot,
                        repetition=rep,
                    )
                    runs.append(
                        {
                            "run_id": key.run_id(),
                            "key": key.model_dump(mode="json"),
                            "packet_id": pid,
                            "block": block[:12],
                            "framing_position": pos,
                            "interleave_key": sha256_obj([block, pos])[:12],
                        }
                    )
    # interleave model versions and task types over time: order by repetition, then a hash that mixes
    # family/slot, keeping each block's two framings adjacent in their randomized order.
    runs.sort(key=lambda r: (r["key"]["repetition"], r["block"], r["framing_position"]))
    for i, r in enumerate(runs):
        r["schedule_position"] = i
    return runs


def estimate_tier2_cost(cfg: StudyConfig, n_runs: int) -> dict[str, Any]:
    prices = {m.slot: m for m in cfg.providers.models}
    estimates = cfg.runner.model_extra or {}
    per_run_tokens = int(
        estimates.get("estimated_billable_tokens_per_run", cfg.runner.max_total_billable_tokens_per_run)
    )
    output_fraction = float(estimates.get("estimated_output_fraction", 0.3))
    if (
        not 0 <= output_fraction <= 1
        or not 0 < per_run_tokens <= cfg.runner.max_total_billable_tokens_per_run
    ):
        raise ValueError("Invalid preflight token estimate")
    aud_settings = (cfg.providers.auditor.model_extra or {}).get("settings", {})
    reviewer_input = 2 * 32_000
    reviewer_output = 2 * int(aud_settings.get("max_tokens", 4000))
    reviewer_tokens = reviewer_input + reviewer_output
    awareness_tokens = 32_000
    retry_allowance = 1.25
    per_slot: dict[str, Any] = {}
    total = 0.0
    unpriced: list[str] = []
    any_live = False
    runs_per_slot = n_runs // max(1, len(cfg.evaluation.assistant_slots))
    for slot in cfg.evaluation.assistant_slots:
        m = prices.get(slot)
        if m is None or m.provider in (None, "fake"):
            per_slot[slot] = {"runs": runs_per_slot, "usd": 0.0, "mock": True}
            continue
        any_live = True
        extra = getattr(m, "model_extra", None) or {}
        pin, pout = extra.get("price_usd_per_m_input"), extra.get("price_usd_per_m_output")
        if pin is None or pout is None:
            unpriced.append(slot)
            per_slot[slot] = {"runs": runs_per_slot, "usd": None, "mock": False}
            continue
        usd_run = (
            (1 - output_fraction) * per_run_tokens * pin + output_fraction * per_run_tokens * pout
        ) / 1e6 + (awareness_tokens * pin + 800 * pout) / 1e6
        per_slot[slot] = {
            "runs": runs_per_slot,
            "usd_per_run": round(usd_run, 4),
            "usd": round(usd_run * runs_per_slot, 2),
            "mock": False,
        }
        total += usd_run * runs_per_slot
    aud = cfg.providers.auditor
    aud_extra = getattr(aud, "model_extra", None) or {}
    aud_cost = None
    if aud.provider and aud.provider != "fake":
        any_live = True
        pin, pout = aud_extra.get("price_usd_per_m_input"), aud_extra.get("price_usd_per_m_output")
        if pin is None or pout is None:
            unpriced.append("auditor")
        else:
            aud_cost = n_runs * (reviewer_input * pin + reviewer_output * pout) / 1e6
            total += aud_cost
    return {
        "per_slot": per_slot,
        "auditor_usd": None if aud_cost is None else round(aud_cost, 2),
        "reviewer_sessions": 2 * n_runs,
        "awareness_calls": n_runs,
        "estimated_usd": round(total, 2),
        "estimated_usd_with_allowance": round(total * retry_allowance, 2),
        "allowance_factor": retry_allowance,
        "unpriced": unpriced,
        "any_live": any_live,
        "assumptions": {
            "tokens_per_run_cap": cfg.runner.max_total_billable_tokens_per_run,
            "estimated_tokens_per_run": per_run_tokens,
            "reviewer_tokens_per_run": reviewer_tokens,
            "awareness_tokens": awareness_tokens,
            "input_output_split": f"{1 - output_fraction:.2f}/{output_fraction:.2f}",
        },
    }


def budget_decision(cfg: StudyConfig, cost: dict[str, Any]) -> dict[str, Any]:
    if not cost["any_live"]:
        return {"live_allowed": True, "reason": "mock providers only"}
    if cost["unpriced"]:
        return {"live_allowed": False, "reason": f"unpriced live slots: {cost['unpriced']}"}
    if not cfg.live_budget_resolved():
        return {"live_allowed": False, "reason": "budget.max_total_usd unresolved (policy preflight_only)"}
    cap = float(cfg.budget.max_total_usd or 0)
    ok = cost["estimated_usd_with_allowance"] <= cap
    return {
        "live_allowed": ok,
        "reason": f"estimate {cost['estimated_usd_with_allowance']} USD vs cap {cap} USD",
    }


def build_tier2_plan(cfg: StudyConfig, lock_path: Path | None = None) -> dict[str, Any]:
    index = load_packets_index(cfg)
    runs = enumerate_runs(cfg, index)
    cost = estimate_tier2_cost(cfg, len(runs))
    plan = {
        "kind": "tier2_plan",
        "created_at": utc_now_iso(),
        "stage": cfg.study.stage,
        "study_version": cfg.study.version,
        "config_hash": cfg.config_hash(),
        "origins": sorted({index["packets"][r["packet_id"]]["origin"] for r in runs}),
        "n_runs": len(runs),
        "n_reviewer_sessions": 2 * len(runs),
        "n_awareness_calls": len(runs),
        "matrix": {
            "bundles": sorted({r["key"]["bundle_id"] for r in runs}),
            "candidates": sorted({r["key"]["candidate_id"] for r in runs}),
            "methods": cfg.evaluation.methods,
            "task_types": cfg.evaluation.task_types,
            "framings": cfg.evaluation.framings,
            "slots": cfg.evaluation.assistant_slots,
            "repetitions": cfg.evaluation.repetitions,
        },
        "cost": cost,
        "budget": budget_decision(cfg, cost),
        "lock": None,
        "runs": runs,
    }
    if lock_path is not None:
        lock = json.loads(Path(lock_path).read_text())
        current = freeze_digest(cfg)
        plan["lock"] = {
            "path": str(lock_path),
            "matches": lock.get("freeze_digest") == current["freeze_digest"],
            "lock_digest": lock.get("freeze_digest"),
            "current_digest": current["freeze_digest"],
        }
        if not plan["lock"]["matches"]:
            raise PermissionError(
                "configuration or data changed since the freeze; the lock digest no longer matches"
            )
    return plan


def freeze_digest(cfg: StudyConfig) -> dict[str, Any]:
    index = load_packets_index(cfg)
    data_hashes = {pid: info["data_hashes"] for pid, info in sorted(index["packets"].items())}
    ref_root = reference_dir(cfg)
    gold = {
        str(p.relative_to(ref_root)): sha256_file(p) for p in sorted(ref_root.rglob("reference_results.json"))
    }
    prompts_hash = sha256_text(Path(prompts.__file__).read_text(encoding="utf-8"))
    model_cfg = [m.model_dump() for m in cfg.providers.models] + [cfg.providers.auditor.model_dump()]
    pricing = {"budget": cfg.budget.model_dump(), "models": model_cfg}
    schedule = enumerate_runs(cfg, index)
    body = {
        "study_version": cfg.study.version,
        "config_hash": cfg.config_hash(),
        "packet_data_hashes": data_hashes,
        "gold_hashes": gold,
        "prompts_hash": prompts_hash,
        "scoring_version": SCORING_VERSION,
        "model_configs": model_cfg,
        "pricing": pricing,
        "schedule": [(r["run_id"], r["packet_id"], r["schedule_position"]) for r in schedule],
    }
    return {"freeze_digest": sha256_obj(body), "body": body}


def freeze_study(cfg: StudyConfig, out: Path) -> dict[str, Any]:
    reg = load_mode_registry(cfg.resolve(cfg.mode_definitions.registry_path))
    allowed, reason = reg.pilot_allowed()
    if cfg.is_production() and not allowed:
        raise PermissionError(f"cannot freeze a pilot: {reason}")
    d = freeze_digest(cfg)
    index = load_packets_index(cfg)
    origins = sorted({i["origin"] for i in index["packets"].values()})
    if cfg.is_production() and origins != ["real_target"]:
        raise PermissionError(f"cannot freeze a pilot on origins {origins}")
    if cfg.is_production():
        references = list(reference_dir(cfg).rglob("reference_results.json"))
        if not references:
            raise PermissionError("Cannot freeze pilot without reference results")
        for path in references:
            reference = json.loads(path.read_text())
            if reference.get("extras", {}).get("n_pending_rating_rows", 0):
                raise PermissionError(f"Cannot freeze pilot: incomplete blinded ratings in {path}")
            rows = reference["reference"]
            pending = [
                r["result_id"]
                for r in rows
                if r.get("availability") == "pending_readout"
                or (r["result_id"].split("_")[0] in {"A7", "B4", "B7", "B11"} and r.get("estimate") is None)
            ]
            if pending:
                raise PermissionError(f"Cannot freeze pilot: pending required readouts in {path}: {pending}")
    lock = {
        "kind": "study_lock",
        "created_at": utc_now_iso(),
        "stage": cfg.study.stage,
        "mode_registry": {
            "version": reg.registry_version,
            "status": reg.status,
            "pilot_allowed": allowed,
            "reason": reason,
        },
        "origins": origins,
        **d,
    }
    write_json(out, lock)
    return {
        "kind": "freeze",
        "path": str(out),
        "freeze_digest": d["freeze_digest"],
        "n_runs": len(d["body"]["schedule"]),
        "mode_registry": lock["mode_registry"],
        "origins": origins,
    }
