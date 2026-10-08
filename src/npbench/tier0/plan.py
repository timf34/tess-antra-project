"""Tier-zero plan: enumerate every individual rating request with stable keys, count ratings, and
preflight cost against the configured budget. ``plan`` counts individual ratings even if a provider
would batch several items per request."""

from __future__ import annotations

import random
from typing import Any

from ..config import StudyConfig
from ..prompts import tier0_rubric_hash
from ..schemas import PanelManifest
from ..util import sha256_obj, stable_key, utc_now_iso


def enumerate_requests(cfg: StudyConfig, panel: PanelManifest) -> list[dict[str, Any]]:
    reqs: list[dict[str, Any]] = []
    for rep in range(1, cfg.tier0.repetitions + 1):
        for judge in cfg.tier0.judges:
            order = list(range(len(panel.items)))
            random.Random(sha256_obj([cfg.study.seed, judge.slot_id, rep, "order"])).shuffle(order)
            for pos, idx in enumerate(order):
                item = panel.items[idx]
                key = stable_key(panel.manifest_id, item.item_id, judge.slot_id, rep)
                reqs.append(
                    {
                        "request_key": key,
                        "item_id": item.item_id,
                        "judge_slot": judge.slot_id,
                        "repetition": rep,
                        "presentation_position": pos,
                        "label_order_seed": int(sha256_obj([key, "labels"])[:8], 16),
                        "category": item.category.value,
                    }
                )
    return reqs


def estimate_cost(cfg: StudyConfig, n_items: int, reps: int) -> dict[str, Any]:
    est = cfg.tier0.estimated_tokens_per_rating
    per_judge = {}
    total = 0.0
    unpriced: list[str] = []
    any_live = False
    for j in cfg.tier0.judges:
        n = n_items * reps
        if j.provider == "fake":
            per_judge[j.slot_id] = {"ratings": n, "usd": 0.0, "mock": True}
            continue
        any_live = True
        if j.price_usd_per_m_input is None or j.price_usd_per_m_output is None:
            unpriced.append(j.slot_id)
            per_judge[j.slot_id] = {"ratings": n, "usd": None, "mock": False}
            continue
        usd = n * (est["input"] * j.price_usd_per_m_input + est["output"] * j.price_usd_per_m_output) / 1e6
        per_judge[j.slot_id] = {"ratings": n, "usd": round(usd, 4), "mock": False}
        total += usd
    # conservative allowance: retries + reasoning tokens
    allowance = 1.3
    return {
        "per_judge": per_judge,
        "estimated_usd": round(total, 4),
        "estimated_usd_with_allowance": round(total * allowance, 4),
        "allowance_factor": allowance,
        "unpriced_live_judges": unpriced,
        "any_live_judges": any_live,
        "assumed_tokens_per_rating": est,
        "pricing_note": "estimates only; provider billing may differ (cached tokens, reasoning tokens, routing)",
    }


def budget_decision(cfg: StudyConfig, cost: dict[str, Any]) -> dict[str, Any]:
    if not cost["any_live_judges"]:
        return {"live_allowed": True, "reason": "no live judges configured (mock only)"}
    if cost["unpriced_live_judges"]:
        return {
            "live_allowed": False,
            "reason": f"unpriced live judges: {cost['unpriced_live_judges']}; set price_usd_per_m_* in the judge config",
        }
    if not cfg.live_budget_resolved():
        return {
            "live_allowed": False,
            "reason": f"budget.max_total_usd is unresolved and policy is {cfg.budget.unresolved_live_budget_policy}; set budget.max_total_usd",
        }
    cap = float(cfg.budget.max_total_usd or 0)
    if cost["estimated_usd_with_allowance"] > cap:
        return {
            "live_allowed": False,
            "reason": f"estimated {cost['estimated_usd_with_allowance']} USD exceeds cap {cap} USD",
        }
    return {
        "live_allowed": True,
        "reason": f"estimated {cost['estimated_usd_with_allowance']} USD within cap {cap} USD",
    }


def build_plan(cfg: StudyConfig, panel: PanelManifest) -> dict[str, Any]:
    reqs = enumerate_requests(cfg, panel)
    cost = estimate_cost(cfg, len(panel.items), cfg.tier0.repetitions)
    decision = budget_decision(cfg, cost)
    return {
        "kind": "tier0_plan",
        "created_at": utc_now_iso(),
        "origin": panel.origin.value,
        "stage": cfg.study.stage,
        "config_hash": cfg.config_hash(),
        "panel_manifest_id": panel.manifest_id,
        "panel_manifest_hash": panel.manifest_hash,
        "corpus_hash": panel.corpus_hash,
        "rubric_id": cfg.tier0.rubric_id,
        "rubric_hash": tier0_rubric_hash(),
        "n_items": len(panel.items),
        "strata_counts": panel.strata_counts,
        "n_judges": len(cfg.tier0.judges),
        "judges": [j.model_dump() for j in cfg.tier0.judges],
        "repetitions": cfg.tier0.repetitions,
        "n_ratings": len(reqs),
        "session_mode": cfg.tier0.session_mode,
        "cost": cost,
        "budget": decision,
        "requests": reqs,
    }
