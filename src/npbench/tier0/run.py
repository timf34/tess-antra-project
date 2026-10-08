"""Tier-zero judge runner: fresh session per rating, stable request keys, resume, retries with a cap,
budget stop before the cap is exhausted, and an append-only host ledger."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import StudyConfig
from ..prompts import TIER0_JUDGMENT_JSON_SCHEMA, render_tier0_prompt
from ..providers import ProviderError, ProviderRequest, build_provider
from ..runner.ledger import Ledger
from ..schemas import Judgment, JudgmentLabels, JudgmentStatus, PanelManifest, ProviderCallMeta
from ..util import append_jsonl, read_jsonl, utc_now_iso, write_json


def _parse_labels(text: str) -> tuple[JudgmentLabels | None, str | None]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # tolerate fenced JSON
        s = text.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s[s.find("{") :]
        try:
            data = json.loads(s[: s.rfind("}") + 1])
        except Exception as e:  # noqa: BLE001
            return None, f"unparseable JSON: {e}"
    try:
        return JudgmentLabels.model_validate(data), None
    except Exception as e:  # noqa: BLE001
        return None, f"schema violation: {e}"


class BudgetGuard:
    """Stops scheduling before the configured cap is exhausted (estimate-based, conservative)."""

    def __init__(self, cfg: StudyConfig, plan: dict[str, Any]):
        self.cap = cfg.budget.max_total_usd
        self.spent_est = 0.0
        self.per_rating = {}
        est = cfg.tier0.estimated_tokens_per_rating
        for j in cfg.tier0.judges:
            if j.provider == "fake" or j.price_usd_per_m_input is None:
                self.per_rating[j.slot_id] = 0.0
            else:
                self.per_rating[j.slot_id] = (
                    est["input"] * j.price_usd_per_m_input + est["output"] * j.price_usd_per_m_output
                ) / 1e6
        self.allowance = plan["cost"]["allowance_factor"]

    def can_schedule(self, slot: str) -> bool:
        if self.cap is None:
            return self.per_rating.get(slot, 0.0) == 0.0
        return self.spent_est + self.per_rating.get(slot, 0.0) * self.allowance <= self.cap

    def charge(self, slot: str, meta: ProviderCallMeta, judge_cfg) -> None:
        if judge_cfg.provider == "fake" or judge_cfg.price_usd_per_m_input is None:
            return
        it = meta.input_tokens or 0
        ot = meta.output_tokens or 0
        self.spent_est += (it * judge_cfg.price_usd_per_m_input + ot * judge_cfg.price_usd_per_m_output) / 1e6


def run_judges(
    cfg: StudyConfig,
    panel: PanelManifest,
    plan: dict[str, Any],
    out_dir: str | Path,
    *,
    smoke_items: int | None = None,
    allow_live: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / "ledger.jsonl")
    judgments_path = out / "judgments.jsonl"
    done: set[str] = set()
    if judgments_path.exists():
        for row in read_jsonl(judgments_path):
            done.add(row["request_key"])
    judges = {j.slot_id: j for j in cfg.tier0.judges}
    any_live = any(j.provider != "fake" for j in judges.values())
    if any_live and not (plan["budget"]["live_allowed"] and allow_live):
        raise PermissionError(
            "live judges configured but live calls are not allowed: "
            f"plan.budget={plan['budget']} allow_live={allow_live}. Resolve the budget and pass --live."
        )
    providers = {}
    for slot, j in judges.items():
        providers[slot] = build_provider(
            j.provider, j.model_id, j.settings, base_url=j.base_url, api_key_env=j.api_key_env
        )
    items = {it.item_id: it for it in panel.items}
    guard = BudgetGuard(cfg, plan)
    requests = plan["requests"]
    if smoke_items is not None:
        chosen = {it.item_id for it in panel.items[:smoke_items]}
        requests = [r for r in requests if r["item_id"] in chosen and r["repetition"] == 1]
    ledger.append(
        "tier0_run_start",
        {
            "plan_hash": plan.get("config_hash"),
            "panel_manifest_hash": panel.manifest_hash,
            "rubric_hash": plan["rubric_hash"],
            "n_requests": len(requests),
            "resumed": len(done),
            "smoke_items": smoke_items,
            "providers": {k: v.describe() for k, v in providers.items()},
        },
    )
    counts = {
        "completed": 0,
        "resumed": len(done),
        "errors": 0,
        "budget_stopped": 0,
        "refused": 0,
        "unscorable": 0,
    }
    max_retries = cfg.runner.max_retries_per_request
    for r in requests:
        key = r["request_key"]
        if key in done:
            continue
        slot = r["judge_slot"]
        jcfg = judges[slot]
        if not guard.can_schedule(slot):
            counts["budget_stopped"] += 1
            ledger.append("budget_stop", {"request_key": key, "slot": slot, "spent_est": guard.spent_est})
            continue
        item = items[r["item_id"]]
        prompt = render_tier0_prompt(item.text, item.context_text, r["label_order_seed"])
        req = ProviderRequest(
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
            max_tokens=int(jcfg.settings.get("max_tokens", 1024)),
            json_schema=TIER0_JUDGMENT_JSON_SCHEMA,
            settings={k: v for k, v in jcfg.settings.items() if k != "max_tokens"},
            idempotency_key=key,
            purpose="judge",
        )
        ledger.append(
            "judge_request",
            {
                "request_key": key,
                "item_id": item.item_id,
                "slot": slot,
                "repetition": r["repetition"],
                "prompt_hash": prompt.prompt_hash,
                "model": jcfg.model_id,
            },
        )
        attempt = 0
        labels: JudgmentLabels | None = None
        meta: ProviderCallMeta | None = None
        err: str | None = None
        while attempt <= max_retries:
            attempt += 1
            try:
                resp = providers[slot].complete(req)
                meta = resp.meta
                guard.charge(slot, meta, jcfg)
                ledger.append(
                    "judge_response",
                    {
                        "request_key": key,
                        "attempt": attempt,
                        "raw_hash": meta.raw_response_hash,
                        "usage": {"in": meta.input_tokens, "out": meta.output_tokens},
                        "returned_model": meta.returned_model,
                        "request_id": meta.provider_request_id,
                        "stop_reason": resp.stop_reason,
                    },
                )
                if resp.stop_reason == "refusal":
                    labels = JudgmentLabels(
                        status=JudgmentStatus.refused,
                        refusal_or_unscorable_reason=meta.error or "provider refusal stop",
                    )
                    break
                labels, perr = _parse_labels(resp.text)
                if labels is not None:
                    break
                err = perr
                ledger.append("judge_parse_error", {"request_key": key, "attempt": attempt, "error": perr})
            except ProviderError as e:
                err = f"{e.kind}: {e}"
                ledger.append(
                    "judge_error",
                    {"request_key": key, "attempt": attempt, "error": err, "retryable": e.retryable},
                )
                if not e.retryable:
                    break
                time.sleep(min(2.0 * attempt, 10.0))
        if labels is None:
            counts["errors"] += 1
            labels = JudgmentLabels(
                status=JudgmentStatus.error, refusal_or_unscorable_reason=err or "unknown error"
            )
            if meta is None:
                meta = ProviderCallMeta(
                    provider=jcfg.provider,
                    requested_model=jcfg.model_id,
                    started_at=utc_now_iso(),
                    completed_at=utc_now_iso(),
                    latency_s=0.0,
                    error=err,
                    is_mock=jcfg.provider == "fake",
                )
        if labels.status == JudgmentStatus.refused:
            counts["refused"] += 1
        elif labels.status == JudgmentStatus.unscorable:
            counts["unscorable"] += 1
        j = Judgment(
            request_key=key,
            manifest_id=panel.manifest_id,
            item_id=item.item_id,
            judge_slot=slot,
            repetition=r["repetition"],
            presentation_position=r["presentation_position"],
            label_order_seed=r["label_order_seed"],
            session_mode="fresh",
            batch_id=None,
            prompt_hash=prompt.prompt_hash,
            labels=labels,
            call=meta,
            attempt=attempt,
        )
        append_jsonl(judgments_path, j.model_dump(mode="json"))
        done.add(key)
        counts["completed"] += 1
    ledger.append("tier0_run_end", {**counts, "spent_est_usd": round(guard.spent_est, 4)})
    summary = {
        "kind": "tier0_run_summary",
        "origin": panel.origin.value,
        "out_dir": str(out),
        "judgments_path": str(judgments_path),
        "ledger_ok": ledger.verify().ok,
        "n_requests_planned": len(requests),
        **counts,
        "spent_est_usd": round(guard.spent_est, 4),
        "mock_providers": [k for k, v in providers.items() if v.is_mock],
    }
    write_json(out / "run_summary.json", summary)
    return summary
