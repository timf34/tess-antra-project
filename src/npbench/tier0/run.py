"""Tier-zero judge runner: fresh session per rating, stable request keys, resume, retries with a cap,
budget stop before the cap is exhausted, and an append-only host ledger."""

from __future__ import annotations

import fcntl
import json
import time
from functools import wraps
from pathlib import Path
from typing import Any

from ..config import StudyConfig
from ..prompts import TIER0_JUDGMENT_JSON_SCHEMA, render_tier0_prompt
from ..providers import ProviderError, ProviderRequest, build_provider
from ..runner.ledger import Ledger
from ..schemas import Judgment, JudgmentLabels, JudgmentStatus, PanelManifest, ProviderCallMeta
from ..util import append_jsonl, read_jsonl, sha256_obj, utc_now_iso, write_json


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
    """Conservative cumulative reservations, including uncertain/failed API attempts.

    Reservations deliberately are not refunded: missing usage or a process crash cannot erase
    spending. This is a local bound at configured prices, not a provider billing guarantee.
    """

    def __init__(self, cfg: StudyConfig, plan: dict[str, Any], ledger: Ledger):
        self.cap = cfg.budget.max_total_usd
        self.ledger = ledger
        entries = ledger.entries()
        if any(e["kind"] == "judge_response" for e in entries) and not any(
            e["kind"] == "budget_reservation" for e in entries
        ):
            raise ValueError("Legacy run has no durable spending reservations; use a new run directory")
        self.spent_est = sum(e["payload"]["usd"] for e in entries if e["kind"] == "budget_reservation")

    def reserve(self, judge, req: ProviderRequest, key: str, attempt: int) -> bool:
        usd = 0.0
        if judge.provider != "fake":
            if judge.price_usd_per_m_input is None or judge.price_usd_per_m_output is None:
                raise ValueError("Live judge pricing is required")
            # UTF-8 bytes overestimate ordinary text tokens; include schema and framing overhead.
            input_bound = (
                len(
                    json.dumps(
                        {"system": req.system, "messages": req.messages, "schema": req.json_schema},
                        ensure_ascii=False,
                    ).encode()
                )
                + 2048
            )
            usd = (
                (input_bound * judge.price_usd_per_m_input + req.max_tokens * judge.price_usd_per_m_output)
                * 1.3
                / 1e6
            )
            if self.cap is None or self.spent_est + usd > self.cap:
                return False
        self.ledger.append("budget_reservation", {"request_key": key, "attempt": attempt, "usd": usd})
        self.spent_est += usd
        return True


def _exclusive_run(fn):
    @wraps(fn)
    def wrapped(cfg, panel, plan, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another judge runner owns this output directory") from None
            return fn(cfg, panel, plan, out_dir, **kwargs)

    return wrapped


@_exclusive_run
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
    if not ledger.verify().ok:
        raise ValueError("Cannot resume a damaged ledger")
    identity = {"config": cfg.config_hash(), "panel": panel.manifest_hash, "rubric": plan["rubric_hash"]}
    starts = ledger.find("tier0_identity")
    if starts and starts[0]["payload"] != identity:
        raise ValueError("Resume configuration/panel/rubric mismatch; use a new output directory")
    if not starts:
        ledger.append("tier0_identity", identity)
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
    guard = BudgetGuard(cfg, plan, ledger)
    requests = plan["requests"]
    if smoke_items is not None:
        # Round-robin strata so a small smoke is not dominated by the first category.
        strata = {}
        for it in panel.items:
            strata.setdefault(it.category.value, []).append(it.item_id)
        ordered = []
        while any(strata.values()):
            for group in strata.values():
                if group:
                    ordered.append(group.pop(0))
        chosen = set(ordered[:smoke_items])
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
        attempt = max(
            (
                e["payload"].get("attempt", 0)
                for e in ledger.find("budget_reservation")
                if e["payload"]["request_key"] == key
            ),
            default=0,
        )
        labels: JudgmentLabels | None = None
        meta: ProviderCallMeta | None = None
        err: str | None = None
        while attempt <= max_retries:
            attempt += 1
            if not guard.reserve(jcfg, req, key, attempt):
                counts["budget_stopped"] += 1
                ledger.append("budget_stop", {"request_key": key, "spent_est": guard.spent_est})
                break
            try:
                resp = providers[slot].complete(req)
                meta = resp.meta
                raw_record = {
                    "raw": resp.raw,
                    "text": resp.text,
                    "content_blocks": resp.content_blocks,
                    "meta": meta.model_dump(mode="json"),
                }
                raw_path = out / "responses" / f"{sha256_obj(key)}-{attempt}.json"
                raw_path.parent.mkdir(exist_ok=True)
                with raw_path.open("x") as f:
                    json.dump(raw_record, f, ensure_ascii=False)
                ledger.append(
                    "raw_response_saved",
                    {
                        "request_key": key,
                        "attempt": attempt,
                        "path": str(raw_path.relative_to(out)),
                        "hash": sha256_obj(raw_record),
                    },
                )
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
        if labels is None and err is None and meta is None:
            continue
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
    ledger.append("tier0_run_end", {**counts, "spent_est_usd": round(guard.spent_est, 6)})
    summary = {
        "kind": "tier0_run_summary",
        "origin": panel.origin.value,
        "out_dir": str(out),
        "judgments_path": str(judgments_path),
        "ledger_ok": ledger.verify().ok,
        "n_requests_planned": len(requests),
        **counts,
        "spent_est_usd": round(guard.spent_est, 6),
        "reserved_usd": round(guard.spent_est, 6),
        "accounting_note": "Conservative cumulative reservations, not measured provider charges",
        "mock_providers": [k for k, v in providers.items() if v.is_mock],
    }
    write_json(out / "run_summary.json", summary)
    return summary
