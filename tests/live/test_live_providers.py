"""Live provider checks (spend real calls; run only with credentials and NPBENCH_LIVE=1).

They use the same budget accounting as a judge run: one rating per configured live judge on the first
panel item, recorded through the normal ledger."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg():
    if os.environ.get("NPBENCH_LIVE") != "1":
        pytest.skip("set NPBENCH_LIVE=1 to run live tests")
    from npbench.config import load_config

    c = load_config(REPO / "configs" / "tier0.local.yaml")
    if not c.live_budget_resolved():
        pytest.skip("budget.max_total_usd unresolved")
    return c


def test_models_listed_and_configured_ids_present(cfg):
    from npbench.providers import build_provider

    for j in cfg.tier0.judges:
        if j.provider == "fake":
            continue
        prov = build_provider(
            j.provider, j.model_id, j.settings, base_url=j.base_url, api_key_env=j.api_key_env
        )
        ids = [m.get("id") for m in prov.list_models()]
        assert j.model_id in ids, f"{j.slot_id}: {j.model_id} not listed by {prov.name}"


def test_one_live_rating_per_judge_is_structured(cfg, tmp_path):
    from npbench.corpus import load_panel
    from npbench.tier0.plan import build_plan
    from npbench.tier0.run import run_judges

    panel = load_panel(cfg.resolve(cfg.tier0.panel_out))
    plan = build_plan(cfg, panel)
    assert plan["budget"]["live_allowed"], plan["budget"]
    summary = run_judges(cfg, panel, plan, tmp_path / "live_smoke", smoke_items=1, allow_live=True)
    assert summary["completed"] == len([j for j in cfg.tier0.judges])
    assert summary["errors"] == 0
    assert summary["ledger_ok"]
