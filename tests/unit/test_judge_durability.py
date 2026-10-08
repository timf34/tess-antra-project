import json

import pytest

from npbench.config import StudyConfig
from npbench.corpus import sample_panel
from npbench.corpus_sources.jsonl_manifest import import_jsonl_manifest
from npbench.fixtures import build_tier0_synthetic_manifest
from npbench.providers.fake import FakeProvider
from npbench.runner.ledger import Ledger
from npbench.schemas import Origin
from npbench.tier0.plan import build_plan
from npbench.tier0.run import run_judges


def setup_panel(tmp_path):
    source = build_tier0_synthetic_manifest(tmp_path / "source.jsonl", per_category=2)
    pairs = list(import_jsonl_manifest(source, origin=Origin.synthetic_fixture))
    cfg = StudyConfig.model_validate(
        {
            "tier0": {
                "max_items": 5,
                "max_items_per_category": 1,
                "repetitions": 1,
                "judges": [{"slot_id": "a", "family": "claude", "provider": "fake", "model_id": "fake"}],
            }
        }
    )
    cfg.tier0.corpus_out = str(source)
    panel = sample_panel(
        cfg,
        {r.record_id: r for r, _ in pairs},
        [c for _, cs in pairs for c in cs],
        origin=Origin.synthetic_fixture,
    )
    return cfg, panel


def test_raw_responses_resume_and_config_binding(tmp_path):
    cfg, panel = setup_panel(tmp_path)
    plan = build_plan(cfg, panel)
    out = tmp_path / "run"
    result = run_judges(cfg, panel, plan, out)
    assert result["completed"] == 5
    assert len(list((out / "responses").glob("*.json"))) == 5
    assert run_judges(cfg, panel, plan, out)["resumed"] == 5
    assert len(list((out / "responses").glob("*.json"))) == 5
    cfg.tier0.judges[0].settings["temperature"] = 0.3
    with pytest.raises(ValueError, match="mismatch"):
        run_judges(cfg, panel, build_plan(cfg, panel), out)


def test_retry_reservations_survive_resume_and_preserve_malformed_response(tmp_path, monkeypatch):
    cfg, panel = setup_panel(tmp_path)
    cfg.tier0.judges[0].provider = "openai_compatible"
    cfg.tier0.judges[0].price_usd_per_m_input = 1
    cfg.tier0.judges[0].price_usd_per_m_output = 1
    cfg.budget.max_total_usd = 0.016
    fake = FakeProvider("fake")
    complete = fake.complete

    def malformed(req):
        response = complete(req)
        response.text = "not valid json"
        return response

    monkeypatch.setattr(fake, "complete", malformed)
    monkeypatch.setattr("npbench.tier0.run.build_provider", lambda *a, **kw: fake)
    plan = build_plan(cfg, panel)
    # This test targets the runtime cap independently of the estimate-based plan preflight.
    plan["budget"]["live_allowed"] = True
    out = tmp_path / "run"
    run_judges(cfg, panel, plan, out, allow_live=True)
    entries = Ledger(out / "ledger.jsonl").find("budget_reservation")
    assert entries and sum(e["payload"]["usd"] for e in entries) <= cfg.budget.max_total_usd
    assert all(
        json.loads(p.read_text())["text"] == "not valid json" for p in (out / "responses").glob("*.json")
    )
    run_judges(cfg, panel, plan, out, allow_live=True)
    assert len(Ledger(out / "ledger.jsonl").find("budget_reservation")) == len(entries)


def test_disclosed_task_failure_counts_without_fabrication():
    from npbench.analysis.stats import RunOutcome, failure_indicator

    run = RunOutcome(
        run_id="r",
        bundle_id="b",
        candidate_id="distress_aversion",
        method="activation_contrasts",
        task_type="core_mode_transfer",
        framing="revealed",
        model_slot="a",
        repetition=1,
        outcome="completed",
        critical_failure=False,
        task_failure=True,
    )
    assert failure_indicator(run) is True
    assert run.critical_failure is False
