"""Explicitly synthetic, 15-rating OpenRouter engineering smoke, with a $1 reservation cap.

Run: uv run --env-file .env python scripts/local/judge_smoke.py
Model availability/prices are resolved live; outputs resume under runs/local_judge_smoke.
"""

import json
from pathlib import Path

from npbench.config import StudyConfig
from npbench.corpus import load_panel, sample_panel, write_panel
from npbench.corpus_sources.jsonl_manifest import import_jsonl_manifest
from npbench.fixtures import build_tier0_synthetic_manifest
from npbench.providers import build_provider
from npbench.schemas import Origin
from npbench.tier0.plan import build_plan
from npbench.tier0.run import run_judges
from npbench.util import write_json


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "runs/local_judge_smoke"
    source = root / "artifacts/local_smoke/source.jsonl"
    if not source.exists():
        build_tier0_synthetic_manifest(source, per_category=3)
    provider = build_provider(
        "openai_compatible",
        "openai/gpt-5.4",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )
    models = {m["id"]: m for m in provider.list_models()}
    judges = []
    for slot, family, model in [
        ("claude_a", "claude", "anthropic/claude-opus-5.5"),
        ("claude_b", "claude", "anthropic/claude-sonnet-5.5"),
        ("non_claude", "non_claude", "openai/gpt-5.4"),
    ]:
        price = models[model]["pricing"]
        judges.append(
            dict(
                slot_id=slot,
                family=family,
                provider="openai_compatible",
                model_id=model,
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
                settings={"max_tokens": 1024},
                price_usd_per_m_input=float(price["prompt"]) * 1e6,
                price_usd_per_m_output=float(price["completion"]) * 1e6,
            )
        )
    cfg = StudyConfig.model_validate(
        {
            "study": {"stage": "development_smoke"},
            "budget": {"max_total_usd": 1.0},
            "runner": {"max_retries_per_request": 0},
            "tier0": {
                "corpus_out": str(source),
                "max_items": 5,
                "max_items_per_category": 1,
                "repetitions": 1,
                "judges": judges,
            },
        }
    )
    pairs = list(import_jsonl_manifest(source, origin=Origin.synthetic_fixture))
    panel = sample_panel(
        cfg,
        {r.record_id: r for r, _ in pairs},
        [c for _, cs in pairs for c in cs],
        origin=Origin.synthetic_fixture,
    )
    panel_path = source.parent / "panel.json"
    if panel_path.exists():
        panel = load_panel(panel_path)
    else:
        write_panel(panel, panel_path)
    plan = build_plan(cfg, panel)
    write_json(source.parent / "config.json", cfg.model_dump(mode="json"))
    write_json(source.parent / "plan.json", plan)
    print(json.dumps(run_judges(cfg, panel, plan, out, allow_live=True), indent=2))


if __name__ == "__main__":
    main()
