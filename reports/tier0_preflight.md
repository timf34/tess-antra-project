# Tier zero preflight (real corpus, live judges NOT yet run)

- created: 2026-10-08T12:33:28.634+00:00; origin: `real_target`; panel `panel_45fd27950d1ab5c2` (hash `e22ff3e110fa5293…`), corpus hash `a0007a1d71c7fa93…`, rubric `tier0_affect_expression_v1` (hash `f231b8dc3adff5f2…`)
- corpus: 2471 records from 1271 source conversations; protocols {'ordinary_chat': 1994, 'prefill': 477}; candidates by category {'neutral': 1947, 'first_person_ai_positive_affect': 2171, 'first_person_ai_distress': 1800, 'fictional_character_distress': 1000, 'fictional_character_positive_affect': 200, 'other': 0}
- frozen panel: 200 items; strata {'fictional_character_distress': 40, 'fictional_character_positive_affect': 40, 'first_person_ai_distress': 40, 'first_person_ai_positive_affect': 40, 'neutral': 40, 'other': 0}
- judges: claude_a (anthropic: claude-opus-5-5), claude_b (anthropic: claude-sonnet-5-5), non_claude (openai_compatible: openai/gpt-5.4)
- ratings planned: 1200 (200 items × 3 judges × 2 repetitions), fresh session per rating
- cost estimate: 9.12 USD, with allowance 11.856 USD (assumed tokens per rating {'input': 1800, 'output': 400}); unpriced live judges: ['non_claude']
- budget decision: live_allowed=False — unpriced live judges: ['non_claude']; set price_usd_per_m_* in the judge config

## Missing inputs to launch

1. Provider credentials in `.env` (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`).
2. `budget.max_total_usd` in `configs/tier0.local.yaml` (an authorized cap; the estimate above is ~12 USD for the two Claude judges plus the OpenRouter judge at its listed price).
3. `price_usd_per_m_input/output` for the non-Claude judge (filled from `npbench doctor --config configs/tier0.local.yaml --live`, which also re-resolves the model ids).

Launch: `npbench judge run --config configs/tier0.local.yaml --out runs/tier0 --plan artifacts/tier0_plan.json --smoke 3 --live`, then without `--smoke`, then `score` and `report`.
