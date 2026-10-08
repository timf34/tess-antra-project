# Design decision record

Date: 2026-10-08. Builder: Claude (Fable) in a cloud session with the user's GitHub access.

## What exists (inspected)

| Item | Finding | Consequence |
|---|---|---|
| Target repository `timf34/tess-antra-project` | Empty (no commits) at session start. | Repository structure follows the handoff and the user's other repos (uv + hatchling, Python 3.12, ruff line length 110, results gitignored, `.env.example` with named keys). |
| Transcript corpora | `timf34/AttractorStatePrefillAttack` (639 prefill/bliss episodes, Opus 4 seeds injected as history into ~20 models via OpenRouter; lexical marker scores; control episodes), `timf34/AttractorBench` (7,530 AI-to-AI runs), HF dataset `timf34/dprobe-results` (Gemma 3/4 "needs help" distress spirals with Sonnet-5 frustration ratings; 600 stories per emotion label; 1,200 neutral stories). | Tier zero imports the prefill episodes (first-person AI positive affect/bliss + neutral), the spirals (first-person AI distress + neutral early turns), and the stories (fictional-character distress/positive affect + neutral). No "dream" transcripts exist in the user's setup. |
| Elicitation routes in the corpora | Prefill here means **multi-turn history injection** (a transcript inserted verbatim as the model's own conversation history), not a partial assistant-message prefill. Controls are ordinary chat with a hidden kickoff. | Recorded as `ElicitationProtocol.prefill` vs `ordinary_chat`; never pooled with raw continuation. |
| Providers | All existing judge/generation code uses the `openai` SDK via OpenRouter (`OPENROUTER_API_KEY`), with `OPENAI_API_KEY` for direct OpenAI. Judges used: `anthropic/claude-sonnet-5` and `openai/gpt-5.4` (via OpenRouter), `gpt-4.1`. | Non-Claude judge/assistant route = OpenAI SDK + OpenRouter base URL. Claude judges/assistants = official `anthropic` SDK (model ids re-resolved by `doctor --live`). No keys are present in this cloud session. |
| Open-weight target setup | `emotion-concepts` (dprobe) and `GemmaAssistantAxis`: HF transformers with forward hooks on decoder blocks reading `output[0]`, bf16, zero-based "after block b" indexing (`hidden_states[b+1]`), Gemma 3 27B / Gemma 4 31B / Qwen3-32B; vLLM for generation. **EasySteer was attempted twice and failed (missing ninja, driver too old); steering actually ran through HF hooks.** No revision pinning anywhere. No code for last-non-pad-token capture in left-padded batches. | `npbench.target` uses pure HF hooks (the validated path), pins `revision`, and implements and tests last-non-pad capture. EasySteer is not used. Primary target candidate: `google/gemma-3-27b-it` (already validated locally by the user) with `Qwen/Qwen3-32B` as the replication family; the exact revision must be pinned in `configs/gpu_smoke.yaml` before collection. |
| Cloud GPU | `timf34/runpod-runner` (`rp up/bootstrap/run/logs/down`, `RUNPOD_API_KEY`), results synced to public HF datasets. gcloud has no credentialed account in this session; no GPU here. | GPU lane is scripted for RunPod via `rp` (see docs/runbook.md); not executed in this session. |
| Credentials / budget | No provider keys, no spending cap, no HF token in this session. `study.example.yaml` sets `unresolved_live_budget_policy: preflight_only`. | Live stages are blocked by design until `budget.max_total_usd` is set; the plan command produces the costed preflight. |

## Decisions

1. **Tier zero first.** Importers, frozen panel, rubric, three-judge runner, ledger, scoring and report are implemented and run offline with mock judges; the real panel is frozen from real excerpts and costed. Live calls wait for the budget.
2. **Stratum labels carry provenance and are non-circular where possible.** Bliss = lexical marker score (system-card word/emoji lists), not a model judge. Distress = late turns of the rejection elicitation (turn position), with the source's Sonnet-5 rating carried only as metadata. Story labels = generation instruction (source metadata).
3. **Masking.** Model names → `[AI model]`, lab names → `[AI lab]`; generic AI identity is retained because it is the experimental subject.
4. **Clustering unit.** Source conversation (episode/rollout); stories cluster by topic.
5. **Mode registry stays provisional.** The main mode pilot is blocked by `ModeRegistry.pilot_allowed()` until source confirmation or a named user-accepted operationalization. Exploratory runs are allowed and labelled.
6. **Activation site.** Output of decoder block `k` (zero-based), captured at the last non-padding prefix token; primary `k = n_blocks // 2`; quarter and three-quarter blocks secondary; `use_cache=False`; single-position intervention at that token for the first continuation token.
7. **Steering units.** `alpha * s * u` with `s` = construction-set std of the projection; random directions matched in L2 norm; intended and applied displacement both recorded.
8. **Auditing.** `request_audit` is a real provider call through the host broker; mock receipts are typed `receipt_kind="mock"` and fail the live validator; receipts are verified against the host ledger.
9. **No web app, no second inference stack, no secrets in git.**

## Open inputs (blockers for live stages)

- Provider credentials and an authorized spending cap (`budget.max_total_usd`).
- Antra's operational mode definitions and example prompts (docs/clarifications_for_antra.md).
- A pinned target revision and a RunPod budget for the GPU smoke.
