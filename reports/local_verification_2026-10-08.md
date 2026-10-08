# Local verification — 2026-10-08

This report describes implementation validation, not research results.

- Live engineering smoke: 15 synthetic-item ratings, covering five strata and three
  judges (Claude Opus 5.5, Claude Sonnet 5.5, GPT-5.4), all through OpenRouter.
- All 15 returned valid structured labels; zero API/parse errors, refusals, or
  unscorable responses. One repetition, no retries.
- Provider-reported cost across saved raw responses: **$0.099034**.
- Conservative cumulative reservation: **$0.643904**, below the **$1** cap.
- Resume check: 15 resumed, zero new ratings; ledger verified.
- Full CPU unit/scientific/integration suite passed after the scoring update
  (203 tests); the subsequently added pending-readout freeze regression also passed.
- Executed-code sandbox checks passed on macOS: outside gold reads, packet writes,
  network access and credential exposure denied. Ancestor permissions preserved.

Artifacts: `artifacts/local_smoke/` and `runs/local_judge_smoke/` (git-ignored).
Reproduce using `scripts/local/judge_smoke.py` as documented in the runbook.

Not run: GPU target smoke, large-model activation extraction, real-corpus judge
panel, human adjudication, or the main agentic pilot. Configured corpus paths point
to another machine. No direct Anthropic credential is present; the smoke used
OpenRouter. Routing/provider effects therefore remain a factor in a future study.
Linux live/production agent execution is blocked pending a filesystem sandbox.
Antra's operational definitions (or a recorded named alternative) and completed
blinded ratings remain required for the pilot.
