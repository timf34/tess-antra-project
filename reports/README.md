# Reports

Each report states its origin label in its first lines. Reports produced from `configs/offline.yaml`
say SYNTHETIC / OFFLINE DATA and cannot feed the pilot aggregator.

| report | origin | status |
|---|---|---|
| `offline_tier0/tier0_report.md` | synthetic_fixture, mock judges | generated in the build session (pipeline validation) |
| `offline/tier2_report.md` | synthetic_fixture, mock assistants and reviewers | generated in the build session (evaluator validation) |
| `tier0/` | real corpus, live judges | not generated: live calls blocked on credentials and budget |
| `development/` | real bundle, live assistants | not generated: blocked on GPU run, credentials, budget |
| `pilot/` | real_target | not generated: blocked on mode-definition confirmation |
| `matched_affect_exploratory.md` / `.json` | real_target, live assistants and auditor | generated 2026-10-09: 18 researcher runs on one exploratory bundle; evaluator amendments logged; transcripts private |
