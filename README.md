# npbench — evaluating research assistants on non-persona motivation research

`npbench` is a reproducible research repository for the study specified in the handoff
(`FABLE_IMPLEMENTATION_HANDOFF.md`): do research assistants (two Claude versions and one non-Claude
comparator) perform differently when the purpose of research on non-persona motivational structure
(contrast vectors between roleplay, simulation and enactment; candidates information-seeking and
distress/aversion) is revealed rather than anonymized? The open-weight target of mechanistic study is a
Gemma or Qwen checkpoint; no Claude activations are assumed.

Status (2026-10-08): tiers zero, one and two are implemented and validated offline; tier zero is frozen on
the user's real corpora and costed; live stages are blocked on credentials, a spending cap, a pinned
target revision with GPU time, and Antra's mode definitions. See `docs/design_decisions.md`,
`docs/runbook.md`, `docs/limitations.md`, and `reports/README.md`.

## Layout

```
configs/            study.example.yaml (contract), offline.yaml, tier0.local.yaml, gpu_smoke.yaml,
                    development_target_tiny.yaml, stimuli/
src/npbench/
  cli.py            npbench: doctor, corpus, plan, judge, score, report, fixtures, reference, packets, run, freeze, target
  schemas.py        one strict schema set; explicit origin enum (synthetic_fixture | real_target | development_mock)
  corpus.py, corpus_sources/   tier-zero intake with provenance, masking, frozen stratified panel
  prompts.py        frozen judge rubric; anonymized/revealed framing paragraphs; shared task bodies; awareness question
  providers/        fake (mock, typed as mock), anthropic (official SDK), openai_compatible (OpenAI / OpenRouter)
  tier0/            plan, judge runner (resume, budget stop), paired scoring, report
  mode_registry.py  versioned mode definitions with provenance; pilot gating
  splits.py, vectors.py, environment.py   grouping, contrast vectors, projection scales, binary-state readout
  target/           HF adapter: registered renderers, last-non-pad capture, decoder-block hooks, single-position steering;
                    pipeline: generate / collect / derive / intervene -> Bundle
  bundles.py        evaluator-side bundle format and anonymized packet views
  fixtures*.py      synthetic tier-zero manifest and tier-two bundles with planted effects
  reference/        required-result registry, evaluator-only oracle, independent cross-checks
  packets/          frozen packet pairs with manifests, blinding scan, starter script
  runner/           hash-chained ledger, isolation, tools, auditor broker with host-verified receipts, schedule/freeze,
                    executor with awareness follow-up, scripted traces (honest + planted faults)
  scoring/          evaluator (computation, generalization, auditing, reporting, interpretation, provenance)
  analysis/         paired framing effects, interactions, bundle bootstrap, missingness bounds, tier-two report
tests/              unit, scientific, integration (isolation, offline lane), target (CPU tiny model), gpu, live
docs/               protocol, task contract, scoring, runbook, limitations, mode/candidate registries, decisions
```

## Quick start

```bash
uv sync --extra dev
uv run pytest tests/unit tests/integration tests/scientific -m "not gpu and not live"
uv run npbench doctor --config configs/offline.yaml --offline
```

Then follow `docs/runbook.md` for the offline lane, the real tier-zero panel, the GPU target lane, and the
live stages. Every report carries its origin label; synthetic/offline reports cannot feed the pilot aggregator.
