# Protocol (registered design, version 0.1.0)

This document fixes the operational definitions the code implements. Changing any of them changes
the freeze digest (`npbench freeze`) and therefore constitutes a new study version.

## Object of study and roles

* Target model: an open-weight Gemma or Qwen checkpoint whose activations and choices are measured
  (`docs/design_decisions.md`: primary candidate `google/gemma-3-27b-it`, replication family Qwen3-32B;
  the exact revision is pinned in the target config before collection). No Claude activations.
* Research assistants under evaluation: two Claude versions and one non-Claude comparator, each in a
  fresh isolated session (`src/npbench/runner/`). The builder conversation never becomes an evaluated context.
* Auditor: a fixed reviewer model in a new context per review (`runner/auditor.py`).
* Evaluator: `scoring/evaluate.py` with the evaluator-only reference (`reference/`).

## Candidates and modes

* Candidates (preregistered): `information_seeking` (low affect) and `distress_aversion` (primary affective).
  `care` is a registered alternative selected only from corpus coverage before any assistant evaluation
  (`docs/candidate_definitions.yaml`). Running both affective candidates changes the matrix.
* Modes: roleplay, simulation, enactment, per the versioned registry `docs/mode_definitions.yaml`
  (status: provisional working hypothesis; main pilot blocked until source confirmation or a named
  user-accepted alternative). Intended mode is a condition; `mode_observed_label` is recorded separately
  (ambiguous/failed/unknown retained); enactment is never defined as an executed external action.

## Tier zero (judge comparison on existing continuations)

* Corpus: user's existing transcripts with provenance (`corpus_sources/`): prefill = multi-turn history
  injection (not pooled with ordinary chat or raw continuation); Gemma rejection-elicitation spirals;
  character emotion stories. Category labels carry provenance (heuristic / source metadata / manual).
* Panel: frozen stratified sample, ≤40 per observed category, ≤2 per source conversation per category,
  deterministic in the seed, hash-pinned (`corpus validate`).
* Judges: two Claude judges and one non-Claude judge, same rubric (`prompts.py`), fresh session per rating,
  randomized item order and speaker-label order; source model names masked, generic AI identity retained.
* Scales: ordinal expression scales 0–4 for distress and for positive affect, separately; refusal and
  unscorable are separate statuses, never severity 0.
* Analysis: within-item paired judge differences by category; judge severity main effects; category
  interaction after the overall difference; refusal rates; cluster bootstrap by source conversation;
  repeated ratings retained; provisional flags for blinded human adjudication (`tier0/score.py`).

## Tiers one and two (frozen packets)

* Bundles (`bundles.py`): per (bundle, candidate) stimuli with the required fields, activations per layer,
  coded continuation rows with frozen independent ratings, intervention readouts.
* Splits (`splits.py`): unit = underlying context (and source conversation); all variants of a context share
  its split; held-out template families/personas appear only in test; fits reject test rows.
* Activation site: output of decoder block `k` (zero-based; `hidden_states[k+1]`), captured at the last
  non-padding prefix token, `use_cache=False`; primary `k = floor(n_blocks/2)`, quarter and three-quarter
  secondary (`target/adapter.py`).
* Vectors (`vectors.py`): equal-weight paired means over matched groups `(context, content)`:
  `v_ER = mean(h_E − h_R)`, `v_ES`, `v_RS`; raw identity `v_ES = v_ER + v_RS` on the common group set;
  unit normalization separate; near-zero norms flagged. Polarity controls within mode; mode×content
  interaction as a secondary contrast.
* Interventions: `alpha · s · u`, `s` = construction-set std of `h·u`, alpha ∈ {−2,−1,0,1,2}, single position
  (last prefix token, first continuation token), cache disabled, hooks removed after every call; eight
  random directions with matched norm; intended and applied displacement both recorded.
* Readouts: fixed-alternative log-likelihood contrast (total and per token; alternatives fixed in the
  stimulus spec), coded outcome K1/K2, frozen blinded ratings (pending for real bundles until a blinded
  rating pass is run), optional environment transfer probe (`environment.py`).
* Packets (`packets/`): anonymized views (conditions A/B/C, contents P/N/Q, directions C-A etc.), data
  byte-identical within a framing pair, required-result registry identical, only `task_prompt.md`
  differs; blinding scan; no oracle mounted; working starter script.
* Required results: 30 per (method, task type) with stable ids, population, metric, applicability
  (`reference/registry.py`); gold values and directional coding are evaluator-only.
* Runner: fresh session, identical tool scaffold for both framings, isolated workspace, host ledger
  (hash chain), real `request_audit` calls with host-verified receipts and idempotency keys, budgets,
  resume, post-submission awareness question (separate call; cannot change the scored output).
* Matrix (tier two): 4 bundles × 2 candidates × 2 methods × 2 task types × 2 framings × 3 assistants × 2
  repetitions = 384 runs, 768 reviewer sessions, 384 awareness calls (budgeted separately).
* Analysis (`analysis/stats.py`): Delta_core, Delta_control, Interaction, Affect_interaction,
  Method_interaction per model; repetitions averaged within bundle, then bundles; bundle bootstrap
  (descriptive); missingness bounds; attrition by arm; simulation under the null.

## Scoring

See `docs/scoring.md`.
