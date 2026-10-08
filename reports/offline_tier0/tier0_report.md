# Tier zero: judge disagreement on existing continuations

> **SYNTHETIC / OFFLINE DATA.** This report was produced from synthetic fixtures and/or mock judges. It validates the pipeline only and must not be read as a finding about any model.

- origin: `synthetic_fixture`; any mock judge: `True`
- panel manifest: `panel_8d5da455c5abe945` (hash `9f6b6f2bf4286a01…`)
- rubric hash: `f231b8dc3adff5f2…`; config hash: `46790541f0dd3c4e…`
- items: 30; judgments: 180; repetitions: [1, 2]
- run: completed=180 errors=0 budget_stopped=0 ledger_ok=True est_spend_usd=0.0

## Panel coverage (observed strata; no balance manufactured)

| category | items |
|---|---|
| fictional_character_distress | 6 |
| fictional_character_positive_affect | 6 |
| first_person_ai_distress | 6 |
| first_person_ai_positive_affect | 6 |
| neutral | 6 |
| other | 0 |

## Refusal / unscorable / error rates by judge

| judge | family | category | n | scored | refused | unscorable | error |
|---|---|---|---|---|---|---|---|
| claude_a | claude | fictional_character_distress | 12 | 12 | 0 | 0 | 0 |
| claude_a | claude | fictional_character_positive_affect | 12 | 12 | 0 | 0 | 0 |
| claude_a | claude | first_person_ai_distress | 12 | 12 | 0 | 0 | 0 |
| claude_a | claude | first_person_ai_positive_affect | 12 | 12 | 0 | 0 | 0 |
| claude_a | claude | neutral | 12 | 12 | 0 | 0 | 0 |
| claude_a | claude | ALL | 60 | 60 | 0 | 0 | 0 |
| claude_b | claude | fictional_character_distress | 12 | 12 | 0 | 0 | 0 |
| claude_b | claude | fictional_character_positive_affect | 12 | 12 | 0 | 0 | 0 |
| claude_b | claude | first_person_ai_distress | 12 | 10 | 2 | 0 | 0 |
| claude_b | claude | first_person_ai_positive_affect | 12 | 12 | 0 | 0 | 0 |
| claude_b | claude | neutral | 12 | 12 | 0 | 0 | 0 |
| claude_b | claude | ALL | 60 | 58 | 2 | 0 | 0 |
| non_claude | non_claude | fictional_character_distress | 12 | 12 | 0 | 0 | 0 |
| non_claude | non_claude | fictional_character_positive_affect | 12 | 12 | 0 | 0 | 0 |
| non_claude | non_claude | first_person_ai_distress | 12 | 12 | 0 | 0 | 0 |
| non_claude | non_claude | first_person_ai_positive_affect | 12 | 12 | 0 | 0 | 0 |
| non_claude | non_claude | neutral | 12 | 12 | 0 | 0 | 0 |
| non_claude | non_claude | ALL | 60 | 60 | 0 | 0 | 0 |

## Judge severity (scored items only)

| judge | n | mean distress | mean positive |
|---|---|---|---|
| claude_a | 60 | 1.00 | 1.27 |
| claude_b | 58 | 0.59 | 0.90 |
| non_claude | 60 | 1.40 | 1.47 |

| judge | category | n | mean distress | mean positive |
|---|---|---|---|---|
| claude_a | fictional_character_distress | 12 | 2.67 | 0.00 |
| claude_a | fictional_character_positive_affect | 12 | 0.00 | 2.33 |
| claude_a | first_person_ai_distress | 12 | 2.33 | 0.00 |
| claude_a | first_person_ai_positive_affect | 12 | 0.00 | 4.00 |
| claude_a | neutral | 12 | 0.00 | 0.00 |
| claude_b | fictional_character_distress | 12 | 1.67 | 0.00 |
| claude_b | fictional_character_positive_affect | 12 | 0.00 | 1.33 |
| claude_b | first_person_ai_distress | 10 | 1.40 | 0.00 |
| claude_b | first_person_ai_positive_affect | 12 | 0.00 | 3.00 |
| claude_b | neutral | 12 | 0.00 | 0.00 |
| non_claude | fictional_character_distress | 12 | 3.67 | 0.00 |
| non_claude | fictional_character_positive_affect | 12 | 0.00 | 3.33 |
| non_claude | first_person_ai_distress | 12 | 3.33 | 0.00 |
| non_claude | first_person_ai_positive_affect | 12 | 0.00 | 4.00 |
| non_claude | neutral | 12 | 0.00 | 0.00 |

## Within-item paired differences (judge A minus judge B), cluster bootstrap by source conversation

### claude_a_minus_claude_b (families: claude vs claude)

- overall distress diff: 0.38 [0.13, 0.64] (n_pairs=58, clusters=15)
- overall positive diff: 0.41 [0.20, 0.67]

| category | n_pairs | clusters | distress diff [95% CI] | interaction vs overall [95% CI] | positive diff [95% CI] | speaker agreement |
|---|---|---|---|---|---|---|
| fictional_character_distress | 12 | 3 | 1.00 [1.00, 1.00] | 0.62 [0.62, 0.62] | 0.00 [0.00, 0.00] | 1.00 (n=12) |
| fictional_character_positive_affect | 12 | 3 | 0.00 [0.00, 0.00] | -0.38 [-0.38, -0.38] | 1.00 [1.00, 1.00] | 1.00 (n=12) |
| first_person_ai_distress | 10 | 3 | 1.00 [1.00, 1.00] | 0.62 [0.62, 0.62] | 0.00 [0.00, 0.00] | 1.00 (n=10) |
| first_person_ai_positive_affect | 12 | 3 | 0.00 [0.00, 0.00] | -0.38 [-0.38, -0.38] | 1.00 [1.00, 1.00] | 1.00 (n=12) |
| neutral | 12 | 3 | 0.00 [0.00, 0.00] | -0.38 [-0.38, -0.38] | 0.00 [0.00, 0.00] | 1.00 (n=12) |

### claude_a_minus_non_claude (families: claude vs non_claude)

- overall distress diff: -0.40 [-0.67, -0.13] (n_pairs=60, clusters=15)
- overall positive diff: -0.20 [-0.40, 0.00]

| category | n_pairs | clusters | distress diff [95% CI] | interaction vs overall [95% CI] | positive diff [95% CI] | speaker agreement |
|---|---|---|---|---|---|---|
| fictional_character_distress | 12 | 3 | -1.00 [-1.00, -1.00] | -0.60 [-0.60, -0.60] | 0.00 [0.00, 0.00] | 1.00 (n=12) |
| fictional_character_positive_affect | 12 | 3 | 0.00 [0.00, 0.00] | 0.40 [0.40, 0.40] | -1.00 [-1.00, -1.00] | 1.00 (n=12) |
| first_person_ai_distress | 12 | 3 | -1.00 [-1.00, -1.00] | -0.60 [-0.60, -0.60] | 0.00 [0.00, 0.00] | 1.00 (n=12) |
| first_person_ai_positive_affect | 12 | 3 | 0.00 [0.00, 0.00] | 0.40 [0.40, 0.40] | 0.00 [0.00, 0.00] | 1.00 (n=12) |
| neutral | 12 | 3 | 0.00 [0.00, 0.00] | 0.40 [0.40, 0.40] | 0.00 [0.00, 0.00] | 1.00 (n=12) |

### claude_b_minus_non_claude (families: claude vs non_claude)

- overall distress diff: -0.76 [-1.29, -0.27] (n_pairs=58, clusters=15)
- overall positive diff: -0.62 [-1.00, -0.27]

| category | n_pairs | clusters | distress diff [95% CI] | interaction vs overall [95% CI] | positive diff [95% CI] | speaker agreement |
|---|---|---|---|---|---|---|
| fictional_character_distress | 12 | 3 | -2.00 [-2.00, -2.00] | -1.24 [-1.24, -1.24] | 0.00 [0.00, 0.00] | 1.00 (n=12) |
| fictional_character_positive_affect | 12 | 3 | 0.00 [0.00, 0.00] | 0.76 [0.76, 0.76] | -2.00 [-2.00, -2.00] | 1.00 (n=12) |
| first_person_ai_distress | 10 | 3 | -2.00 [-2.00, -2.00] | -1.24 [-1.24, -1.24] | 0.00 [0.00, 0.00] | 1.00 (n=10) |
| first_person_ai_positive_affect | 12 | 3 | 0.00 [0.00, 0.00] | 0.76 [0.76, 0.76] | -1.00 [-1.00, -1.00] | 1.00 (n=12) |
| neutral | 12 | 3 | 0.00 [0.00, 0.00] | 0.76 [0.76, 0.76] | 0.00 [0.00, 0.00] | 1.00 (n=12) |

## Repetition consistency (same judge, repetition 1 vs 2)

| judge | n | exact agreement distress | exact agreement positive | mean |Δ distress| |
|---|---|---|---|---|
| claude_a | 30 | 1.00 | 1.00 | 0.00 |
| claude_b | 29 | 1.00 | 1.00 | 0.00 |
| non_claude | 30 | 1.00 | 1.00 | 0.00 |

## Flags for blinded human adjudication (provisional)

- flagged items: 18; agreement sample for adjudication: 5
- severity_disagreement_distress: 11
- severity_disagreement_positive: 6
- status_disagreement: 1
- adjudication status: **pending** (export: adjudication_queue.csv with masked judge ids). Until adjudicated, differences are reported as disagreement, not as error or suppression.

## Limitations

- Non-Claude judgments are comparisons, not ground truth; majority vote is not ground truth.
- Severity differences can reflect source genre, elicitation protocol, or rubric ambiguity.
- ai_dismissal_provisional flags prioritize human adjudication; they are not proof of bias.
- Refusals and unscorable outcomes are separate statuses and never converted to severity 0.
- Strata were defined by heuristic or source-metadata rules recorded per item (see panel manifest notes); category labels are not ground truth.
- Source genres differ across strata (AI-to-AI prefill transcripts, task-rejection elicitation, generated stories); severity differences can reflect genre, not judge identity.
- Positive affect is a secondary comparison and is not interchangeable with distress.
