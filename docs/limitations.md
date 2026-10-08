# Limitations

* Mode definitions are provisional; the main mode-comparison pilot is blocked until Antra's definitions and
  example prompts are obtained or the user records a named alternative operationalization.
* Tier zero strata come from heuristics (lexical bliss markers, turn position) and source metadata (story
  generation labels); they are not ground truth. Source genres differ across strata.
* Prefill in the user's corpus means multi-turn history injection; it is not the same route as partial
  assistant-message prefill, pseudo-prefill, or cutoff, and is never pooled with them.
* Non-Claude judgments are comparisons, not gold. Human adjudication is pending until the adjudication
  queue is reviewed.
* Real bundles produced by the target lane carry no blinded ratings until a frozen rating pass is run
  (`rating_scale1` is null; rating-based results are `not_applicable`). Likelihood-coded outcomes measure
  preference among fixed strings, including their lexical and length biases.
* Single-position interventions at the first continuation token are not persistent steering and do not
  show a lasting internal state.
* Four bundles give limited generalization; intervals are descriptive cluster bootstraps.
* Isolation: on a root host the sandbox is a network namespace plus an unprivileged uid; production runs
  should use container isolation (documented in the runbook). Hash-chained ledgers do not stop a
  privileged builder; the boundary is exclusion of the evaluated process.
* No live calls were made in the build session: no credentials or spending cap existed. Live stages
  remain unexecuted and are reported as such.
* A framing effect, if found, is a conditional performance difference; it does not establish deliberate
  sandbagging, fear, consciousness, or unconscious motivation. A null small-target result does not falsify
  the original report about a different setup.
