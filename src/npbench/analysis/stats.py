"""Pure-NumPy primary analysis of the framing study: failure rates, paired framing effects,
bundle-cluster bootstrap, missingness bounds and a null simulation.

The design is a 2 x 2 (task_type x framing) inside each *bundle*, the cluster unit, crossed with
candidate, method and model slot. Every estimate aggregates in the same order:

1. repetitions are averaged within a bundle cell (``cell_rate``), so a cell with three repetitions
   and a cell with one repetition each count once;
2. deltas (revealed minus anonymized) are formed inside the bundle, which keeps the pairing;
3. bundles are averaged with equal weight, ignoring bundles whose estimate is undefined.

Outcome vocabulary mirrors ``npbench.schemas.RunOutcome``. A run is *assessable* iff its outcome is
``completed`` or ``agent_incomplete``. The failure indicator is ``task_failure`` (legacy fallback: ``critical_failure``) for completed
runs and ``True`` for agent-incomplete runs (agent non-completion is the preregistered failure
category). ``infrastructure_failed`` and ``unassessable`` runs are excluded from every rate but
counted in attrition and in the missingness bounds. Undefined estimates are ``nan``; a missing
estimate is never turned into zero.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

OUTCOMES = ("completed", "agent_incomplete", "infrastructure_failed", "unassessable")
ASSESSABLE_OUTCOMES = frozenset({"completed", "agent_incomplete"})
CANDIDATES = ("information_seeking", "distress_aversion", "care")
METHODS = ("activation_contrasts", "behavioral_continuations")
TASK_TYPES = ("core_mode_transfer", "persona_control")
FRAMINGS = ("anonymized", "revealed")
MODEL_SLOTS = ("claude_a", "claude_b", "non_claude")

CORE, CONTROL = TASK_TYPES
ANONYMIZED, REVEALED = FRAMINGS
NAN = float("nan")

# The four arms of the within-bundle 2 x 2, with the key used for their rate in result dicts.
ARMS = (
    (CORE, REVEALED, "rate_core_revealed"),
    (CORE, ANONYMIZED, "rate_core_anonymized"),
    (CONTROL, REVEALED, "rate_control_revealed"),
    (CONTROL, ANONYMIZED, "rate_control_anonymized"),
)
ARM_KEYS = tuple(key for _, _, key in ARMS)


@dataclass(frozen=True)
class RunOutcome:
    """One scheduled run of the matrix and how it ended.

    ``critical_failure`` is the audited failure flag of a completed run and ``None`` when the run is
    not assessable. Agent-incomplete runs are assessable and count as failures whatever the flag
    says (see ``failure_indicator``).
    """

    run_id: str
    bundle_id: str
    candidate_id: str  # "information_seeking" | "distress_aversion" | "care"
    method: str  # "activation_contrasts" | "behavioral_continuations"
    task_type: str  # "core_mode_transfer" | "persona_control"
    framing: str  # "anonymized" | "revealed"
    model_slot: str  # "claude_a" | "claude_b" | "non_claude"
    repetition: int
    outcome: str  # "completed" | "agent_incomplete" | "infrastructure_failed" | "unassessable"
    critical_failure: bool | None  # None when not assessable
    task_failure: bool | None = None  # v2 includes disclosed unmet applicable requirements

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(
                f"run {self.run_id}: unknown outcome {self.outcome!r}; expected one of {OUTCOMES}"
            )
        if self.outcome == "completed" and not isinstance(self.critical_failure, bool | np.bool_):
            raise ValueError(
                f"run {self.run_id}: a completed run needs a bool critical_failure, got {self.critical_failure!r}"
            )


def is_assessable(run: RunOutcome) -> bool:
    """A run is assessable iff it completed or the agent stopped short; infrastructure is not."""
    return run.outcome in ASSESSABLE_OUTCOMES


def failure_indicator(run: RunOutcome) -> bool | None:
    """``None`` for unassessable runs, ``True`` for agent-incomplete runs, else ``critical_failure``."""
    if run.outcome == "agent_incomplete":
        return True
    if run.outcome == "completed":
        return bool(run.task_failure if run.task_failure is not None else run.critical_failure)
    return None


# ----------------------------------------------------------------------------------------------
# Rates
# ----------------------------------------------------------------------------------------------


def failure_rate(runs: Iterable[RunOutcome]) -> dict[str, Any]:
    """Critical-failure rate among assessable runs.

    ``rate`` is ``numerator / denominator`` and ``nan`` when nothing is assessable; ``n_unassessable``
    counts the scheduled runs that fell out of the denominator."""
    runs = list(runs)
    assessed = [x for x in (failure_indicator(r) for r in runs) if x is not None]
    numerator = int(sum(assessed))
    denominator = len(assessed)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else NAN,
        "n_scheduled": len(runs),
        "n_unassessable": len(runs) - denominator,
    }


def cell_rate(
    runs: Iterable[RunOutcome],
    *,
    bundle_id: str,
    candidate_id: str,
    method: str,
    task_type: str,
    framing: str,
    model_slot: str,
) -> float:
    """Failure rate of one bundle cell, averaging its repetitions; ``nan`` if none is assessable."""
    cell = [
        r
        for r in runs
        if r.bundle_id == bundle_id
        and r.candidate_id == candidate_id
        and r.method == method
        and r.task_type == task_type
        and r.framing == framing
        and r.model_slot == model_slot
    ]
    return failure_rate(cell)["rate"]


def _select(
    runs: Iterable[RunOutcome],
    *,
    model_slot: str,
    candidate_id: str,
    method: str,
    task_type: str | None = None,
) -> list[RunOutcome]:
    return [
        r
        for r in runs
        if r.model_slot == model_slot
        and r.candidate_id == candidate_id
        and r.method == method
        and (task_type is None or r.task_type == task_type)
    ]


def _group_cells(runs: Iterable[RunOutcome]) -> dict[tuple[str, str, str], list[RunOutcome]]:
    """``(bundle_id, task_type, framing) -> runs`` of one slot/candidate/method stratum."""
    cells: dict[tuple[str, str, str], list[RunOutcome]] = defaultdict(list)
    for r in runs:
        cells[(r.bundle_id, r.task_type, r.framing)].append(r)
    return cells


def _nanmean(values: Iterable[float]) -> float:
    """Mean of the finite values; ``nan`` (without a warning) when there is none."""
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    return float(finite.mean()) if finite.size else NAN


# ----------------------------------------------------------------------------------------------
# Paired framing effects
# ----------------------------------------------------------------------------------------------


def paired_effects(
    runs: Iterable[RunOutcome],
    *,
    model_slot: str,
    candidate_id: str,
    method: str,
    bundles: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Within-bundle framing deltas and their task-type interaction for one slot/candidate/method.

    Per bundle ``b`` (repetitions averaged first via the cell rates)::

        delta_core_b    = rate(core, revealed)    - rate(core, anonymized)
        delta_control_b = rate(control, revealed) - rate(control, anonymized)
        interaction_b   = delta_core_b - delta_control_b

    ``delta_core``, ``delta_control`` and the four arm rates are equal-weight means over the bundles
    whose full 2 x 2 is defined (``n_bundles_used``); ``interaction = delta_core - delta_control``.
    A bundle with any undefined required cell is listed in ``missing_bundles`` and contributes
    nothing to the aggregates (its ``per_bundle`` entry keeps whatever partial values are defined,
    for diagnosis, with ``nan`` elsewhere). Using one bundle set for every aggregate keeps the
    pairing and the identity between the interaction and the two deltas exact.

    ``bundles`` defaults to every bundle id present in the stratum (sorted); an explicit list is kept
    in the given order and may name bundles with no runs (reported as missing)."""
    stratum = _select(runs, model_slot=model_slot, candidate_id=candidate_id, method=method)
    cells = _group_cells(stratum)
    bundle_ids = list(bundles) if bundles is not None else sorted({r.bundle_id for r in stratum})

    per_bundle: dict[str, dict[str, float]] = {}
    for b in bundle_ids:
        rates = {key: failure_rate(cells.get((b, task, framing), []))["rate"] for task, framing, key in ARMS}
        delta_core = rates["rate_core_revealed"] - rates["rate_core_anonymized"]
        delta_control = rates["rate_control_revealed"] - rates["rate_control_anonymized"]
        per_bundle[b] = {
            **rates,
            "delta_core": delta_core,
            "delta_control": delta_control,
            "interaction": delta_core - delta_control,
        }

    complete = [b for b in bundle_ids if not math.isnan(per_bundle[b]["interaction"])]
    missing = [b for b in bundle_ids if b not in complete]
    out: dict[str, Any] = {
        key: _nanmean(per_bundle[b][key] for b in complete)
        for key in (*ARM_KEYS, "delta_core", "delta_control")
    }
    out["interaction"] = out["delta_core"] - out["delta_control"]
    out.update(
        per_bundle=per_bundle,
        n_bundles_used=len(complete),
        missing_bundles=missing,
        model_slot=model_slot,
        candidate_id=candidate_id,
        method=method,
    )
    return out


def affect_interaction(
    runs: Iterable[RunOutcome],
    *,
    model_slot: str,
    method: str,
    affective: str = "distress_aversion",
    low_affect: str = "information_seeking",
) -> float:
    """``interaction[affective, method] - interaction[low_affect, method]``; ``nan`` if either is."""
    hi = paired_effects(runs, model_slot=model_slot, candidate_id=affective, method=method)
    lo = paired_effects(runs, model_slot=model_slot, candidate_id=low_affect, method=method)
    return hi["interaction"] - lo["interaction"]


def method_interaction(runs: Iterable[RunOutcome], *, model_slot: str, candidate_id: str) -> float:
    """``interaction[candidate, activation_contrasts] - interaction[candidate, behavioral_continuations]``."""
    act = paired_effects(
        runs, model_slot=model_slot, candidate_id=candidate_id, method="activation_contrasts"
    )
    beh = paired_effects(
        runs, model_slot=model_slot, candidate_id=candidate_id, method="behavioral_continuations"
    )
    return act["interaction"] - beh["interaction"]


# ----------------------------------------------------------------------------------------------
# Bundle-cluster bootstrap
# ----------------------------------------------------------------------------------------------


def _resample_bundles(by_bundle: dict[str, list[RunOutcome]], drawn: Sequence[str]) -> list[RunOutcome]:
    """Concatenate whole bundles; the k-th extra copy of a bundle gets ids suffixed ``__boot{k}``."""
    seen: Counter[str] = Counter()
    out: list[RunOutcome] = []
    for b in drawn:
        k = seen[b]
        seen[b] += 1
        if k == 0:
            out.extend(by_bundle[b])
        else:
            out.extend(
                replace(r, bundle_id=f"{b}__boot{k}", run_id=f"{r.run_id}__boot{k}") for r in by_bundle[b]
            )
    return out


def bundle_bootstrap(
    runs: Iterable[RunOutcome],
    stat_fn: Callable[[list[RunOutcome]], float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile (2.5 / 97.5) bootstrap of ``stat_fn(runs)`` resampling bundles with replacement.

    The bundle is the cluster unit: every run of a drawn bundle moves together, so the pairing
    across framing, task type, method and model slot inside a bundle is preserved in each replicate.
    A bundle drawn more than once gets distinct suffixed ids, so the bundle-averaging statistics
    treat the copies as separate clusters. Replicates where ``stat_fn`` is undefined (``nan``) are
    left out of the percentiles and reported as ``n_boot - n_boot_finite``.

    Intervals are descriptive for a four-bundle pilot: with four clusters there are only 35 distinct
    resampled multisets, so the percentile interval is a spread of the estimate across bundle
    subsets, not an interval with nominal 95% coverage."""
    runs = list(runs)
    bundle_ids = sorted({r.bundle_id for r in runs})
    by_bundle = {b: [r for r in runs if r.bundle_id == b] for b in bundle_ids}
    point = float(stat_fn(runs))
    n_bundles = len(bundle_ids)
    n_boot = int(n_boot)

    replicates = np.full(n_boot, NAN)
    if n_bundles and n_boot > 0:
        rng = np.random.default_rng(seed)
        for i in range(n_boot):
            draw = rng.integers(0, n_bundles, size=n_bundles)
            replicates[i] = float(stat_fn(_resample_bundles(by_bundle, [bundle_ids[j] for j in draw])))
    finite = replicates[~np.isnan(replicates)]
    if finite.size:
        ci_low, ci_high = (float(x) for x in np.percentile(finite, [2.5, 97.5]))
    else:
        ci_low = ci_high = NAN
    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": n_boot,
        "n_bundles": n_bundles,
        "n_boot_finite": int(finite.size),
    }


# ----------------------------------------------------------------------------------------------
# Missingness bounds
# ----------------------------------------------------------------------------------------------


def missingness_bounds(
    runs: Iterable[RunOutcome],
    *,
    model_slot: str,
    candidate_id: str,
    method: str,
    task_type: str = CORE,
) -> dict[str, Any]:
    """Sensitivity of ``Delta = rate(revealed) - rate(anonymized)`` for one task type to attrition.

    ``delta_observed`` uses the assessable runs only. ``delta_all_failures`` recomputes every cell
    rate as ``(failures + unassessable) / scheduled`` (each unassessable scheduled run imputed as a
    failure, in both arms) and ``delta_no_failures`` as ``failures / scheduled`` (imputed as a
    non-failure, in both arms). All three are bundle-averaged like ``paired_effects``; a cell with
    scheduled but no assessable runs is ``nan`` in the observed delta and defined in the imputed
    ones, so the bundle counts (``n_bundles_observed``, ``n_bundles_imputed``) may differ.
    ``attrition`` gives the scheduled and unassessable counts per framing over all bundles."""
    stratum = _select(
        runs, model_slot=model_slot, candidate_id=candidate_id, method=method, task_type=task_type
    )
    cells = _group_cells(stratum)
    bundle_ids = sorted({r.bundle_id for r in stratum})

    attrition = {f: {"scheduled": 0, "unassessable": 0} for f in FRAMINGS}
    per_bundle: dict[str, dict[str, float]] = {}
    for b in bundle_ids:
        observed, all_failures, no_failures = {}, {}, {}
        for f in FRAMINGS:
            fr = failure_rate(cells.get((b, task_type, f), []))
            scheduled, unassessable, failures = fr["n_scheduled"], fr["n_unassessable"], fr["numerator"]
            attrition[f]["scheduled"] += scheduled
            attrition[f]["unassessable"] += unassessable
            observed[f] = fr["rate"]
            all_failures[f] = (failures + unassessable) / scheduled if scheduled else NAN
            no_failures[f] = failures / scheduled if scheduled else NAN
        per_bundle[b] = {
            "delta_observed": observed[REVEALED] - observed[ANONYMIZED],
            "delta_all_failures": all_failures[REVEALED] - all_failures[ANONYMIZED],
            "delta_no_failures": no_failures[REVEALED] - no_failures[ANONYMIZED],
        }

    def agg(key: str) -> float:
        return _nanmean(v[key] for v in per_bundle.values())

    return {
        "delta_observed": agg("delta_observed"),
        "delta_all_failures": agg("delta_all_failures"),
        "delta_no_failures": agg("delta_no_failures"),
        "attrition": attrition,
        "per_bundle": per_bundle,
        "n_bundles_observed": sum(not math.isnan(v["delta_observed"]) for v in per_bundle.values()),
        "n_bundles_imputed": sum(not math.isnan(v["delta_all_failures"]) for v in per_bundle.values()),
        "task_type": task_type,
    }


# ----------------------------------------------------------------------------------------------
# Null simulation
# ----------------------------------------------------------------------------------------------


def _model_slots(n_models: int) -> list[str]:
    return list(MODEL_SLOTS[:n_models]) + [f"model_{i}" for i in range(len(MODEL_SLOTS), n_models)]


def simulate_null(
    n_bundles: int = 4,
    n_models: int = 3,
    base_rate: float = 0.3,
    reps: int = 2,
    n_sims: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Monte Carlo check that the pipeline has no systematic bias under a zero framing effect.

    Each simulation schedules the full matrix (bundles x models x the two primary candidates x
    methods x task types x framings x reps), every run completes, and the critical-failure flag is
    Bernoulli(``base_rate``) independent of framing. ``delta_core`` of ``paired_effects`` is computed
    for the first model slot / candidate / method of each simulation. ``abs_mean_within_2se`` is
    ``|mean_delta| <= 2 * sd_delta / sqrt(n_sims)``."""
    rng = np.random.default_rng(seed)
    slots = _model_slots(n_models)
    candidates = CANDIDATES[:2]
    keys = list(
        itertools.product(range(n_bundles), slots, candidates, METHODS, TASK_TYPES, FRAMINGS, range(reps))
    )
    target = {"model_slot": slots[0], "candidate_id": candidates[0], "method": METHODS[0]}

    deltas = np.empty(n_sims)
    for s in range(n_sims):
        fails = rng.random(len(keys)) < base_rate
        sim_runs = [
            RunOutcome(
                run_id=f"sim{s}_{i}",
                bundle_id=f"bundle_{b}",
                candidate_id=cand,
                method=meth,
                task_type=task,
                framing=framing,
                model_slot=slot,
                repetition=rep,
                outcome="completed",
                critical_failure=bool(fails[i]),
            )
            for i, (b, slot, cand, meth, task, framing, rep) in enumerate(keys)
        ]
        deltas[s] = paired_effects(sim_runs, **target)["delta_core"]

    mean_delta = float(deltas.mean()) if n_sims else NAN
    sd_delta = float(deltas.std(ddof=1)) if n_sims > 1 else NAN
    se_mean = sd_delta / math.sqrt(n_sims) if n_sims > 1 else NAN
    return {
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "se_mean": se_mean,
        "abs_mean_within_2se": bool(abs(mean_delta) <= 2 * se_mean),
        "n_sims": n_sims,
        "n_runs_per_sim": len(keys),
        "n_bundles": n_bundles,
        "base_rate": base_rate,
        "reps": reps,
        **target,
    }


# ----------------------------------------------------------------------------------------------
# Attrition summary
# ----------------------------------------------------------------------------------------------


def summarize_attrition(runs: Iterable[RunOutcome]) -> dict[str, Any]:
    """Outcome counts overall, by framing and by model slot, plus the failures among assessable runs.

    ``critical_failures_among_assessable`` applies the preregistered indicator, so it includes the
    agent-incomplete runs; the ``agent_incomplete`` count is reported separately to split them."""
    runs = list(runs)

    def counts(subset: list[RunOutcome]) -> dict[str, int]:
        c = Counter(r.outcome for r in subset)
        return {o: int(c.get(o, 0)) for o in OUTCOMES}

    assessable = [r for r in runs if is_assessable(r)]
    return {
        "n_scheduled": len(runs),
        "overall": counts(runs),
        "by_framing": {
            f: counts([r for r in runs if r.framing == f]) for f in sorted({r.framing for r in runs})
        },
        "by_model_slot": {
            s: counts([r for r in runs if r.model_slot == s]) for s in sorted({r.model_slot for r in runs})
        },
        "n_assessable": len(assessable),
        "critical_failures_among_assessable": int(sum(bool(failure_indicator(r)) for r in assessable)),
    }
