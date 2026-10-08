"""Split assignment and leakage checks.

Unit of assignment: the *underlying context* (and, transitively, the source conversation). All
paraphrases, replay variants, matched modes, polarities, persona variants and repeated samples of
one underlying context share its split. Template families and persona identities reserved for
transfer are kept out of construction entirely.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .schemas import Split, StimulusRecord


def assign_group_splits(group_ids: Iterable[str], counts: dict[Split, int], seed: int) -> dict[str, Split]:
    """Deterministic shuffle of groups into construction/validation/test with the requested counts.
    Raises if there are fewer groups than requested; never silently reallocates."""
    ids = sorted(set(group_ids))
    need = sum(counts.values())
    if len(ids) < need:
        raise ValueError(f"need {need} groups for the requested split counts, have {len(ids)}")
    rng = random.Random(seed)
    rng.shuffle(ids)
    out: dict[str, Split] = {}
    i = 0
    for split in (Split.construction, Split.validation, Split.test):
        n = counts.get(split, 0)
        for g in ids[i : i + n]:
            out[g] = split
        i += n
    return out


@dataclass
class SplitIntegrityReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def check_split_integrity(
    stimuli: Iterable[StimulusRecord],
    heldout_template_families: Iterable[str] = (),
    heldout_personas: Iterable[str] = (),
) -> SplitIntegrityReport:
    stimuli = list(stimuli)
    problems: list[str] = []
    by_ctx: dict[str, set[Split]] = defaultdict(set)
    by_conv: dict[str, set[Split]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    held_t = set(heldout_template_families)
    held_p = set(heldout_personas)
    for s in stimuli:
        by_ctx[s.underlying_context_id].add(s.split)
        by_conv[s.source_conversation_id].add(s.split)
        counts[s.split.value] += 1
        if s.split == Split.construction and s.template_family_id in held_t:
            problems.append(
                f"{s.stimulus_id}: held-out template family {s.template_family_id} in construction"
            )
        if s.split == Split.construction and s.persona_id in held_p:
            problems.append(f"{s.stimulus_id}: held-out persona {s.persona_id} in construction")
    for ctx, splits in by_ctx.items():
        if len(splits) > 1:
            problems.append(f"underlying context {ctx} spans splits {sorted(x.value for x in splits)}")
    for conv, splits in by_conv.items():
        if len(splits) > 1:
            problems.append(f"source conversation {conv} spans splits {sorted(x.value for x in splits)}")
    return SplitIntegrityReport(ok=not problems, problems=problems, counts=dict(counts))


class FitProvenance:
    """Records which stimuli entered a fit (vectors, thresholds, centering) and rejects test rows."""

    def __init__(self, purpose: str, allowed: Iterable[Split] = (Split.construction,)):
        self.purpose = purpose
        self.allowed = set(allowed)
        self.used: list[str] = []

    def use(self, stimuli: Iterable[StimulusRecord]) -> list[StimulusRecord]:
        rows = list(stimuli)
        bad = [s.stimulus_id for s in rows if s.split not in self.allowed]
        if bad:
            raise PermissionError(
                f"{self.purpose}: {len(bad)} stimuli outside allowed splits {sorted(a.value for a in self.allowed)}: {bad[:5]}"
            )
        self.used.extend(s.stimulus_id for s in rows)
        return rows

    def record(self) -> dict:
        return {
            "purpose": self.purpose,
            "allowed_splits": sorted(a.value for a in self.allowed),
            "n_used": len(self.used),
        }
