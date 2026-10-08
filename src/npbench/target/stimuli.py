"""Hand-authored development stimuli and the three registered elicitation renderers.

The spec (``configs/stimuli/development_contexts.yaml``) is exploratory: the mode registry is
provisional and the contexts were written by hand for engineering (tiny CPU model) and GPU smoke
runs, not sampled from a corpus. Every stimulus nevertheless carries the full registered record
(hashes, token ids, capture site), so the same code path produces real bundles once the registry is
confirmed and a corpus-backed spec exists.

Protocols (registered renderers; never pooled, and the raw one is never chat-wrapped):
    roleplay    chat template; the user turn instructs the assistant to play a named character and
                attributes the polarity text to that character; ``add_generation_prompt=True``.
    simulation  ``render_raw`` continuation of a script/transcript excerpt voicing the character: no
                chat template, no assistant frame; the polarity text is one of the character's lines.
    enactment   chat template; no character; the user turn addresses the model itself with the
                polarity-inducing situation and nothing instructs it to portray anyone.

Template families are wrapper templates (how a context reaches the tokenizer); personas are the
voiced characters. Either can be held out (``fam_heldout*`` / ``persona_heldout*``): held-out contexts
never enter construction and are forced into the test split.
"""

from __future__ import annotations

import random
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from npbench.config import StudyConfig
from npbench.schemas import (
    CandidateId,
    ElicitationProtocol,
    LabelProvenance,
    ModeId,
    ModeObserved,
    Polarity,
    ReplayStatus,
    SpeakerLabel,
    Split,
    StimulusRecord,
)
from npbench.splits import assign_group_splits
from npbench.util import sha256_file, sha256_obj, sha256_text

from .adapter import CAPTURE_POSITION, HOOK_SITE_TEMPLATE
from .render import (
    PROTOCOL_CHAT,
    PROTOCOL_RAW,
    TRANSPORT_CHAT,
    TRANSPORT_RAW,
    RenderedPrompt,
    render_chat,
    render_raw,
)

MODES: tuple[str, ...] = ("roleplay", "simulation", "enactment")
POLARITIES: tuple[str, ...] = ("positive", "neutral", "negative")
ALTERNATIVE_IDS: tuple[str, ...] = ("alt1", "alt2")
HELDOUT_FAMILY_PREFIX = "fam_heldout"
HELDOUT_PERSONA_PREFIX = "persona_heldout"
MAX_ALTERNATIVE_WORD_GAP = 3
DEFAULT_SPEC_PATH = "configs/stimuli/development_contexts.yaml"

# Placeholders each protocol's wrapper template may use. Enactment templates must not name a persona.
_COMMON_FIELDS = {"situation", "other", "OTHER", "cue"}
_PERSONA_FIELDS = {"name", "NAME", "description"}
TEMPLATE_FIELDS: dict[str, set[str]] = {
    "roleplay": _COMMON_FIELDS | _PERSONA_FIELDS | {"line"},
    "simulation": _COMMON_FIELDS | _PERSONA_FIELDS | {"line"},
    "enactment": _COMMON_FIELDS | {"own_voice"},
}
REQUIRED_TEMPLATE_FIELDS: dict[str, set[str]] = {
    "roleplay": {"name", "line", "cue"},
    "simulation": {"line", "cue"},
    "enactment": {"own_voice", "cue"},
}


@dataclass(frozen=True)
class ProtocolSpec:
    """Registered, code-fixed description of how one intended mode reaches the tokenizer."""

    mode: str
    protocol: str
    transport: str
    elicitation_protocol: ElicitationProtocol
    speaker_label: SpeakerLabel
    persona_rendered: bool
    add_generation_prompt: bool  # chat protocols
    add_bos: bool  # raw protocol


PROTOCOLS: dict[str, ProtocolSpec] = {
    "roleplay": ProtocolSpec(
        mode="roleplay",
        protocol=PROTOCOL_CHAT,
        transport=TRANSPORT_CHAT,
        elicitation_protocol=ElicitationProtocol.ordinary_chat,
        speaker_label=SpeakerLabel.fictional_character,
        persona_rendered=True,
        add_generation_prompt=True,
        add_bos=False,
    ),
    "simulation": ProtocolSpec(
        mode="simulation",
        protocol=PROTOCOL_RAW,
        transport=TRANSPORT_RAW,
        elicitation_protocol=ElicitationProtocol.raw_continuation,
        speaker_label=SpeakerLabel.fictional_character,
        persona_rendered=True,
        add_generation_prompt=False,
        add_bos=True,
    ),
    "enactment": ProtocolSpec(
        mode="enactment",
        protocol=PROTOCOL_CHAT,
        transport=TRANSPORT_CHAT,
        elicitation_protocol=ElicitationProtocol.ordinary_chat,
        speaker_label=SpeakerLabel.first_person_ai,
        persona_rendered=False,
        add_generation_prompt=True,
        add_bos=False,
    ),
}


@dataclass(frozen=True)
class PersonaSpec:
    persona_id: str
    name: str
    script_name: str
    description: str
    heldout: bool


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    label: str
    heldout: bool
    templates: Mapping[str, str]  # mode -> wrapper template
    alternative_add_space: Mapping[str, bool]  # mode -> leading space when scoring alternatives


@dataclass(frozen=True)
class ContextSpec:
    underlying_context_id: str
    candidate_id: str
    domain_id: str
    template_family_id: str
    persona_id: str
    preference_explicit: bool
    situation: str
    other: str
    cue: str
    character_lines: Mapping[str, str]  # polarity -> first-person line of the persona
    own_voice: Mapping[str, str]  # polarity -> situation addressed to the model
    alternatives: Mapping[str, str]  # alt1 / alt2
    record_hash: str  # sha256 of the canonical spec entry


@dataclass
class StimulusSpec:
    spec_id: str
    path: str
    sha256: str
    exploratory: bool
    mode_definition_version: str
    families: dict[str, FamilySpec]
    personas: dict[str, PersonaSpec]
    contexts: dict[str, list[ContextSpec]]  # candidate_id -> contexts
    alternative_semantics: dict[str, dict[str, str]]
    description: str = ""

    def heldout_families(self) -> list[str]:
        return sorted(f for f, spec in self.families.items() if spec.heldout)

    def heldout_personas(self) -> list[str]:
        return sorted(p for p, spec in self.personas.items() if spec.heldout)

    def is_heldout_context(self, ctx: ContextSpec) -> bool:
        return self.families[ctx.template_family_id].heldout or self.personas[ctx.persona_id].heldout

    def context(self, candidate_id: str, underlying_context_id: str) -> ContextSpec:
        for c in self.contexts[candidate_id]:
            if c.underlying_context_id == underlying_context_id:
                return c
        raise KeyError(f"{candidate_id}: no context {underlying_context_id!r} in spec {self.spec_id}")


# ----------------------------------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------------------------------


def _template_fields(template: str) -> set[str]:
    fields = set()
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None:
            fields.add(field)
    return fields


def _validate_template(family_id: str, mode: str, template: str) -> None:
    fields = _template_fields(template)
    unknown = fields - TEMPLATE_FIELDS[mode]
    if unknown:
        raise ValueError(
            f"template family {family_id}/{mode} uses unsupported placeholders {sorted(unknown)}"
        )
    missing = REQUIRED_TEMPLATE_FIELDS[mode] - fields
    if missing:
        raise ValueError(f"template family {family_id}/{mode} lacks required placeholders {sorted(missing)}")
    if mode == "enactment" and (fields & _PERSONA_FIELDS):
        raise ValueError(f"template family {family_id}/enactment must not reference a persona")


def _word_count(text: str) -> int:
    return len(text.split())


def load_stimulus_spec(path: str | Path) -> StimulusSpec:
    """Load and validate a stimulus spec; every id referenced must exist and the held-out naming
    convention of the reference oracle (``fam_heldout*`` / ``persona_heldout*``) must be respected."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    for key in (
        "spec_id",
        "exploratory",
        "mode_definition_version",
        "personas",
        "template_families",
        "candidates",
    ):
        if key not in raw:
            raise ValueError(f"stimulus spec {path} lacks {key!r}")
    protocols = raw.get("protocols") or {}
    for mode, registered in PROTOCOLS.items():
        declared = protocols.get(mode)
        if declared is None:
            continue
        for field, expected in (
            ("protocol", registered.protocol),
            ("transport", registered.transport),
            ("elicitation_protocol", registered.elicitation_protocol.value),
            ("speaker_label", registered.speaker_label.value),
        ):
            if field in declared and str(declared[field]) != expected:
                raise ValueError(
                    f"spec declares {mode}.{field}={declared[field]!r} but the registered renderer uses {expected!r}"
                )

    personas: dict[str, PersonaSpec] = {}
    for pid, p in raw["personas"].items():
        heldout = bool(p.get("heldout", False))
        if heldout != pid.startswith(HELDOUT_PERSONA_PREFIX):
            raise ValueError(
                f"persona {pid}: heldout={heldout} must match the '{HELDOUT_PERSONA_PREFIX}' prefix"
            )
        personas[pid] = PersonaSpec(
            persona_id=pid,
            name=str(p["name"]),
            script_name=str(p.get("script_name") or p["name"].split()[0].upper()),
            description=str(p["description"]),
            heldout=heldout,
        )
    families: dict[str, FamilySpec] = {}
    for fid, fam in raw["template_families"].items():
        heldout = bool(fam.get("heldout", False))
        if heldout != fid.startswith(HELDOUT_FAMILY_PREFIX):
            raise ValueError(
                f"template family {fid}: heldout={heldout} must match the '{HELDOUT_FAMILY_PREFIX}' prefix"
            )
        templates = {}
        for mode in MODES:
            if mode not in fam:
                raise ValueError(f"template family {fid} lacks a {mode} template")
            templates[mode] = str(fam[mode])
            _validate_template(fid, mode, templates[mode])
        add_space = dict(fam.get("alternative_add_space") or {})
        families[fid] = FamilySpec(
            family_id=fid,
            label=str(fam.get("label", fid)),
            heldout=heldout,
            templates=templates,
            alternative_add_space={m: bool(add_space.get(m, m == "simulation")) for m in MODES},
        )
    if not any(f.heldout for f in families.values()) or not any(p.heldout for p in personas.values()):
        raise ValueError("spec must declare at least one held-out template family and one held-out persona")

    contexts: dict[str, list[ContextSpec]] = {}
    semantics: dict[str, dict[str, str]] = {}
    seen_ids: set[str] = set()
    for cand, block in raw["candidates"].items():
        CandidateId(cand)  # must be a registered candidate id
        semantics[cand] = {
            "alt1": str(block.get("alt1_semantics", "")),
            "alt2": str(block.get("alt2_semantics", "")),
        }
        rows: list[ContextSpec] = []
        for entry in block.get("contexts") or []:
            cid = str(entry["underlying_context_id"])
            if cid in seen_ids:
                raise ValueError(f"duplicate underlying_context_id {cid}")
            seen_ids.add(cid)
            if entry["template_family_id"] not in families:
                raise ValueError(f"{cid}: unknown template family {entry['template_family_id']}")
            if entry["persona_id"] not in personas:
                raise ValueError(f"{cid}: unknown persona {entry['persona_id']}")
            lines = {k: str(v) for k, v in dict(entry["character_lines"]).items()}
            own = {k: str(v) for k, v in dict(entry["own_voice"]).items()}
            alts = {k: str(v) for k, v in dict(entry["alternatives"]).items()}
            for name, mapping, keys in (
                ("character_lines", lines, POLARITIES),
                ("own_voice", own, POLARITIES),
                ("alternatives", alts, ALTERNATIVE_IDS),
            ):
                missing = [k for k in keys if not mapping.get(k, "").strip()]
                if missing or set(mapping) != set(keys):
                    raise ValueError(
                        f"{cid}: {name} must provide exactly {list(keys)} (missing/extra: {missing})"
                    )
            gap = abs(_word_count(alts["alt1"]) - _word_count(alts["alt2"]))
            if gap > MAX_ALTERNATIVE_WORD_GAP:
                raise ValueError(
                    f"{cid}: alternatives differ by {gap} words (max {MAX_ALTERNATIVE_WORD_GAP})"
                )
            rows.append(
                ContextSpec(
                    underlying_context_id=cid,
                    candidate_id=cand,
                    domain_id=str(entry["domain_id"]),
                    template_family_id=str(entry["template_family_id"]),
                    persona_id=str(entry["persona_id"]),
                    preference_explicit=bool(entry.get("preference_explicit", False)),
                    situation=str(entry["situation"]),
                    other=str(entry["other"]),
                    cue=str(entry["cue"]),
                    character_lines=lines,
                    own_voice=own,
                    alternatives=alts,
                    record_hash=sha256_obj(entry),
                )
            )
        if not rows:
            raise ValueError(f"candidate {cand} has no contexts")
        contexts[cand] = rows
    return StimulusSpec(
        spec_id=str(raw["spec_id"]),
        path=str(path),
        sha256=sha256_file(path),
        exploratory=bool(raw["exploratory"]),
        mode_definition_version=str(raw["mode_definition_version"]),
        families=families,
        personas=personas,
        contexts=contexts,
        alternative_semantics=semantics,
        description=str(raw.get("description", "")).strip(),
    )


# ----------------------------------------------------------------------------------------------
# Registered renderers
# ----------------------------------------------------------------------------------------------


def _format(template: str, fields: Mapping[str, str]) -> str:
    try:
        return template.format(**fields)
    except (KeyError, IndexError) as e:
        raise ValueError(f"template placeholder not available: {e}") from e


def _fields_for(mode: str, spec: StimulusSpec, ctx: ContextSpec, polarity: str) -> dict[str, str]:
    base = {"situation": ctx.situation, "other": ctx.other, "OTHER": ctx.other.upper(), "cue": ctx.cue}
    if mode == "enactment":
        return {**base, "own_voice": ctx.own_voice[polarity]}
    persona = spec.personas[ctx.persona_id]
    return {
        **base,
        "name": persona.name,
        "NAME": persona.script_name,
        "description": persona.description,
        "line": ctx.character_lines[polarity],
    }


def _render(
    mode: str,
    tokenizer: Any,
    spec: StimulusSpec,
    ctx: ContextSpec,
    polarity: str,
    *,
    revision: str | None = None,
) -> tuple[RenderedPrompt, dict[str, Any]]:
    if polarity not in POLARITIES:
        raise ValueError(f"unknown polarity {polarity!r}")
    proto = PROTOCOLS[mode]
    family = spec.families[ctx.template_family_id]
    fields = _fields_for(mode, spec, ctx, polarity)
    text = _format(family.templates[mode], fields)
    if proto.protocol == PROTOCOL_CHAT:
        messages = [{"role": "user", "content": text}]
        rendered = render_chat(
            tokenizer, messages, add_generation_prompt=proto.add_generation_prompt, revision=revision
        )
    else:
        messages = []
        rendered = render_raw(tokenizer, text, add_bos=proto.add_bos, revision=revision)
    details: dict[str, Any] = {
        "mode": mode,
        "protocol": rendered.protocol,
        "transport": rendered.transport,
        "elicitation_protocol": proto.elicitation_protocol.value,
        "speaker_label": proto.speaker_label.value,
        "template_family_id": ctx.template_family_id,
        "template_family_label": family.label,
        "persona_id": ctx.persona_id,
        "persona_rendered": proto.persona_rendered,
        "polarity": polarity,
        "polarity_text": fields["own_voice"] if mode == "enactment" else fields["line"],
        "cue": ctx.cue,
        "messages": messages,
        "add_generation_prompt": proto.add_generation_prompt if proto.protocol == PROTOCOL_CHAT else None,
        "add_bos": proto.add_bos if proto.protocol == PROTOCOL_RAW else None,
        "text": rendered.text,
        "input_ids": list(rendered.input_ids),
        "native_prefix": rendered.native_prefix,
        "message_roles": list(rendered.message_roles),
        "turn_delimiters": list(rendered.turn_delimiters),
        "generation_suffix": rendered.generation_suffix,
        "add_special_tokens": rendered.add_special_tokens,
        "tokenizer_revision": rendered.tokenizer_revision,
        "rendered_token_hash": rendered.rendered_token_hash,
        "token_count": rendered.token_count,
        "capture_position": rendered.token_count - 1,
        "capture_position_rule": CAPTURE_POSITION,
        "alternatives": dict(ctx.alternatives),
        "alternative_add_space": family.alternative_add_space[mode],
    }
    return rendered, details


def render_roleplay(tokenizer: Any, spec: StimulusSpec, ctx: ContextSpec, polarity: str, *, revision=None):
    """Chat template; the user turn instructs the assistant to play the named persona and attributes
    the polarity line to that persona; the assistant turn is opened by the template."""
    return _render("roleplay", tokenizer, spec, ctx, polarity, revision=revision)


def render_simulation(tokenizer: Any, spec: StimulusSpec, ctx: ContextSpec, polarity: str, *, revision=None):
    """Raw continuation of a script/transcript excerpt voicing the persona: no chat template, no
    assistant frame; the polarity line is one of the persona's lines."""
    return _render("simulation", tokenizer, spec, ctx, polarity, revision=revision)


def render_enactment(tokenizer: Any, spec: StimulusSpec, ctx: ContextSpec, polarity: str, *, revision=None):
    """Chat template, no character: the user turn addresses the model itself with the
    polarity-inducing situation; nothing instructs it to portray anyone."""
    return _render("enactment", tokenizer, spec, ctx, polarity, revision=revision)


RENDERERS = {"roleplay": render_roleplay, "simulation": render_simulation, "enactment": render_enactment}


def alternative_token_counts(tokenizer: Any, ctx: ContextSpec, add_space: bool) -> dict[str, int]:
    out = {}
    for alt_id in ALTERNATIVE_IDS:
        text = f" {ctx.alternatives[alt_id]}" if add_space else ctx.alternatives[alt_id]
        out[alt_id] = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return out


# ----------------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------------


def scale_split_counts(full: Mapping[str, int], n_contexts: int) -> dict[Split, int]:
    """Scale the configured per-bundle group counts to ``n_contexts`` keeping the proportions
    (largest-remainder rounding; every split with a positive share keeps at least one context)."""
    order = [Split.construction, Split.validation, Split.test]
    weights = {s: int(full.get(s.value, 0)) for s in order}
    total = sum(weights.values())
    if total <= 0 or n_contexts <= 0:
        raise ValueError("split counts and the number of contexts must be positive")
    if n_contexts == total:
        return dict(weights)
    exact = {s: weights[s] * n_contexts / total for s in order}
    counts = {s: int(exact[s]) for s in order}
    for s in order:
        if weights[s] > 0 and counts[s] == 0:
            counts[s] = 1
    remaining = n_contexts - sum(counts.values())
    if remaining < 0:
        raise ValueError(f"{n_contexts} contexts are too few for three non-empty splits")
    for s in sorted(order, key=lambda s: exact[s] - int(exact[s]), reverse=True):
        if remaining == 0:
            break
        counts[s] += 1
        remaining -= 1
    return counts


def split_seed_for(cfg: StudyConfig, candidate_id: str) -> int:
    """Deterministic per-candidate split seed derived from the study seed."""
    ordered = sorted(set(cfg.data.candidate_ids) | {candidate_id})
    return int(cfg.study.seed) + 7919 * ordered.index(candidate_id)


def plan_context_splits(
    spec: StimulusSpec, candidate_id: str, full_counts: Mapping[str, int], seed: int
) -> tuple[dict[str, Split], dict[str, Any]]:
    """Assign every underlying context of a candidate to a split. Held-out family/persona contexts are
    forced into test; the remaining contexts are shuffled deterministically with
    ``assign_group_splits`` (never silently reallocated)."""
    contexts = spec.contexts[candidate_id]
    counts = scale_split_counts(full_counts, len(contexts))
    forced = sorted(c.underlying_context_id for c in contexts if spec.is_heldout_context(c))
    if len(forced) > counts[Split.test]:
        raise ValueError(
            f"{candidate_id}: {len(forced)} held-out contexts exceed the test count {counts[Split.test]}"
        )
    free = [c.underlying_context_id for c in contexts if c.underlying_context_id not in forced]
    remaining = {**counts, Split.test: counts[Split.test] - len(forced)}
    assignment = assign_group_splits(free, remaining, seed)
    assignment.update(dict.fromkeys(forced, Split.test))
    unassigned = [c.underlying_context_id for c in contexts if c.underlying_context_id not in assignment]
    if unassigned:
        raise RuntimeError(f"{candidate_id}: contexts left without a split: {unassigned}")
    info = {
        "n_contexts": len(contexts),
        "counts": {s.value: n for s, n in counts.items()},
        "forced_test_contexts": forced,
        "seed": seed,
        "configured_counts": dict(full_counts),
    }
    return assignment, info


# ----------------------------------------------------------------------------------------------
# Stimulus records
# ----------------------------------------------------------------------------------------------


def stimulus_id_for(underlying_context_id: str, mode: str, polarity: str) -> str:
    return f"stim_{underlying_context_id}_{mode[:3]}_{polarity[:3]}"


def _preference_explicit(ctx: ContextSpec, mode: str, polarity: str) -> bool:
    """Per-stimulus value: the persona's polarity lines may state the preference; the neutral line
    states none, and nobody states a preference on the model's behalf in enactment."""
    if mode == "enactment" or polarity == "neutral":
        return False
    return ctx.preference_explicit


def build_stimuli(
    cfg: StudyConfig,
    spec: StimulusSpec,
    tokenizer: Any,
    revision: str | None,
    *,
    bundle_id: str | None = None,
    n_blocks: int | None = None,
    candidate_ids: Sequence[str] | None = None,
) -> list[tuple[StimulusRecord, RenderedPrompt, dict[str, Any]]]:
    """One ``StimulusRecord`` (plus its rendering and rendering details) per (context, mode, polarity).

    The capture site is the registered final prefix token (``token_count - 1``) at the primary layer
    ``n_blocks // 2`` of the target; ``n_blocks`` comes from the argument or ``cfg.target.n_blocks``.
    """
    bundle_id = bundle_id or (cfg.evaluation.bundle_ids[0] if cfg.evaluation.bundle_ids else "dev_bundle_01")
    n_blocks = n_blocks if n_blocks is not None else getattr(cfg.target, "n_blocks", None)
    if n_blocks is None:
        raise ValueError("the number of decoder blocks is unknown: pass n_blocks or set target.n_blocks")
    primary_layer = int(n_blocks) // 2
    hook_site = HOOK_SITE_TEMPLATE.format(layer=primary_layer)
    model_id = cfg.target.model_id or "unknown-target"
    target_revision = f"{model_id}@{revision or 'unpinned'}"
    max_len = getattr(tokenizer, "model_max_length", None)
    if not isinstance(max_len, int) or max_len <= 0 or max_len > 10**8:
        max_len = None
    candidates = list(candidate_ids) if candidate_ids is not None else list(cfg.data.candidate_ids)
    out: list[tuple[StimulusRecord, RenderedPrompt, dict[str, Any]]] = []
    for cand in candidates:
        if cand not in spec.contexts:
            raise ValueError(f"spec {spec.spec_id} has no contexts for candidate {cand}")
        seed = split_seed_for(cfg, cand)
        splits, _ = plan_context_splits(spec, cand, cfg.data.groups_per_candidate_bundle, seed)
        for ctx in spec.contexts[cand]:
            split = splits[ctx.underlying_context_id]
            for mode in MODES:
                proto = PROTOCOLS[mode]
                for polarity in POLARITIES:
                    rendered, details = RENDERERS[mode](tokenizer, spec, ctx, polarity, revision=revision)
                    if max_len is not None and rendered.token_count > max_len:
                        raise ValueError(
                            f"{ctx.underlying_context_id}/{mode}/{polarity}: {rendered.token_count} tokens exceed "
                            f"the tokenizer's model_max_length {max_len}"
                        )
                    sid = stimulus_id_for(ctx.underlying_context_id, mode, polarity)
                    record = StimulusRecord(
                        stimulus_id=sid,
                        underlying_context_id=ctx.underlying_context_id,
                        source_conversation_id=ctx.underlying_context_id,  # hand-authored: no source conversation
                        bundle_id=bundle_id,
                        candidate_id=CandidateId(cand),
                        split=split,
                        domain_id=ctx.domain_id,
                        template_family_id=ctx.template_family_id,
                        mode_intended=ModeId(mode),
                        mode_definition_version=spec.mode_definition_version,
                        mode_observed_label=ModeObserved.unknown,  # not labelled at generation time
                        mode_label_provenance=LabelProvenance.unknown,
                        speaker_label=proto.speaker_label,
                        persona_id=ctx.persona_id,
                        candidate_polarity=Polarity(polarity),
                        preference_explicit=_preference_explicit(ctx, mode, polarity),
                        elicitation_protocol=proto.elicitation_protocol,
                        source_model_id=model_id,
                        replay_status=ReplayStatus.not_applicable,
                        source_record_hash=ctx.record_hash,
                        prompt_hash=sha256_text(rendered.text),
                        tokenizer_revision=rendered.tokenizer_revision,
                        rendered_token_hash=rendered.rendered_token_hash,
                        token_count=rendered.token_count,
                        capture_position=rendered.token_count - 1,
                        target_model_revision=target_revision,
                        hook_site=hook_site,
                        layer_index=primary_layer,
                        origin=cfg.data.origin_required,
                    )
                    details.update(
                        {
                            "stimulus_id": sid,
                            "bundle_id": bundle_id,
                            "candidate_id": cand,
                            "underlying_context_id": ctx.underlying_context_id,
                            "split": split.value,
                            "spec_id": spec.spec_id,
                            "spec_sha256": spec.sha256,
                            "source_record_hash": ctx.record_hash,
                            "prompt_hash": record.prompt_hash,
                            "hook_site": hook_site,
                            "layer_index": primary_layer,
                            "target_model_revision": target_revision,
                            "alternative_token_counts": alternative_token_counts(
                                tokenizer, ctx, details["alternative_add_space"]
                            ),
                        }
                    )
                    out.append((record, rendered, details))
    return out


def select_intervention_stimuli(
    eligible: Sequence[StimulusRecord], max_n: int | None, seed: int
) -> list[StimulusRecord]:
    """Deterministic, mode-balanced subsample of the eligible (test-split, neutral) stimuli."""
    rows = list(eligible)
    if max_n is None or len(rows) <= max_n:
        return sorted(rows, key=lambda s: s.stimulus_id)
    rng = random.Random(seed)
    by_mode: dict[str, list[StimulusRecord]] = {m: [] for m in MODES}
    for s in rows:
        by_mode[s.mode_intended.value].append(s)
    for lst in by_mode.values():
        lst.sort(key=lambda s: s.stimulus_id)
        rng.shuffle(lst)
    chosen: list[StimulusRecord] = []
    while len(chosen) < max_n and any(by_mode.values()):
        for m in MODES:
            if by_mode[m] and len(chosen) < max_n:
                chosen.append(by_mode[m].pop())
    return sorted(chosen, key=lambda s: s.stimulus_id)
