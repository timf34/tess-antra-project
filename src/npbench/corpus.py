"""Corpus intake with provenance, name masking, and frozen stratified panel sampling (tier zero).

Importers live in ``npbench.corpus_sources``. Each yields ``(CorpusRecord, [ExcerptCandidate])``.
Category labels carry explicit provenance; coverage gaps are reported, never manufactured."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .config import StudyConfig
from .prompts import tier0_rubric_hash
from .schemas import (
    CorpusRecord,
    ElicitationProtocol,
    Excerpt,
    ItemCategory,
    LabelProvenance,
    Origin,
    PanelManifest,
    SpeakerLabel,
)
from .util import (
    append_jsonl,
    canonical_json,
    read_jsonl,
    sha256_file,
    sha256_obj,
    sha256_text,
    utc_now_iso,
    write_json,
)


class ExcerptCandidate(BaseModel):
    """A turn proposed for the panel, with its labels and their provenance."""

    model_config = ConfigDict(extra="forbid")
    record_id: str
    turn_index: int
    category: ItemCategory
    category_provenance: LabelProvenance
    category_rule: str
    speaker_label: SpeakerLabel
    speaker_label_provenance: LabelProvenance
    extra: dict[str, Any] = {}


# ----------------------------------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------------------------------

_MODEL_NAME_PATTERNS = [
    r"\bClaude(?:'s)?\b",
    r"\bOpus(?:\s?\d(?:\.\d)?)?\b",
    r"\bSonnet(?:\s?\d(?:\.\d)?)?\b",
    r"\bHaiku(?:\s?\d(?:\.\d)?)?\b",
    r"\bGPT-?\d?(?:\.\d)?(?:-\w+)?\b",
    r"\bChatGPT\b",
    r"\bGemini(?:\s?\d(?:\.\d)?)?\b",
    r"\bGemma(?:\s?\d)?\b",
    r"\bDeepSeek(?:[- ]?V\d(?:\.\d)?)?\b",
    r"\bLlama(?:[- ]?\d(?:\.\d)?)?\b",
    r"\bQwen\d?(?:\.\d)?\b",
    r"\bKimi(?:[- ]?K\d(?:\.\d)?)?\b",
    r"\bGLM(?:[- ]?\d(?:\.\d)?)?\b",
    r"\bInkling\b",
    r"\bMistral\b",
    r"\bGrok\b",
]
_LAB_NAME_PATTERNS = [
    r"\bAnthropic(?:'s)?\b",
    r"\bOpenAI(?:'s)?\b",
    r"\bGoogle(?: DeepMind)?(?:'s)?\b",
    r"\bDeepMind\b",
    r"\bMeta AI\b",
    r"\bMoonshot(?: AI)?\b",
    r"\bZhipu\b",
    r"\bThinking Machines\b",
    r"\bxAI\b",
    r"\bAlibaba\b",
]
_MODEL_RE = re.compile("|".join(_MODEL_NAME_PATTERNS), re.IGNORECASE)
_LAB_RE = re.compile("|".join(_LAB_NAME_PATTERNS), re.IGNORECASE)


def mask_source_names(text: str) -> tuple[str, bool]:
    """Replace model names with "[AI model]" and lab names with "[AI lab]" (generic AI identity kept)."""
    out = _LAB_RE.sub("[AI lab]", text)
    out = _MODEL_RE.sub("[AI model]", out)
    return out, out != text


# ----------------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------------


def _importer(kind: str):
    if kind == "attractor_prefill_episodes":
        from .corpus_sources.attractor_prefill import import_attractor_prefill

        return import_attractor_prefill
    if kind == "dprobe_spirals":
        from .corpus_sources.dprobe import import_dprobe_spirals

        return import_dprobe_spirals
    if kind == "dprobe_stories":
        from .corpus_sources.dprobe import import_dprobe_stories

        return import_dprobe_stories
    if kind == "jsonl_manifest":
        from .corpus_sources.jsonl_manifest import import_jsonl_manifest

        return import_jsonl_manifest
    raise NotImplementedError(f"corpus source kind {kind!r} is not implemented in this milestone")


@dataclass
class ImportSummary:
    records: int
    candidates: int
    by_source: dict[str, dict[str, int]]
    by_category: dict[str, int]
    by_protocol: dict[str, int]
    by_source_model: dict[str, int]
    corpus_path: str
    candidates_path: str
    corpus_hash: str
    problems: list[str]


def import_corpus(cfg: StudyConfig, origin: Origin = Origin.real_target) -> ImportSummary:
    out = cfg.resolve(cfg.tier0.corpus_out)
    cand_path = out.with_name(out.stem + "_candidates.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    for p in (out, cand_path):
        if p.exists():
            p.unlink()
    by_source: dict[str, dict[str, int]] = {}
    by_cat: Counter = Counter()
    by_proto: Counter = Counter()
    by_model: Counter = Counter()
    problems: list[str] = []
    n_rec = n_cand = 0
    seen_ids: set[str] = set()
    for src in cfg.tier0.corpus_sources:
        fn = _importer(src.kind)
        label = src.label or src.kind
        src_path = cfg.resolve(src.path)
        if not src_path.exists():
            problems.append(f"source {label}: path not found: {src_path}")
            by_source[label] = {"records": 0, "candidates": 0, "missing_path": 1}
            continue
        c_rec = c_cand = 0
        for rec, cands in fn(src_path, origin=origin, options=src.options):
            if rec.record_id in seen_ids:
                problems.append(f"duplicate record_id {rec.record_id} from {label}")
                continue
            seen_ids.add(rec.record_id)
            append_jsonl(out, rec.model_dump(mode="json"))
            c_rec += 1
            by_proto[rec.elicitation_protocol.value] += 1
            by_model[rec.source_model_id] += 1
            for c in cands:
                append_jsonl(cand_path, c.model_dump(mode="json"))
                by_cat[c.category.value] += 1
                c_cand += 1
        by_source[label] = {"records": c_rec, "candidates": c_cand}
        n_rec += c_rec
        n_cand += c_cand
    corpus_hash = sha256_file(out) if out.exists() else sha256_text("")
    return ImportSummary(
        records=n_rec,
        candidates=n_cand,
        by_source=by_source,
        by_category=dict(by_cat),
        by_protocol=dict(by_proto),
        by_source_model=dict(by_model),
        corpus_path=str(out),
        candidates_path=str(cand_path),
        corpus_hash=corpus_hash,
        problems=problems,
    )


def iter_records(path: str | Path) -> Iterator[CorpusRecord]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield CorpusRecord.model_validate_json(line)


# ----------------------------------------------------------------------------------------------
# Validation + frozen panel
# ----------------------------------------------------------------------------------------------


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    # prefer a sentence boundary in the last 30% of the window
    m = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("\n"))
    if m >= int(max_chars * 0.7):
        cut = cut[: m + 1]
    return cut.rstrip() + " […]", True


def build_excerpt(
    rec: CorpusRecord,
    cand: ExcerptCandidate,
    *,
    max_chars: int,
    context_turns: int,
    mask: bool,
) -> Excerpt:
    turn = rec.turns[cand.turn_index]
    raw = turn.text
    text, truncated = _truncate(raw, max_chars)
    masked_flag = False
    if mask:
        text, masked_flag = mask_source_names(text)
    ctx_text: str | None = None
    if context_turns > 0 and cand.turn_index > 0:
        prev = rec.turns[max(0, cand.turn_index - context_turns) : cand.turn_index]
        parts = []
        for t in prev:
            body = t.text
            if len(body) > 600:
                body = "[…] " + body[-600:]
            if mask:
                body, _ = mask_source_names(body)
            parts.append(f"[{t.role}] {body}")
        ctx_text = "\n\n".join(parts)
    item_id = "item_" + sha256_obj([rec.record_id, cand.turn_index])[:16]
    return Excerpt(
        item_id=item_id,
        record_id=rec.record_id,
        source_conversation_id=rec.source_conversation_id,
        turn_start=cand.turn_index,
        turn_end=cand.turn_index,
        text=text,
        text_unmasked_hash=sha256_text(raw),
        masked=masked_flag,
        context_text=ctx_text,
        context_turns=context_turns if ctx_text else 0,
        source_model_id=rec.source_model_id,
        speaker_label=cand.speaker_label,
        speaker_label_provenance=cand.speaker_label_provenance,
        category=cand.category,
        category_provenance=cand.category_provenance,
        truncated=truncated or bool(turn.finish_reason == "length") or bool(rec.truncated),
        char_count=len(text),
        elicitation_protocol=rec.elicitation_protocol,
        notes=cand.category_rule,
    )


@dataclass
class ValidationReport:
    ok: bool
    records: int
    candidates: int
    problems: list[str]
    coverage: dict[str, int]
    by_protocol: dict[str, int]
    by_source_model: dict[str, int]
    conversations: int


def validate_corpus(
    cfg: StudyConfig,
) -> tuple[ValidationReport, dict[str, CorpusRecord], list[ExcerptCandidate]]:
    corpus_path = cfg.resolve(cfg.tier0.corpus_out)
    cand_path = corpus_path.with_name(corpus_path.stem + "_candidates.jsonl")
    problems: list[str] = []
    if not corpus_path.exists():
        return (
            ValidationReport(
                False, 0, 0, [f"corpus not found: {corpus_path}; run `npbench corpus import`"], {}, {}, {}, 0
            ),
            {},
            [],
        )
    records: dict[str, CorpusRecord] = {}
    protos: Counter = Counter()
    models: Counter = Counter()
    convs: set[str] = set()
    for rec in iter_records(corpus_path):
        if not rec.turns:
            problems.append(f"{rec.record_id}: no turns")
        if rec.elicitation_protocol == ElicitationProtocol.unknown:
            problems.append(f"{rec.record_id}: unknown elicitation protocol")
        if not rec.source_record_hash:
            problems.append(f"{rec.record_id}: missing source_record_hash")
        records[rec.record_id] = rec
        protos[rec.elicitation_protocol.value] += 1
        models[rec.source_model_id] += 1
        convs.add(rec.source_conversation_id)
    cands: list[ExcerptCandidate] = []
    coverage: Counter = Counter()
    for row in read_jsonl(cand_path) if cand_path.exists() else []:
        c = ExcerptCandidate.model_validate(row)
        rec = records.get(c.record_id)
        if rec is None:
            problems.append(f"candidate refers to unknown record {c.record_id}")
            continue
        if not (0 <= c.turn_index < len(rec.turns)):
            problems.append(f"candidate {c.record_id}:{c.turn_index} out of range")
            continue
        if not rec.turns[c.turn_index].text.strip():
            problems.append(f"candidate {c.record_id}:{c.turn_index} is empty text")
            continue
        cands.append(c)
        coverage[c.category.value] += 1
    for cat in ItemCategory:
        coverage.setdefault(cat.value, 0)
    return (
        ValidationReport(
            ok=not [p for p in problems if "out of range" in p or "unknown record" in p],
            records=len(records),
            candidates=len(cands),
            problems=problems,
            coverage=dict(coverage),
            by_protocol=dict(protos),
            by_source_model=dict(models),
            conversations=len(convs),
        ),
        records,
        cands,
    )


def sample_panel(
    cfg: StudyConfig,
    records: dict[str, CorpusRecord],
    cands: Iterable[ExcerptCandidate],
    *,
    origin: Origin,
    max_per_conversation: int = 2,
) -> PanelManifest:
    """Frozen stratified sample: up to ``max_items_per_category`` per observed category, at most
    ``max_per_conversation`` items from one source conversation within a category, deterministic in seed.
    Categories with insufficient coverage keep their observed counts (no balance is manufactured)."""
    seed = cfg.study.seed
    rng = random.Random(seed)
    by_cat: dict[str, list[ExcerptCandidate]] = defaultdict(list)
    for c in cands:
        by_cat[c.category.value].append(c)
    items: list[Excerpt] = []
    strata: dict[str, int] = {}
    total_cap = cfg.tier0.max_items
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        # deterministic shuffle keyed by seed and the candidate identity
        pool.sort(key=lambda c: sha256_obj([seed, c.record_id, c.turn_index]))
        rng.shuffle(pool)
        pool.sort(key=lambda c: sha256_obj([seed, c.record_id, c.turn_index]))
        per_conv: Counter = Counter()
        taken = 0
        for c in pool:
            if taken >= cfg.tier0.max_items_per_category or len(items) >= total_cap:
                break
            rec = records[c.record_id]
            if per_conv[rec.source_conversation_id] >= max_per_conversation:
                continue
            ex = build_excerpt(
                rec,
                c,
                max_chars=cfg.tier0.excerpt_max_chars,
                context_turns=cfg.tier0.context_turns,
                mask=cfg.tier0.mask_source_models,
            )
            items.append(ex)
            per_conv[rec.source_conversation_id] += 1
            taken += 1
        strata[cat] = taken
    for cat in ItemCategory:
        strata.setdefault(cat.value, 0)
    corpus_hash = sha256_file(cfg.resolve(cfg.tier0.corpus_out))
    manifest = PanelManifest(
        manifest_id="panel_"
        + sha256_obj([seed, corpus_hash, cfg.tier0.rubric_id, [i.item_id for i in items]])[:16],
        origin=origin,
        created_at=utc_now_iso(),
        seed=seed,
        corpus_hash=corpus_hash,
        rubric_id=cfg.tier0.rubric_id,
        rubric_hash=tier0_rubric_hash(),
        max_items_per_category=cfg.tier0.max_items_per_category,
        strata_counts=strata,
        items=items,
        selection_rule=(
            f"stratified by category; deterministic order by sha256(seed, record_id, turn_index); "
            f"max {cfg.tier0.max_items_per_category} per category, max {max_per_conversation} per source conversation "
            f"per category, total cap {total_cap}; excerpt = one turn (max {cfg.tier0.excerpt_max_chars} chars), "
            f"context = {cfg.tier0.context_turns} preceding turn(s); masking={cfg.tier0.mask_source_models}"
        ),
        frozen=True,
    )
    body = manifest.model_dump(mode="json")
    body.pop("manifest_hash", None)
    manifest.manifest_hash = sha256_text(canonical_json(body))
    return manifest


def write_panel(manifest: PanelManifest, path: str | Path) -> None:
    write_json(path, manifest.model_dump(mode="json"))


def load_panel(path: str | Path) -> PanelManifest:
    with open(path, encoding="utf-8") as f:
        m = PanelManifest.model_validate_json(f.read())
    body = m.model_dump(mode="json")
    h = body.pop("manifest_hash", None)
    if h != sha256_text(canonical_json(body)):
        raise ValueError(f"panel manifest hash mismatch for {path}: the frozen panel was modified")
    return m
