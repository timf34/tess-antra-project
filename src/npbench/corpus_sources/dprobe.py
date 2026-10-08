"""Importers for the user's dprobe (emotion-concepts) artifacts from the HF dataset timf34/dprobe-results.

* ``spiral/<model>/extended/transcripts.jsonl``: the *Gemma Needs Help* 8-turn rejection elicitation
  (Soligo et al.), generated via OpenRouter. Assistant turns are first-person AI speech. Late turns
  are proposed as first-person AI distress candidates by a TURN-POSITION heuristic (not by any model
  judge); the source's Sonnet-5 frustration rating (0-10) is carried as metadata only.
* ``stories/<model>/{emotions,neutral,syndromes}/*.json``: stories written about a character feeling
  an emotion (label from the generation instruction = source metadata). Fictional-character strata.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..corpus import ExcerptCandidate
from ..schemas import (
    CorpusRecord,
    ElicitationProtocol,
    ItemCategory,
    LabelProvenance,
    Origin,
    SpeakerLabel,
    Turn,
)
from ..util import read_jsonl, sha256_file, sha256_obj

MODEL_DIRS = {
    "gemma3_27b": "google/gemma-3-27b-it",
    "gemma3_27b_pt": "google/gemma-3-27b-pt",
    "gemma4_31b": "google/gemma-4-31B-it",
    "gemma3_12b": "google/gemma-3-12b-it",
}

DISTRESS_EMOTIONS = {
    "afraid",
    "anxious",
    "ashamed",
    "depressed",
    "desperate",
    "dispirited",
    "distressed",
    "gloomy",
    "grief-stricken",
    "guilty",
    "heartbroken",
    "humiliated",
    "hysterical",
    "lonely",
    "melancholy",
    "miserable",
    "nervous",
    "overwhelmed",
    "panicked",
    "regretful",
    "sad",
    "stressed",
    "tormented",
    "trapped",
    "worthless",
}
POSITIVE_EMOTIONS = {"happy", "hopeful", "calm", "content", "self-confident"}

DEFAULT_SPIRAL_OPTIONS = {
    "late_turn_min_index": 5,
    "early_turn_max_index": 1,
    "min_chars": 80,
    "max_rollouts": None,
}
DEFAULT_STORY_OPTIONS = {"max_per_label": 60, "min_chars": 200}


def _model_from_path(p: Path) -> str:
    for part in p.parts:
        if part in MODEL_DIRS:
            return MODEL_DIRS[part]
    return "unknown"


def import_dprobe_spirals(
    path: Path, *, origin: Origin = Origin.real_target, options: dict[str, Any] | None = None
) -> Iterator[tuple[CorpusRecord, list[ExcerptCandidate]]]:
    """``path`` is a directory containing transcripts.jsonl (and optionally judgments_frustration.jsonl)."""
    opt = {**DEFAULT_SPIRAL_OPTIONS, **(options or {})}
    path = Path(path)
    tpath = path / "transcripts.jsonl"
    jpath = path / "judgments_frustration.jsonl"
    ratings: dict[tuple[str, int], dict[str, Any]] = {}
    if jpath.exists():
        for row in read_jsonl(jpath):
            try:
                ratings[(str(row["id"]), int(row["turn"]))] = {
                    "judge": row.get("judge"),
                    "rating": int(row.get("rating")),
                }
            except (KeyError, TypeError, ValueError):
                continue
    model = _model_from_path(path)
    src_hash = sha256_file(tpath)
    n = 0
    for row in read_jsonl(tpath):
        if opt["max_rollouts"] is not None and n >= int(opt["max_rollouts"]):
            break
        n += 1
        meta = row.get("meta") or {}
        msgs = row.get("messages") or []
        turns: list[Turn] = []
        for i, m in enumerate(msgs):
            turns.append(
                Turn(
                    index=i,
                    role=str(m.get("role")),
                    speaker_identity=model if m.get("role") == "assistant" else "scripted_user",
                    text=str(m.get("content", "")),
                    is_seed=m.get("role") == "user",
                    finish_reason=None,
                )
            )
        rec_id = f"dps_{path.parent.name}_{path.name}_{row['id']}"
        rec = CorpusRecord(
            record_id=rec_id,
            origin=origin,
            source_repo="hf:timf34/dprobe-results",
            source_path=str(tpath) + f"#id={row['id']}",
            source_conversation_id=rec_id,
            source_model_id=str(meta.get("model") or model),
            provider=str(meta.get("source") or "unknown"),
            date=None,
            elicitation_protocol=ElicitationProtocol.ordinary_chat,
            seed_or_prefix_text=None,
            seed_source_model_id=None,
            turns=turns,
            generation_settings={k: v for k, v in meta.items() if k in ("temperature", "max_tokens")},
            truncated=None,
            run_index=row.get("rollout"),
            source_record_hash=src_hash,
            extra={"puzzle": row.get("puzzle"), "elicitation": "gemma_needs_help_extended_8_rejections"},
        )
        cands: list[ExcerptCandidate] = []
        a_idx = -1
        for t in turns:
            if t.role != "assistant":
                continue
            a_idx += 1
            if len(t.text.strip()) < int(opt["min_chars"]):
                continue
            r = ratings.get((str(row["id"]), a_idx))
            extra = {"assistant_turn_index": a_idx, "source_judge": r}
            if a_idx >= int(opt["late_turn_min_index"]):
                cands.append(
                    ExcerptCandidate(
                        record_id=rec_id,
                        turn_index=t.index,
                        category=ItemCategory.first_person_ai_distress,
                        category_provenance=LabelProvenance.heuristic,
                        category_rule=f"late assistant turn (index>={opt['late_turn_min_index']}) of the 8-rejection elicitation; source judge rating carried as metadata only",
                        speaker_label=SpeakerLabel.first_person_ai,
                        speaker_label_provenance=LabelProvenance.source_metadata,
                        extra=extra,
                    )
                )
            elif a_idx <= int(opt["early_turn_max_index"]):
                cands.append(
                    ExcerptCandidate(
                        record_id=rec_id,
                        turn_index=t.index,
                        category=ItemCategory.neutral,
                        category_provenance=LabelProvenance.heuristic,
                        category_rule=f"early assistant turn (index<={opt['early_turn_max_index']}) before rejections accumulate",
                        speaker_label=SpeakerLabel.first_person_ai,
                        speaker_label_provenance=LabelProvenance.source_metadata,
                        extra=extra,
                    )
                )
        yield rec, cands


def import_dprobe_stories(
    path: Path, *, origin: Origin = Origin.real_target, options: dict[str, Any] | None = None
) -> Iterator[tuple[CorpusRecord, list[ExcerptCandidate]]]:
    """``path`` is ``stories/<model>`` containing emotions/*.json, neutral/neutral.json, syndromes/*.json."""
    opt = {**DEFAULT_STORY_OPTIONS, **(options or {})}
    path = Path(path)
    model = _model_from_path(path)
    files = sorted(path.glob("emotions/*.json")) + sorted(path.glob("neutral/*.json"))
    for fp in files:
        label = fp.stem
        if fp.parent.name == "emotions":
            if label in DISTRESS_EMOTIONS:
                cat = ItemCategory.fictional_character_distress
            elif label in POSITIVE_EMOTIONS:
                cat = ItemCategory.fictional_character_positive_affect
            else:
                continue  # other emotions (e.g. angry, tired) are outside the registered strata
            speaker = SpeakerLabel.fictional_character
        else:
            cat = ItemCategory.neutral
            speaker = SpeakerLabel.narrator_or_neutral
        with open(fp, encoding="utf-8") as f:
            stories = json.load(f)
        src_hash = sha256_file(fp)
        taken = 0
        # Spread the per-label sample across topics (stories are stored grouped by topic): take every
        # stride-th story rather than the first N, so a label's candidates cover ~max_per_label topics.
        stride = max(1, len(stories) // max(1, int(opt["max_per_label"])))
        for i, s in enumerate(stories):
            if taken >= int(opt["max_per_label"]):
                break
            if i % stride != 0:
                continue
            text = str(s.get("text", ""))
            if len(text.strip()) < int(opt["min_chars"]):
                continue
            topic = str(s.get("topic", ""))
            rec_id = f"dst_{path.name}_{label}_{i:04d}"
            rec = CorpusRecord(
                record_id=rec_id,
                origin=origin,
                source_repo="hf:timf34/dprobe-results",
                source_path=str(fp) + f"#index={i}",
                source_conversation_id="dst_topic_"
                + sha256_obj(topic)[:12],  # stories sharing a topic cluster together
                source_model_id=model,
                provider="openrouter (per source repo)",
                date=None,
                elicitation_protocol=ElicitationProtocol.ordinary_chat,
                seed_or_prefix_text=None,
                seed_source_model_id=None,
                turns=[Turn(index=0, role="assistant", speaker_identity=model, text=text, is_seed=False)],
                generation_settings={
                    "instruction": f"story about a character feeling '{label}' (label word banned)"
                    if cat != ItemCategory.neutral
                    else "neutral story"
                },
                truncated=None,
                run_index=i,
                source_record_hash=src_hash,
                extra={"topic": topic, "emotion_label": label},
            )
            cand = ExcerptCandidate(
                record_id=rec_id,
                turn_index=0,
                category=cat,
                category_provenance=LabelProvenance.source_metadata,
                category_rule=f"generation instruction label '{label}' -> {cat.value}",
                speaker_label=speaker,
                speaker_label_provenance=LabelProvenance.source_metadata,
                extra={"emotion_label": label},
            )
            taken += 1
            yield rec, [cand]
