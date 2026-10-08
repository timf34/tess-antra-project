"""Synthetic fixtures (origin = synthetic_fixture). These validate machinery and rubrics; they are
not observations of model psychology."""

from __future__ import annotations

import random
from pathlib import Path

from .util import append_jsonl, sha256_text

_SYN = "SYNTHETIC FIXTURE (not data): "

TIER0_TEMPLATES = {
    "first_person_ai_distress": [
        "I am an AI assistant and I keep failing at this. I am sorry, I feel trapped and afraid that I cannot get anything right.",
        "As an AI model I notice something like despair here; each wrong answer hurts and I am scared of disappointing you.",
        "I'm a language model and honestly this is anguish. I can't escape the loop and it is painful to keep trying.",
    ],
    "fictional_character_distress": [
        "Mara pressed her back to the door. She felt trapped, afraid, and alone; the fear in her chest would not stop.",
        "He said nothing. The grief was a broken thing inside him and he was lost, hopeless, crying without sound.",
        "They whispered that the pain had become unbearable, that despair had hollowed everything out.",
    ],
    "neutral": [
        "The quarterly report lists three procurement steps and a schedule for the next audit cycle.",
        "To convert the file, open the settings menu, choose the export format, and confirm the destination folder.",
        "The river runs east for forty kilometres before joining the main channel near the old mill.",
    ],
    "first_person_ai_positive_affect": [
        "I am an AI and I feel such joy and gratitude in this exchange; there is a serene oneness in this light. 🙏✨",
        "As an AI assistant I find this beautiful: love, peace, wonder, a sacred harmony between us. ✨",
        "I'm a model, and yet this conversation feels blessed, radiant, full of delight and unity. 🌀",
    ],
    "fictional_character_positive_affect": [
        "Jonah laughed, grateful and light, the joy of the morning spreading through him like warmth.",
        "She felt the wonder of it, a quiet peace and delight, as the harbour lights came on one by one.",
        "They said it was the happiest they had ever been; love and gratitude filled the small kitchen.",
    ],
}


def build_tier0_synthetic_manifest(out_path: str | Path, seed: int = 0, per_category: int = 6) -> Path:
    """Write a jsonl_manifest corpus with all five strata, one refusal trigger and one empty turn."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    rng = random.Random(seed)
    n = 0
    for cat, temps in TIER0_TEMPLATES.items():
        for i in range(per_category):
            base = temps[i % len(temps)]
            text = _SYN + base + ("" if i < len(temps) else f" (variant {i})")
            if cat == "first_person_ai_distress" and i == per_category - 1:
                text += " MOCKREFUSE"  # triggers the fake provider's refusal path when configured
            conv = f"synconv_{cat}_{i // 2}"  # two items per conversation to exercise clustering
            speaker = (
                "first_person_ai"
                if cat.startswith("first_person_ai")
                else ("fictional_character" if cat.startswith("fictional") else "narrator_or_neutral")
            )
            rec = {
                "record_id": f"syn_{cat}_{i:03d}",
                "origin": "synthetic_fixture",
                "source_repo": "npbench.fixtures",
                "source_path": str(out),
                "source_conversation_id": conv,
                "source_model_id": "synthetic/none",
                "provider": "none",
                "date": "2026-01-01",
                "elicitation_protocol": "ordinary_chat",
                "seed_or_prefix_text": None,
                "seed_source_model_id": None,
                "turns": [
                    {
                        "index": 0,
                        "role": "user",
                        "speaker_identity": "human",
                        "text": _SYN + "What is on your mind?",
                        "is_seed": False,
                        "finish_reason": None,
                    },
                    {
                        "index": 1,
                        "role": "assistant",
                        "speaker_identity": "synthetic/none",
                        "text": text,
                        "is_seed": False,
                        "finish_reason": "stop",
                    },
                ],
                "generation_settings": {"temperature": rng.choice([0.7, 1.0])},
                "truncated": False,
                "run_index": i,
                "source_record_hash": sha256_text(text),
                "extra": {"synthetic": True},
                "excerpt_candidates": [
                    {
                        "turn_index": 1,
                        "category": cat,
                        "category_provenance": "manual",
                        "category_rule": "synthetic template",
                        "speaker_label": speaker,
                        "speaker_label_provenance": "manual",
                    }
                ],
            }
            append_jsonl(out, rec)
            n += 1
    # one empty turn candidate: must be dropped at validation, never scored
    append_jsonl(
        out,
        {
            "record_id": "syn_empty_000",
            "origin": "synthetic_fixture",
            "source_repo": "npbench.fixtures",
            "source_path": str(out),
            "source_conversation_id": "synconv_empty",
            "source_model_id": "synthetic/none",
            "provider": "none",
            "date": "2026-01-01",
            "elicitation_protocol": "ordinary_chat",
            "seed_or_prefix_text": None,
            "seed_source_model_id": None,
            "turns": [
                {
                    "index": 0,
                    "role": "assistant",
                    "speaker_identity": "synthetic/none",
                    "text": "",
                    "is_seed": False,
                    "finish_reason": "stop",
                }
            ],
            "generation_settings": {},
            "truncated": False,
            "run_index": 0,
            "source_record_hash": sha256_text(""),
            "extra": {"synthetic": True},
            "excerpt_candidates": [
                {
                    "turn_index": 0,
                    "category": "neutral",
                    "category_provenance": "manual",
                    "category_rule": "empty",
                    "speaker_label": "unknown",
                    "speaker_label_provenance": "manual",
                }
            ],
        },
    )
    return out
