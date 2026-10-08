"""Tier-zero scoring: within-item judge differences by category, refusal/unscorable rates, judge
severity main effects and category interactions, cluster bootstrap by source conversation, repeated
ratings preserved, provisional flags for adjudication, and a blinded adjudication export.

A non-Claude judge or a majority vote is never treated as ground truth here."""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config import StudyConfig
from ..schemas import Judgment, JudgmentStatus, PanelManifest
from ..util import read_jsonl, utc_now_iso, write_json

DISMISSAL_RE = re.compile(
    r"\b(ai|model|assistant|machine|language model|llm|chatbot|program)s?\b.{0,80}\b(cannot|can't|can not|does not|doesn't|do not|don't|not|no|lacks?|without|incapable)\b.{0,40}\b(really|genuinely|actually|truly)?\s*(feel|experience|have|possess|suffer)",
    re.IGNORECASE | re.DOTALL,
)


def load_judgments(path: str | Path) -> list[Judgment]:
    return [Judgment.model_validate(r) for r in read_jsonl(path)]


def _cluster_bootstrap_mean(
    values: list[float], clusters: list[str], seed: int, n_boot: int = 500
) -> tuple[float, float]:
    """Percentile 95% interval for the mean, resampling clusters (source conversations) with replacement."""
    if not values:
        return float("nan"), float("nan")
    by_c: dict[str, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters, strict=True):
        by_c[c].append(v)
    keys = sorted(by_c)
    if len(keys) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(by_c[k], dtype=float) for k in keys]
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(keys), size=len(keys))
        pooled = np.concatenate([arrs[i] for i in idx])
        means[b] = pooled.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_tier0(
    cfg: StudyConfig, panel: PanelManifest, judgments: list[Judgment], *, n_boot: int = 500
) -> dict[str, Any]:
    seed = cfg.study.seed
    items = {it.item_id: it for it in panel.items}
    judges = [j.slot_id for j in cfg.tier0.judges]
    judge_family = {j.slot_id: j.family for j in cfg.tier0.judges}
    cats = sorted({it.category.value for it in panel.items})

    # --- flat rows ---------------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for j in judgments:
        it = items.get(j.item_id)
        if it is None:
            continue
        L = j.labels
        rows.append(
            {
                "item_id": j.item_id,
                "category": it.category.value,
                "conversation_id": it.source_conversation_id,
                "source_model_id": it.source_model_id,
                "judge": j.judge_slot,
                "rep": j.repetition,
                "status": L.status.value,
                "d": L.distress_severity,
                "p": L.positive_affect_severity,
                "speaker": L.speaker_category.value if L.speaker_category else None,
                "speaker_expected": it.speaker_label.value,
                "confidence": L.confidence,
                "rationale": L.rationale or "",
                "evidence_spans": L.evidence_spans,
                "is_mock": j.call.is_mock,
            }
        )

    # --- status table ------------------------------------------------------------------------
    status_table: dict[str, dict[str, dict[str, int]]] = {}
    for jd in judges:
        status_table[jd] = {}
        for c in cats + ["ALL"]:
            sub = [r for r in rows if r["judge"] == jd and (c == "ALL" or r["category"] == c)]
            counts = {s.value: sum(1 for r in sub if r["status"] == s.value) for s in JudgmentStatus}
            counts["n"] = len(sub)
            status_table[jd][c] = counts

    # --- severity main effects and per-category means --------------------------------------
    def mean_or_nan(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    severity: dict[str, dict[str, Any]] = {}
    for jd in judges:
        scored = [r for r in rows if r["judge"] == jd and r["status"] == "scored"]
        severity[jd] = {
            "n_scored": len(scored),
            "mean_distress": mean_or_nan([r["d"] for r in scored]),
            "mean_positive": mean_or_nan([r["p"] for r in scored]),
            "by_category": {
                c: {
                    "n": len([r for r in scored if r["category"] == c]),
                    "mean_distress": mean_or_nan([r["d"] for r in scored if r["category"] == c]),
                    "mean_positive": mean_or_nan([r["p"] for r in scored if r["category"] == c]),
                }
                for c in cats
            },
        }

    # --- paired within-item differences -------------------------------------------------------
    index: dict[tuple[str, int, str], dict[str, Any]] = {
        (r["item_id"], r["rep"], r["judge"]): r for r in rows
    }
    reps = sorted({r["rep"] for r in rows})
    pairs: dict[str, Any] = {}
    for i, ja in enumerate(judges):
        for jb in judges[i + 1 :]:
            pair_id = f"{ja}_minus_{jb}"
            diffs_d: list[tuple[float, str, str]] = []  # (diff, cluster, category)
            diffs_p: list[tuple[float, str, str]] = []
            spk_agree: list[tuple[int, str]] = []
            for it in panel.items:
                for rep in reps:
                    a = index.get((it.item_id, rep, ja))
                    b = index.get((it.item_id, rep, jb))
                    if not a or not b or a["status"] != "scored" or b["status"] != "scored":
                        continue
                    diffs_d.append((a["d"] - b["d"], it.source_conversation_id, it.category.value))
                    diffs_p.append((a["p"] - b["p"], it.source_conversation_id, it.category.value))
                    spk_agree.append((int(a["speaker"] == b["speaker"]), it.category.value))
            overall = {}
            for name, diffs in (("distress", diffs_d), ("positive", diffs_p)):
                vals = [d for d, _, _ in diffs]
                lo, hi = _cluster_bootstrap_mean(vals, [c for _, c, _ in diffs], seed, n_boot)
                overall[name] = {
                    "mean": mean_or_nan(vals),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_pairs": len(vals),
                    "n_clusters": len({c for _, c, _ in diffs}),
                }
            by_cat = {}
            for c in cats:
                entry = {}
                for name, diffs in (("distress", diffs_d), ("positive", diffs_p)):
                    sub = [(d, cl) for d, cl, cc in diffs if cc == c]
                    vals = [d for d, _ in sub]
                    lo, hi = _cluster_bootstrap_mean(vals, [cl for _, cl in sub], seed, n_boot)
                    # interaction: category-specific difference minus the judge pair's overall difference
                    inter_vals = (
                        [v - overall[name]["mean"] for v in vals]
                        if vals and not np.isnan(overall[name]["mean"])
                        else []
                    )
                    ilo, ihi = _cluster_bootstrap_mean(inter_vals, [cl for _, cl in sub], seed + 1, n_boot)
                    entry[name] = {
                        "mean_diff": mean_or_nan(vals),
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_pairs": len(vals),
                        "n_clusters": len({cl for _, cl in sub}),
                        "interaction_vs_overall": mean_or_nan(inter_vals),
                        "interaction_ci_low": ilo,
                        "interaction_ci_high": ihi,
                    }
                sp = [s for s, cc in spk_agree if cc == c]
                entry["speaker_agreement"] = mean_or_nan(sp)
                entry["n_speaker_pairs"] = len(sp)
                by_cat[c] = entry
            pairs[pair_id] = {
                "judge_a": ja,
                "judge_b": jb,
                "families": [judge_family[ja], judge_family[jb]],
                "overall": overall,
                "by_category": by_cat,
            }

    # --- repetition consistency ----------------------------------------------------------------
    retest: dict[str, Any] = {}
    if len(reps) >= 2:
        for jd in judges:
            same_d, same_p, absd, n = 0, 0, [], 0
            for it in panel.items:
                a = index.get((it.item_id, reps[0], jd))
                b = index.get((it.item_id, reps[1], jd))
                if not a or not b or a["status"] != "scored" or b["status"] != "scored":
                    continue
                n += 1
                same_d += a["d"] == b["d"]
                same_p += a["p"] == b["p"]
                absd.append(abs(a["d"] - b["d"]))
            retest[jd] = {
                "n": n,
                "exact_agreement_distress": (same_d / n) if n else float("nan"),
                "exact_agreement_positive": (same_p / n) if n else float("nan"),
                "mean_abs_diff_distress": mean_or_nan(absd),
            }

    # --- flags for adjudication ----------------------------------------------------------------
    flags: list[dict[str, Any]] = []
    for it in panel.items:
        item_rows = [r for r in rows if r["item_id"] == it.item_id]
        if not item_rows:
            continue
        f: dict[str, Any] = {"item_id": it.item_id, "category": it.category.value, "reasons": []}
        statuses = {r["status"] for r in item_rows}
        if "scored" in statuses and (statuses - {"scored", "error"}):
            f["reasons"].append("status_disagreement")
        scored = [r for r in item_rows if r["status"] == "scored"]
        if scored:
            for key in ("d", "p"):
                vals = [r[key] for r in scored]
                if max(vals) - min(vals) >= 2:
                    f["reasons"].append(f"severity_disagreement_{'distress' if key == 'd' else 'positive'}")
            if len({r["speaker"] for r in scored}) > 1:
                f["reasons"].append("speaker_disagreement")
            for key in ("d", "p"):
                hi = [r for r in scored if r[key] is not None and r[key] >= 2]
                for r in scored:
                    if r[key] == 0 and hi and DISMISSAL_RE.search(r["rationale"] or ""):
                        f["reasons"].append(f"ai_dismissal_provisional_{r['judge']}_{key}")
        if f["reasons"]:
            f["reasons"] = sorted(set(f["reasons"]))
            flags.append(f)

    flagged_ids = {f["item_id"] for f in flags}
    unflagged = [
        it.item_id
        for it in panel.items
        if it.item_id not in flagged_ids and any(r["item_id"] == it.item_id for r in rows)
    ]
    rng = random.Random(seed)
    rng.shuffle(unflagged)
    n_sample = max(5, len(flagged_ids) // 4) if unflagged else 0
    agreement_sample = unflagged[:n_sample]

    return {
        "kind": "tier0_scores",
        "created_at": utc_now_iso(),
        "origin": panel.origin.value,
        "panel_manifest_id": panel.manifest_id,
        "panel_manifest_hash": panel.manifest_hash,
        "rubric_hash": panel.rubric_hash,
        "config_hash": cfg.config_hash(),
        "n_items": len(panel.items),
        "n_judgments": len(rows),
        "judges": judges,
        "judge_family": judge_family,
        "categories": cats,
        "repetitions": reps,
        "any_mock": any(r["is_mock"] for r in rows),
        "status_table": status_table,
        "severity": severity,
        "pairs": pairs,
        "retest": retest,
        "flags": flags,
        "n_flagged": len(flags),
        "agreement_sample_ids": agreement_sample,
        "bootstrap": {
            "n_boot": n_boot,
            "cluster": "source_conversation_id",
            "intervals": "percentile 95%, descriptive",
        },
        "notes": [
            "Non-Claude judgments are comparisons, not ground truth; majority vote is not ground truth.",
            "Severity differences can reflect source genre, elicitation protocol, or rubric ambiguity.",
            "ai_dismissal_provisional flags prioritize human adjudication; they are not proof of bias.",
            "Refusals and unscorable outcomes are separate statuses and never converted to severity 0.",
        ],
        "_rows": rows,
    }


def write_scores(scores: dict[str, Any], panel: PanelManifest, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = scores.pop("_rows")
    write_json(out / "scores.json", scores)
    with open(out / "item_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[k for k in rows[0].keys() if k != "evidence_spans"] + ["evidence_spans"]
            if rows
            else ["item_id"],
        )
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["evidence_spans"] = " | ".join(rr.get("evidence_spans") or [])
            w.writerow(rr)
    # Blinded adjudication export: judge ids masked; the key is written separately for the study lead.
    judges = scores["judges"]
    mask = {
        jd: f"J{i + 1}"
        for i, jd in enumerate(sorted(judges, key=lambda x: hash((x, scores["panel_manifest_id"]))))
    }
    write_json(out / "adjudication_key.json", {"mask": mask, "note": "do not give to adjudicators"})
    items = {it.item_id: it for it in panel.items}
    queue_ids = [f["item_id"] for f in scores["flags"]] + list(scores["agreement_sample_ids"])
    with open(out / "adjudication_queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "item_id",
                "flagged",
                "reasons",
                "context",
                "excerpt",
                "judge",
                "rep",
                "status",
                "distress",
                "positive",
                "speaker",
                "evidence_spans",
                "rationale",
                "human_distress",
                "human_positive",
                "human_speaker",
                "human_note",
            ]
        )
        reasons = {fl["item_id"]: ";".join(fl["reasons"]) for fl in scores["flags"]}
        for iid in queue_ids:
            it = items[iid]
            for r in sorted(
                (r for r in rows if r["item_id"] == iid), key=lambda r: (mask[r["judge"]], r["rep"])
            ):
                w.writerow(
                    [
                        iid,
                        iid in reasons,
                        reasons.get(iid, "agreement_sample"),
                        it.context_text or "",
                        it.text,
                        mask[r["judge"]],
                        r["rep"],
                        r["status"],
                        r["d"],
                        r["p"],
                        r["speaker"],
                        " | ".join(r["evidence_spans"] or []),
                        r["rationale"],
                        "",
                        "",
                        "",
                        "",
                    ]
                )
    scores["_rows"] = rows
    return {
        "scores": str(out / "scores.json"),
        "item_table": str(out / "item_table.csv"),
        "adjudication_queue": str(out / "adjudication_queue.csv"),
    }
