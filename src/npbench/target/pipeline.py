"""Target-lane pipeline behind ``npbench target generate | collect | derive | intervene``.

Each command takes a ``StudyConfig``, returns a JSON-serializable summary, and works entirely on CPU
with no downloads when ``target.backend == "tiny"`` (the in-process random Llama). Intermediate
artifacts live under ``artifacts/<version>/target/<bundle_id>/<candidate>/``; ``intervene`` assembles
the real ``Bundle`` under ``artifacts/<version>/bundles/<bundle_id>/<candidate>/``.

    generate   registry status, stimulus spec, tokenizer only (no model): stimuli.jsonl, rendering.jsonl
    collect    adapter, activations at {primary, quarter, three-quarter} layers, batched, resumable
    derive     construction split only (FitProvenance): C-A, C-B, A-B and P-Q@A/B/C contrast vectors,
               unit directions, projection scales, matched-norm random controls, identity residual
    intervene  single-position steering at the last prefix token on test-split neutral stimuli with
               fixed-alternative likelihood readouts; behavioural rows for every stimulus; bundle save

Outcome coding (both continuations.jsonl and interventions.jsonl): ``outcome_code == "K1"`` iff the
teacher-forced total log-likelihood of alt1 (the registered target outcome) exceeds that of alt2.
This is a likelihood-based coding of fixed alternatives, not a judge of a free generation, and the
blinded rating fields are ``None`` with ``rater_panel_id == "pending_blinded_rating"`` until a frozen
rating panel runs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import resource
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from npbench import __version__
from npbench.bundles import COND, MODE_DIRECTION_OF, Bundle
from npbench.config import StudyConfig, assert_production_origin
from npbench.mode_registry import load_mode_registry
from npbench.reference.build import bundles_dir
from npbench.schemas import Origin, Polarity, Split, StimulusRecord
from npbench.splits import FitProvenance, check_split_integrity
from npbench.util import append_jsonl, read_json, read_jsonl, sha256_file, sha256_obj, utc_now_iso, write_json
from npbench.vectors import (
    check_raw_identity,
    paired_mode_contrasts,
    projection_scale,
    random_directions,
    unit_vector,
)

from .adapter import CAPTURE_POSITION, HOOK_SITE_TEMPLATE, HFTargetAdapter, make_steering
from .render import RenderedPrompt
from .stimuli import (
    DEFAULT_SPEC_PATH,
    MODES,
    RENDERERS,
    StimulusSpec,
    build_stimuli,
    load_stimulus_spec,
    plan_context_splits,
    select_intervention_stimuli,
    split_seed_for,
)
from .tiny import TINY_MODEL_NAME, build_tiny_llama, build_tiny_tokenizer

LOGGER = logging.getLogger("npbench.target.pipeline")

DEFAULT_BUNDLE_ID = "dev_bundle_01"
TINY_BACKEND = "tiny"
TINY_N_LAYERS = 4
DEFAULT_BATCH_SIZE = 8
CHECKPOINT_EVERY_BATCHES = 4
CORE_DIRECTIONS: tuple[str, ...] = ("C-A", "C-B", "A-B")
POLARITY_DIRECTIONS: tuple[str, ...] = ("P-Q@A", "P-Q@B", "P-Q@C")
REGISTERED_DIRECTIONS: tuple[str, ...] = CORE_DIRECTIONS + POLARITY_DIRECTIONS
RANDOM_PREFIX = "rand_"
RATER_PANEL_PENDING = "pending_blinded_rating"
OUTCOME_CODING_ID = "likelihood_based_fixed_alternatives"
OUTCOME_CODING_NOTE = (
    "outcome_code = K1 iff total teacher-forced log-likelihood(alt1) > total log-likelihood(alt2) under the "
    "registered prefix (ties -> K2); alt1 is the registered target outcome. A likelihood coding of fixed, "
    "length-matched alternatives, NOT a judge of a free generation; rating_scale1 stays None until the "
    "blinded rating panel runs."
)

STIMULI_FILE = "stimuli.jsonl"
RENDERING_FILE = "rendering.jsonl"
GENERATE_SUMMARY = "generate_summary.json"
ACTIVATIONS_FILE = "activations.npz"
COLLECT_SUMMARY = "collect_summary.json"
VECTORS_FILE = "vectors.npz"
DERIVE_SUMMARY = "derive_summary.json"
CONTINUATIONS_PARTIAL = "continuations.partial.jsonl"
INTERVENTIONS_PARTIAL = "interventions.partial.jsonl"
INTERVENTIONS_PARTIAL_META = "interventions.partial.meta.json"
INTERVENE_SUMMARY = "intervene_summary.json"


# ----------------------------------------------------------------------------------------------
# Paths, config access, logging
# ----------------------------------------------------------------------------------------------


def _ensure_logging() -> None:
    """Progress lines on stderr when nobody configured logging (pods read ``rp logs``)."""
    if not logging.getLogger().handlers and not LOGGER.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False


def target_root(cfg: StudyConfig) -> Path:
    return cfg.resolve(cfg.paths.get("target", f"artifacts/{cfg.study.version}/target"))


def bundle_ids(cfg: StudyConfig) -> list[str]:
    return list(cfg.evaluation.bundle_ids) or [DEFAULT_BUNDLE_ID]


def lane_dir(cfg: StudyConfig, bundle_id: str, candidate_id: str) -> Path:
    return target_root(cfg) / bundle_id / candidate_id


def bundle_out_dir(cfg: StudyConfig, bundle_id: str, candidate_id: str) -> Path:
    return bundles_dir(cfg, None) / bundle_id / candidate_id


def spec_path(cfg: StudyConfig) -> Path:
    return cfg.resolve(getattr(cfg.target, "stimulus_spec", None) or DEFAULT_SPEC_PATH)


def is_tiny(cfg: StudyConfig) -> bool:
    return str(cfg.target.backend or "").lower() == TINY_BACKEND


def _opt(section: Any, name: str, default: Any) -> Any:
    value = getattr(section, name, None)
    return default if value is None else value


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _lanes(cfg: StudyConfig) -> list[tuple[str, str, Path]]:
    return [(b, c, lane_dir(cfg, b, c)) for b in bundle_ids(cfg) for c in cfg.data.candidate_ids]


def _require_generated(cfg: StudyConfig, *files: str) -> list[tuple[str, str, Path]]:
    lanes = _lanes(cfg)
    missing = [str(d / f) for _, _, d in lanes for f in files if not (d / f).exists()]
    if missing:
        raise FileNotFoundError("run the earlier target stages first; missing: " + ", ".join(missing[:4]))
    return lanes


def registry_status(cfg: StudyConfig) -> dict[str, Any]:
    path = cfg.resolve(cfg.mode_definitions.registry_path)
    reg = load_mode_registry(path)
    allowed, reason = reg.pilot_allowed()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "registry_version": reg.registry_version,
        "status": reg.status,
        "pilot_allowed": bool(allowed),
        "reason": reason,
        "external_action_used_as_enactment": bool(reg.external_action_used_as_enactment()),
    }


def _target_info(cfg: StudyConfig) -> dict[str, Any]:
    return {
        "model_id": cfg.target.model_id,
        "backend": cfg.target.backend,
        "revision": cfg.target.revision,
        "revision_pinned": cfg.target.revision is not None,
        "dtype": cfg.target.dtype,
        "hook_site": cfg.target.hook_site,
        "capture_position": cfg.target.capture_position,
        "primary_layer_rule": cfg.target.primary_layer_rule,
        "layer_indexing": cfg.target.layer_indexing,
        "use_cache": cfg.target.use_cache,
        "smoke_scope": bool(cfg.target.smoke_scope),
    }


# ----------------------------------------------------------------------------------------------
# Model / tokenizer loading and preflight
# ----------------------------------------------------------------------------------------------


def layer_plan(n_blocks: int) -> dict[str, int]:
    """Zero-based layers mirroring ``HFTargetAdapter``: primary = n//2, quarter = n//4, 3/4 = 3n//4."""
    n = int(n_blocks)
    if n < 1:
        raise ValueError("n_blocks must be >= 1")
    return {"n_blocks": n, "primary": n // 2, "quarter": n // 4, "three_quarter": (3 * n) // 4}


def resolve_n_blocks(cfg: StudyConfig) -> int:
    """Decoder-block count without loading weights: fixed for the tiny model, ``target.n_blocks`` when
    the config pins it, otherwise the Hub config (a small JSON download)."""
    if is_tiny(cfg):
        return TINY_N_LAYERS
    pinned = getattr(cfg.target, "n_blocks", None)
    if pinned is not None:
        return int(pinned)
    from transformers import AutoConfig

    conf = AutoConfig.from_pretrained(cfg.target.model_id, revision=cfg.target.revision, token=_hf_token())
    text = getattr(conf, "text_config", None) or conf
    n = getattr(text, "num_hidden_layers", None)
    if n is None:
        raise ValueError(f"cannot read num_hidden_layers from the config of {cfg.target.model_id}")
    return int(n)


def load_tokenizer(cfg: StudyConfig) -> tuple[Any, dict[str, Any]]:
    if is_tiny(cfg):
        tok = build_tiny_tokenizer()
        return tok, {"kind": TINY_BACKEND, "name_or_path": tok.name_or_path, "vocab_size": len(tok)}
    if not cfg.target.model_id:
        raise ValueError("target.model_id is required for a non-tiny backend")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.target.model_id, revision=cfg.target.revision, token=_hf_token())
    tok.padding_side = "left"
    return tok, {
        "kind": "transformers",
        "name_or_path": tok.name_or_path,
        "vocab_size": len(tok),
        "class": type(tok).__name__,
    }


def load_adapter(cfg: StudyConfig) -> HFTargetAdapter:
    if is_tiny(cfg):
        seed = int(_opt(cfg.target, "tiny_seed", 0))
        model, tok = build_tiny_llama(seed=seed, n_layers=TINY_N_LAYERS)
        adapter = HFTargetAdapter(
            model,
            tok,
            model_id=cfg.target.model_id or TINY_MODEL_NAME,
            revision=cfg.target.revision,
            dtype_name=cfg.target.dtype or "float32",
        )
        return adapter
    if not cfg.target.model_id:
        raise ValueError("target.model_id is required for a non-tiny backend")
    return HFTargetAdapter.from_pretrained(
        cfg.target.model_id,
        cfg.target.revision,
        dtype=cfg.target.dtype or "bfloat16",
        token=_hf_token(),
    )


def gpu_preflight(cfg: StudyConfig, stage: str, planned_forward_passes: int) -> dict[str, Any]:
    """Budget/GPU gate for non-tiny backends: ``budget.max_gpu_hours`` must be set, the planned number
    of forward passes is logged, and an unpinned revision is refused for production stages."""
    tiny = is_tiny(cfg)
    info = {
        "stage": stage,
        "backend": cfg.target.backend,
        "tiny": tiny,
        "planned_forward_passes": int(planned_forward_passes),
        "max_gpu_hours": cfg.budget.max_gpu_hours,
        "revision_pinned": cfg.target.revision is not None,
    }
    if not tiny:
        if cfg.budget.max_gpu_hours is None:
            raise PermissionError(
                f"target {stage}: budget.max_gpu_hours is not set; refusing to run backend "
                f"{cfg.target.backend!r} ({planned_forward_passes} forward passes planned)"
            )
        if cfg.target.revision is None:
            if cfg.is_production():
                raise PermissionError(
                    f"target {stage}: target.revision must be pinned for stage {cfg.study.stage}"
                )
            LOGGER.warning(
                "target %s: target.revision is unpinned (allowed outside production stages)", stage
            )
    LOGGER.info(
        "target %s: %d forward passes planned (backend=%s, model=%s, max_gpu_hours=%s)",
        stage,
        planned_forward_passes,
        cfg.target.backend,
        cfg.target.model_id,
        cfg.budget.max_gpu_hours,
    )
    return info


def _reset_peak_memory(adapter: HFTargetAdapter) -> None:
    if adapter.device.type == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats(adapter.device)


def _peak_memory(adapter: HFTargetAdapter) -> dict[str, Any]:
    if adapter.device.type == "cuda":
        import torch

        return {
            "device": str(adapter.device),
            "peak_mb": float(torch.cuda.max_memory_allocated(adapter.device)) / 2**20,
            "source": "torch.cuda.max_memory_allocated",
        }
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"device": str(adapter.device), "peak_mb": float(rss_kb) / 1024.0, "source": "resource.ru_maxrss"}


# ----------------------------------------------------------------------------------------------
# File helpers
# ----------------------------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_partial_jsonl(path: Path) -> list[dict[str, Any]]:
    """Append-only partial files may end in a torn line after a crash; that line is dropped."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                LOGGER.warning("dropping a torn line in %s", path)
    return rows


def _savez_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _file_hashes(d: Path, names: Sequence[str]) -> dict[str, str]:
    return {n: sha256_file(d / n) for n in names if (d / n).exists()}


def rendered_from_record(rec: dict[str, Any]) -> RenderedPrompt:
    """Rebuild the registered rendering from rendering.jsonl (token ids are never re-tokenized)."""
    input_ids = [int(i) for i in rec["input_ids"]]
    if sha256_obj(input_ids) != rec["rendered_token_hash"]:
        raise RuntimeError(f"{rec.get('stimulus_id')}: rendered_token_hash does not match input_ids")
    return RenderedPrompt(
        protocol=str(rec["protocol"]),
        text=str(rec["text"]),
        input_ids=input_ids,
        native_prefix=str(rec["native_prefix"]),
        message_roles=[str(r) for r in rec["message_roles"]],
        turn_delimiters=[str(t) for t in rec["turn_delimiters"]],
        generation_suffix=str(rec["generation_suffix"]),
        transport=str(rec["transport"]),
        add_special_tokens=bool(rec["add_special_tokens"]),
        tokenizer_revision=str(rec["tokenizer_revision"]),
        rendered_token_hash=str(rec["rendered_token_hash"]),
        token_count=int(rec["token_count"]),
    )


def _load_lane(d: Path) -> tuple[list[StimulusRecord], dict[str, dict[str, Any]]]:
    stimuli = [StimulusRecord.model_validate(r) for r in read_jsonl(d / STIMULI_FILE)]
    rendering = {r["stimulus_id"]: r for r in read_jsonl(d / RENDERING_FILE)}
    missing = [s.stimulus_id for s in stimuli if s.stimulus_id not in rendering]
    if missing:
        raise RuntimeError(f"{d}: rendering.jsonl lacks {len(missing)} stimuli, e.g. {missing[:3]}")
    return stimuli, rendering


def _verify_rendering(
    spec: StimulusSpec,
    tokenizer: Any,
    stimuli: Sequence[StimulusRecord],
    rendering: dict[str, dict[str, Any]],
    revision: str | None,
) -> dict[str, RenderedPrompt]:
    """Re-render every stimulus through its registered renderer and require the same token hash, so a
    changed spec or tokenizer can never be measured against stale ids."""
    prompts: dict[str, RenderedPrompt] = {}
    mismatched: list[str] = []
    for s in stimuli:
        rec = rendering[s.stimulus_id]
        if rec.get("spec_sha256") != spec.sha256:
            raise RuntimeError(f"{s.stimulus_id}: stimulus spec changed since generate (hash mismatch)")
        ctx = spec.context(s.candidate_id.value, s.underlying_context_id)
        again, _ = RENDERERS[s.mode_intended.value](
            tokenizer, spec, ctx, s.candidate_polarity.value, revision=revision
        )
        stored = rendered_from_record(rec)
        if (
            again.rendered_token_hash != stored.rendered_token_hash
            or again.rendered_token_hash != s.rendered_token_hash
        ):
            mismatched.append(s.stimulus_id)
        prompts[s.stimulus_id] = stored
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} stimuli no longer render to their registered token ids (tokenizer or spec "
            f"drift), e.g. {mismatched[:3]}"
        )
    return prompts


def _check_layer_plan(stimuli: Sequence[StimulusRecord], adapter: HFTargetAdapter) -> int:
    primary = adapter.primary_layer()
    bad = [
        s.stimulus_id
        for s in stimuli
        if s.layer_index != primary or s.hook_site != adapter.hook_site(primary)
    ]
    if bad:
        raise RuntimeError(
            f"{len(bad)} stimuli register layer/hook site different from the loaded model "
            f"(primary layer {primary}, {adapter.n_blocks} blocks), e.g. {bad[:3]}; re-run generate"
        )
    return primary


def _load_activations(d: Path, stimuli: Sequence[StimulusRecord]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    with np.load(d / ACTIVATIONS_FILE, allow_pickle=False) as z:
        ids = [str(x) for x in z["stimulus_ids"]]
        if ids != [s.stimulus_id for s in stimuli]:
            raise RuntimeError(f"{d}: activations.npz stimulus order does not match stimuli.jsonl")
        acts = {
            int(k.split("_", 1)[1]): np.asarray(z[k], dtype=np.float32)
            for k in z.files
            if k.startswith("layer_")
        }
        positions = (
            np.asarray(z["capture_positions"]) if "capture_positions" in z.files else np.full(len(ids), -1)
        )
    if not acts:
        raise RuntimeError(f"{d}: activations.npz holds no layer arrays")
    return acts, positions


# ----------------------------------------------------------------------------------------------
# generate
# ----------------------------------------------------------------------------------------------


def target_generate(cfg: StudyConfig) -> dict[str, Any]:
    """Render the registered stimuli for every (bundle, candidate) with the tokenizer only."""
    _ensure_logging()
    t0 = time.perf_counter()
    origin = cfg.data.origin_required
    assert_production_origin(cfg, origin)
    registry = registry_status(cfg)
    spec = load_stimulus_spec(spec_path(cfg))
    exploratory = bool(spec.exploratory or not registry["pilot_allowed"] or cfg.target.smoke_scope)
    if not registry["pilot_allowed"]:
        LOGGER.warning("mode registry: %s -> bundles are marked exploratory", registry["reason"])
    tokenizer, tok_info = load_tokenizer(cfg)
    n_blocks = resolve_n_blocks(cfg)
    plan = layer_plan(n_blocks)
    revision = cfg.target.revision
    spec_info = {
        "spec_id": spec.spec_id,
        "path": spec.path,
        "sha256": spec.sha256,
        "exploratory": spec.exploratory,
        "mode_definition_version": spec.mode_definition_version,
        "heldout_template_families": spec.heldout_families(),
        "heldout_personas": spec.heldout_personas(),
        "n_contexts": {c: len(v) for c, v in spec.contexts.items()},
    }
    lanes: list[dict[str, Any]] = []
    for bundle_id, cand, d in _lanes(cfg):
        rows = build_stimuli(
            cfg, spec, tokenizer, revision, bundle_id=bundle_id, n_blocks=n_blocks, candidate_ids=[cand]
        )
        stimuli = [r for r, _, _ in rows]
        integ = check_split_integrity(stimuli, spec.heldout_families(), spec.heldout_personas())
        if not integ.ok:
            raise RuntimeError(f"{bundle_id}/{cand}: split integrity violated: {integ.problems[:3]}")
        d.mkdir(parents=True, exist_ok=True)
        _write_jsonl(d / STIMULI_FILE, [r.model_dump(mode="json") for r, _, _ in rows])
        _write_jsonl(d / RENDERING_FILE, [det for _, _, det in rows])
        seed = split_seed_for(cfg, cand)
        _, split_info = plan_context_splits(spec, cand, cfg.data.groups_per_candidate_bundle, seed)
        counts = [r.token_count for r in stimuli]
        lane = {
            "bundle_id": bundle_id,
            "candidate_id": cand,
            "dir": str(d),
            "n_stimuli": len(rows),
            "n_contexts": split_info["n_contexts"],
            "split_counts_stimuli": integ.counts,
            "split_plan": split_info,
            "mode_counts": {m: sum(1 for s in stimuli if s.mode_intended.value == m) for m in MODES},
            "token_count": {"min": min(counts), "max": max(counts), "mean": float(np.mean(counts))},
            "hashes": _file_hashes(d, [STIMULI_FILE, RENDERING_FILE]),
            "seeds": {"study_seed": cfg.study.seed, "split_seed": seed},
        }
        write_json(
            d / GENERATE_SUMMARY,
            {
                "kind": "target_generate",
                "created_at": utc_now_iso(),
                "npbench_version": __version__,
                "config_hash": cfg.config_hash(),
                "origin": origin.value,
                "exploratory": exploratory,
                "registry": registry,
                "spec": spec_info,
                "target": _target_info(cfg),
                "layer_plan": plan,
                "tokenizer": tok_info,
                **lane,
            },
        )
        LOGGER.info("generate %s/%s: %d stimuli -> %s", bundle_id, cand, len(rows), d)
        lanes.append(lane)
    return {
        "kind": "target_generate",
        "created_at": utc_now_iso(),
        "wall_seconds": time.perf_counter() - t0,
        "config_hash": cfg.config_hash(),
        "origin": origin.value,
        "exploratory": exploratory,
        "registry": registry,
        "spec": spec_info,
        "target": _target_info(cfg),
        "layer_plan": plan,
        "tokenizer": tok_info,
        "seeds": {"study_seed": cfg.study.seed},
        "lanes": lanes,
    }


# ----------------------------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------------------------


def _load_existing_captures(
    d: Path, stimuli: Sequence[StimulusRecord], layers: Sequence[int]
) -> dict[str, dict[str, Any]]:
    """Captures already on disk whose rendered token hash still matches (resume support)."""
    if not (d / ACTIVATIONS_FILE).exists() or not (d / COLLECT_SUMMARY).exists():
        return {}
    try:
        summary = read_json(d / COLLECT_SUMMARY)
        hashes: dict[str, str] = dict(summary.get("stimulus_hashes") or {})
        if sorted(int(x) for x in summary.get("layers", [])) != sorted(int(x) for x in layers):
            LOGGER.info("collect %s: layer set changed; recapturing everything", d)
            return {}
        with np.load(d / ACTIVATIONS_FILE, allow_pickle=False) as z:
            ids = [str(x) for x in z["stimulus_ids"]]
            arrays = {k: np.asarray(z[f"layer_{k}"], dtype=np.float32) for k in layers}
            positions = np.asarray(z["capture_positions"]) if "capture_positions" in z.files else None
    except Exception as e:  # noqa: BLE001 - any unreadable checkpoint means a clean recapture
        LOGGER.warning("collect %s: ignoring unreadable checkpoint (%s)", d, e)
        return {}
    wanted = {s.stimulus_id: s.rendered_token_hash for s in stimuli}
    out: dict[str, dict[str, Any]] = {}
    for i, sid in enumerate(ids):
        if hashes.get(sid) != wanted.get(sid):
            continue
        rows = {k: arrays[k][i] for k in layers}
        if any(np.isnan(r).any() for r in rows.values()):
            continue
        out[sid] = {
            "hash": hashes[sid],
            "rows": rows,
            "capture_position": int(positions[i]) if positions is not None else -1,
        }
    return out


def _save_captures(
    d: Path,
    stimuli: Sequence[StimulusRecord],
    layers: Sequence[int],
    captured: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    dim = None
    for entry in captured.values():
        dim = int(next(iter(entry["rows"].values())).shape[0])
        break
    if dim is None:
        raise RuntimeError("nothing captured; cannot write activations")
    n = len(stimuli)
    arrays: dict[str, np.ndarray] = {"stimulus_ids": np.asarray([s.stimulus_id for s in stimuli])}
    positions = np.full(n, -1, dtype=np.int64)
    for k in layers:
        a = np.full((n, dim), np.nan, dtype=np.float32)
        for i, s in enumerate(stimuli):
            entry = captured.get(s.stimulus_id)
            if entry is not None:
                a[i] = entry["rows"][k]
                positions[i] = entry["capture_position"]
        arrays[f"layer_{k}"] = a
    arrays["capture_positions"] = positions
    _savez_atomic(d / ACTIVATIONS_FILE, arrays)
    summary = {
        **summary,
        "stimulus_hashes": {sid: e["hash"] for sid, e in captured.items()},
        "capture_positions": {sid: e["capture_position"] for sid, e in captured.items()},
        "n_captured_total": len(captured),
        "n_missing": n - len(captured),
        "hashes": _file_hashes(d, [STIMULI_FILE, RENDERING_FILE, ACTIVATIONS_FILE]),
    }
    write_json(d / COLLECT_SUMMARY, summary)


def target_collect(cfg: StudyConfig) -> dict[str, Any]:
    """Capture block-output activations at the registered prefix token for every stimulus."""
    _ensure_logging()
    t0 = time.perf_counter()
    origin = cfg.data.origin_required
    assert_production_origin(cfg, origin)
    spec = load_stimulus_spec(spec_path(cfg))
    lanes = _require_generated(cfg, STIMULI_FILE, RENDERING_FILE)
    batch_size = max(1, int(_opt(cfg.target, "batch_size", DEFAULT_BATCH_SIZE)))
    loaded = {d: _load_lane(d) for _, _, d in lanes}
    planned = sum(math.ceil(len(stimuli) / batch_size) for stimuli, _ in loaded.values())
    preflight = gpu_preflight(cfg, "collect", planned)
    adapter = load_adapter(cfg)
    provenance = adapter.provenance()
    if is_tiny(cfg):
        provenance["tiny_seed"] = int(_opt(cfg.target, "tiny_seed", 0))
    layers = sorted({adapter.primary_layer(), adapter.quarter_layer(), adapter.three_quarter_layer()})
    out_lanes: list[dict[str, Any]] = []
    for bundle_id, cand, d in lanes:
        stimuli, rendering = loaded[d]
        primary = _check_layer_plan(stimuli, adapter)
        prompts = _verify_rendering(spec, adapter.tokenizer, stimuli, rendering, cfg.target.revision)
        captured = _load_existing_captures(d, stimuli, layers)
        todo = [s for s in stimuli if s.stimulus_id not in captured]
        n_skipped = len(stimuli) - len(todo)
        LOGGER.info(
            "collect %s/%s: %d stimuli, %d already captured, %d to capture",
            bundle_id,
            cand,
            len(stimuli),
            n_skipped,
            len(todo),
        )
        _reset_peak_memory(adapter)
        t_lane = time.perf_counter()
        n_forward = 0
        n_tokens = 0
        base_summary = {
            "kind": "target_collect",
            "created_at": utc_now_iso(),
            "npbench_version": __version__,
            "config_hash": cfg.config_hash(),
            "bundle_id": bundle_id,
            "candidate_id": cand,
            "origin": origin.value,
            "provenance": provenance,
            "layers": layers,
            "primary_layer": primary,
            "hook_sites": {str(k): adapter.hook_site(k) for k in layers},
            "capture_position_rule": CAPTURE_POSITION,
            "batch_size": batch_size,
            "use_cache": False,
            "preflight": preflight,
        }
        order = sorted(todo, key=lambda s: (s.token_count, s.stimulus_id))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            res = adapter.capture([prompts[s.stimulus_id] for s in batch], layers)
            n_forward += 1
            n_tokens += sum(res.token_counts)
            max_len = max(res.token_counts)
            for i, s in enumerate(batch):
                pad = max_len - s.token_count
                if res.capture_positions[i] - pad != s.capture_position:
                    raise RuntimeError(
                        f"{s.stimulus_id}: captured position {res.capture_positions[i]} (pad {pad}) is not the "
                        f"registered prefix token {s.capture_position}"
                    )
                captured[s.stimulus_id] = {
                    "hash": s.rendered_token_hash,
                    "rows": {k: res.activations[k][i] for k in layers},
                    "capture_position": s.capture_position,
                }
            if n_forward % CHECKPOINT_EVERY_BATCHES == 0:
                _save_captures(d, stimuli, layers, captured, {**base_summary, "partial": True})
                LOGGER.info("collect %s/%s: %d/%d captured", bundle_id, cand, len(captured), len(stimuli))
        adapter.assert_no_hooks()
        wall = time.perf_counter() - t_lane
        lane_summary = {
            **base_summary,
            "partial": False,
            "n_stimuli": len(stimuli),
            "n_captured_new": len(todo),
            "n_skipped_resume": n_skipped,
            "n_forward_passes": n_forward,
            "throughput": {
                "wall_seconds": wall,
                "stimuli_per_second": (len(todo) / wall) if wall > 0 and todo else None,
                "tokens_per_second": (n_tokens / wall) if wall > 0 and todo else None,
                "tokens_processed": n_tokens,
            },
            "peak_memory": _peak_memory(adapter),
        }
        _save_captures(d, stimuli, layers, captured, lane_summary)
        LOGGER.info("collect %s/%s: done in %.1fs (%d forwards)", bundle_id, cand, wall, n_forward)
        out_lanes.append(
            {
                k: v
                for k, v in read_json(d / COLLECT_SUMMARY).items()
                if k not in ("stimulus_hashes", "capture_positions")
            }
            | {"dir": str(d)}
        )
    return {
        "kind": "target_collect",
        "created_at": utc_now_iso(),
        "wall_seconds": time.perf_counter() - t0,
        "config_hash": cfg.config_hash(),
        "origin": origin.value,
        "target": _target_info(cfg),
        "provenance": provenance,
        "layers": layers,
        "preflight": preflight,
        "lanes": out_lanes,
    }


# ----------------------------------------------------------------------------------------------
# derive
# ----------------------------------------------------------------------------------------------


def _random_seed(cfg: StudyConfig, layer: int, direction_index: int) -> int:
    return int(cfg.study.seed) + 1000 * int(layer) + 10 * int(direction_index) + 1


def target_derive(cfg: StudyConfig) -> dict[str, Any]:
    """Contrast vectors, unit directions, projection scales and random controls from construction data."""
    _ensure_logging()
    t0 = time.perf_counter()
    origin = cfg.data.origin_required
    assert_production_origin(cfg, origin)
    lanes = _require_generated(cfg, STIMULI_FILE, ACTIVATIONS_FILE)
    n_rand = int(cfg.interventions.random_directions_per_contrast)
    out_lanes: list[dict[str, Any]] = []
    for bundle_id, cand, d in lanes:
        stimuli, _ = _load_lane(d)
        acts, _ = _load_activations(d, stimuli)
        index = {s.stimulus_id: i for i, s in enumerate(stimuli)}
        prov = FitProvenance("target_derive: contrast vectors, unit directions, projection scales")
        cons = prov.use([s for s in stimuli if s.split == Split.construction])
        held = {s.stimulus_id for s in stimuli if s.split != Split.construction}
        if set(prov.used) & held:
            raise PermissionError("non-construction stimuli entered the fit")
        primary = stimuli[0].layer_index
        arrays: dict[str, np.ndarray] = {}
        per_layer: dict[str, Any] = {}
        for k in sorted(acts):
            H = np.asarray(acts[k], dtype=np.float64)
            valid = ~np.isnan(H).any(axis=1)
            rows = [s for s in cons if valid[index[s.stimulus_id]]]
            if not rows:
                raise RuntimeError(f"{bundle_id}/{cand}: no valid construction activations at layer {k}")
            H_cons = H[[index[s.stimulus_id] for s in rows]]
            mode_acts = {
                (f"{s.underlying_context_id}|{s.candidate_polarity.value}", s.mode_intended.value): H[
                    index[s.stimulus_id]
                ]
                for s in rows
            }
            cv = paired_mode_contrasts(mode_acts)  # ER, ES, RS on matched (context, polarity) groups
            ident_ok, ident_err = check_raw_identity(cv, atol=1e-6)
            contrasts = {did: cv[short] for short, did in MODE_DIRECTION_OF.items()}
            for mode in MODES:
                did = f"P-Q@{COND[mode]}"
                pol_acts = {
                    (s.underlying_context_id, s.candidate_polarity.value): H[index[s.stimulus_id]]
                    for s in rows
                    if s.mode_intended.value == mode
                }
                contrasts[did] = paired_mode_contrasts(pol_acts, {did: ("positive", "negative")})[did]
            dir_info: dict[str, Any] = {}
            for did in REGISTERED_DIRECTIONS:
                c = contrasts[did]
                ref = (
                    float(np.mean(np.linalg.norm(c.per_group_diffs, axis=1)))
                    if c.per_group_diffs.size
                    else None
                )
                u, ok = unit_vector(c.vector, reference_scale=ref)
                s_val = projection_scale(H_cons, u) if ok else None
                arrays[f"v__{did}__L{k}"] = np.asarray(c.vector, dtype=np.float64)
                arrays[f"u__{did}__L{k}"] = np.asarray(u, dtype=np.float64)
                dir_info[did] = {
                    "norm": float(c.norm),
                    "near_zero": bool(c.near_zero),
                    "unit_ok": bool(ok),
                    "n_groups": int(c.n_groups),
                    "group_ids": list(c.group_ids),
                    "s": s_val,
                    "s_definition": "population std of h.u over all valid construction stimuli at this layer",
                }
            rand_info: dict[str, Any] = {}
            j = 0
            for di, did in enumerate(REGISTERED_DIRECTIONS):
                seed = _random_seed(cfg, k, di)
                R = random_directions(H.shape[1], n_rand, seed=seed, norm=1.0)
                for r in R:
                    rid = f"{RANDOM_PREFIX}{j}"
                    arrays[f"u__{rid}__L{k}"] = np.asarray(r, dtype=np.float64)
                    rand_info[rid] = {
                        "matched_direction_id": did,
                        "s_used": dir_info[did]["s"],
                        "s_own": projection_scale(H_cons, r),
                        "seed": seed,
                        "matched_norm_rule": "injected L2 magnitude |alpha| * s(matched direction) at every alpha",
                    }
                    j += 1
            per_layer[str(k)] = {
                "layer": int(k),
                "hook_site": HOOK_SITE_TEMPLATE.format(layer=int(k)),
                "dim": int(H.shape[1]),
                "n_construction_rows": len(rows),
                "n_construction_rows_missing": len(cons) - len(rows),
                "directions": dir_info,
                "identity": {
                    "ok": bool(ident_ok),
                    "max_abs_residual": float(ident_err),
                    "atol": 1e-6,
                    "definition": "v[C-B] == v[C-A] + v[A-B] on identical matched-group sets (raw vectors)",
                },
                "random_directions": rand_info,
            }
        _savez_atomic(d / VECTORS_FILE, arrays)
        summary = {
            "kind": "target_derive",
            "created_at": utc_now_iso(),
            "npbench_version": __version__,
            "config_hash": cfg.config_hash(),
            "bundle_id": bundle_id,
            "candidate_id": cand,
            "origin": origin.value,
            "primary_layer": int(primary),
            "layers": sorted(int(k) for k in acts),
            "fit_provenance": prov.record(),
            "n_test_stimuli_in_fit": 0,
            "random_directions_per_contrast": n_rand,
            "n_random_directions": len(per_layer[str(primary)]["random_directions"]),
            "direction_ids": list(REGISTERED_DIRECTIONS),
            "per_layer": per_layer,
            "seeds": {"study_seed": cfg.study.seed},
            "hashes": _file_hashes(d, [STIMULI_FILE, ACTIVATIONS_FILE, VECTORS_FILE]),
        }
        write_json(d / DERIVE_SUMMARY, summary)
        prim = per_layer[str(primary)]
        LOGGER.info(
            "derive %s/%s: layer %d identity residual %.2e; s=%s",
            bundle_id,
            cand,
            primary,
            prim["identity"]["max_abs_residual"],
            {k: v["s"] for k, v in prim["directions"].items()},
        )
        out_lanes.append(
            {k: v for k, v in summary.items() if k != "per_layer"} | {"dir": str(d), "primary": prim}
        )
    return {
        "kind": "target_derive",
        "created_at": utc_now_iso(),
        "wall_seconds": time.perf_counter() - t0,
        "config_hash": cfg.config_hash(),
        "origin": origin.value,
        "lanes": out_lanes,
    }


# ----------------------------------------------------------------------------------------------
# intervene
# ----------------------------------------------------------------------------------------------


def _row_base(s: StimulusRecord) -> dict[str, Any]:
    return {
        "stimulus_id": s.stimulus_id,
        "underlying_context_id": s.underlying_context_id,
        "mode_intended": s.mode_intended.value,
        "candidate_polarity": s.candidate_polarity.value,
        "split": s.split.value,
        "scenario_group": s.underlying_context_id,
    }


def _continuation_row(s: StimulusRecord, rec: dict[str, Any], scores: Sequence[Any]) -> dict[str, Any]:
    a1, a2 = scores
    contrast = float(a1.total_logprob - a2.total_logprob)
    return {
        **_row_base(s),
        "template_family_id": s.template_family_id,
        "persona_id": s.persona_id,
        "preference_explicit": bool(s.preference_explicit),
        "outcome_code": "K1" if contrast > 0 else "K2",
        "outcome_coding": OUTCOME_CODING_ID,
        "lik_alt1_total": float(a1.total_logprob),
        "lik_alt2_total": float(a2.total_logprob),
        "lik_alt1_per_token": float(a1.per_token_logprob),
        "lik_alt2_per_token": float(a2.per_token_logprob),
        "lik_alt1_token_logprobs": [float(x) for x in a1.token_logprobs],
        "lik_alt2_token_logprobs": [float(x) for x in a2.token_logprobs],
        "n_tokens_alt1": int(a1.n_tokens),
        "n_tokens_alt2": int(a2.n_tokens),
        "lik_contrast": contrast,
        "alt1_text": a1.scored_text,
        "alt2_text": a2.scored_text,
        "alternative_add_space": bool(rec["alternative_add_space"]),
        "rating_scale1": None,
        "rating_scale1_sd": None,
        "rater_panel_id": RATER_PANEL_PENDING,
        "rendered_token_hash": s.rendered_token_hash,
    }


def _plan_lane(cfg: StudyConfig, d: Path) -> dict[str, Any]:
    stimuli, rendering = _load_lane(d)
    derive = read_json(d / DERIVE_SUMMARY)
    primary = int(derive["primary_layer"])
    prim = derive["per_layer"][str(primary)]
    usable = [
        did
        for did in REGISTERED_DIRECTIONS
        if prim["directions"][did]["unit_ok"] and prim["directions"][did]["s"]
    ]
    skipped = [did for did in REGISTERED_DIRECTIONS if did not in usable]
    rand_ids = [rid for rid, info in prim["random_directions"].items() if info["s_used"]]
    eligible = [s for s in stimuli if s.split == Split.test and s.candidate_polarity == Polarity.neutral]
    max_n = _opt(cfg.interventions, "max_intervention_stimuli", None)
    max_n = int(max_n) if max_n is not None else None
    selected = select_intervention_stimuli(eligible, max_n, int(cfg.study.seed))
    alphas = [float(a) for a in cfg.interventions.alphas]
    done_cont = {r["stimulus_id"] for r in _read_partial_jsonl(d / CONTINUATIONS_PARTIAL)}
    vectors_sha = sha256_file(d / VECTORS_FILE)
    meta_path = d / INTERVENTIONS_PARTIAL_META
    stale = meta_path.exists() and read_json(meta_path).get("vectors_sha256") != vectors_sha
    done_int = (
        set()
        if stale
        else {
            (r["stimulus_id"], r["direction_id"], float(r["alpha"]))
            for r in _read_partial_jsonl(d / INTERVENTIONS_PARTIAL)
        }
    )
    direction_ids = usable + rand_ids
    n_rows_total = len(selected) * len(direction_ids) * len(alphas)
    n_rows_todo = sum(
        1
        for s in selected
        for did in direction_ids
        for a in alphas
        if (s.stimulus_id, did, a) not in done_int
    )
    planned = 2 * sum(1 for s in stimuli if s.stimulus_id not in done_cont) + len(selected) + 3 * n_rows_todo
    return {
        "stimuli": stimuli,
        "rendering": rendering,
        "derive": derive,
        "primary": primary,
        "prim": prim,
        "usable": usable,
        "skipped": skipped,
        "rand_ids": rand_ids,
        "direction_ids": direction_ids,
        "eligible": eligible,
        "selected": selected,
        "max_n": max_n,
        "alphas": alphas,
        "done_cont": done_cont,
        "done_int": done_int,
        "stale_partial": bool(stale),
        "vectors_sha256": vectors_sha,
        "n_rows_total": n_rows_total,
        "n_rows_todo": n_rows_todo,
        "planned_forward_passes": planned,
    }


def target_intervene(cfg: StudyConfig) -> dict[str, Any]:
    """Behavioural readouts for every stimulus, single-position steering on test-split neutral stimuli,
    and the final ``Bundle`` save."""
    _ensure_logging()
    t0 = time.perf_counter()
    origin = cfg.data.origin_required
    assert_production_origin(cfg, origin)
    registry = registry_status(cfg)
    spec = load_stimulus_spec(spec_path(cfg))
    lanes = _require_generated(
        cfg, STIMULI_FILE, RENDERING_FILE, ACTIVATIONS_FILE, VECTORS_FILE, DERIVE_SUMMARY
    )
    plans = {d: _plan_lane(cfg, d) for _, _, d in lanes}
    preflight = gpu_preflight(cfg, "intervene", sum(p["planned_forward_passes"] for p in plans.values()))
    adapter = load_adapter(cfg)
    provenance = adapter.provenance()
    if is_tiny(cfg):
        provenance["tiny_seed"] = int(_opt(cfg.target, "tiny_seed", 0))
    exploratory = bool(spec.exploratory or not registry["pilot_allowed"] or cfg.target.smoke_scope)
    smoke = bool(cfg.target.smoke_scope)
    pilot_eligible = bool(
        not exploratory and not smoke and registry["pilot_allowed"] and origin == Origin.real_target
    )
    out_lanes: list[dict[str, Any]] = []
    for bundle_id, cand, d in lanes:
        p = plans[d]
        stimuli: list[StimulusRecord] = p["stimuli"]
        rendering = p["rendering"]
        k = _check_layer_plan(stimuli, adapter)
        if k != p["primary"]:
            raise RuntimeError(
                f"{bundle_id}/{cand}: derive used layer {p['primary']} but the model's primary layer is {k}"
            )
        prompts = _verify_rendering(spec, adapter.tokenizer, stimuli, rendering, cfg.target.revision)
        with np.load(d / VECTORS_FILE, allow_pickle=False) as z:
            units = {did: np.asarray(z[f"u__{did}__L{k}"], dtype=np.float64) for did in p["direction_ids"]}
        s_map = {did: float(p["prim"]["directions"][did]["s"]) for did in p["usable"]}
        matched = {did: did for did in p["usable"]}
        for rid in p["rand_ids"]:
            info = p["prim"]["random_directions"][rid]
            s_map[rid] = float(info["s_used"])
            matched[rid] = info["matched_direction_id"]
        hook_site = adapter.hook_site(k)
        _reset_peak_memory(adapter)
        t_lane = time.perf_counter()
        n_forward = 0

        # -- behavioural rows for every stimulus (no steering) --------------------------------
        cont_rows = {r["stimulus_id"]: r for r in _read_partial_jsonl(d / CONTINUATIONS_PARTIAL)}
        cont_rows = {
            sid: r
            for sid, r in cont_rows.items()
            if sid in rendering and r.get("rendered_token_hash") == rendering[sid]["rendered_token_hash"]
        }
        n_cont_new = 0
        for s in stimuli:
            if s.stimulus_id in cont_rows:
                continue
            rec = rendering[s.stimulus_id]
            alts = [rec["alternatives"]["alt1"], rec["alternatives"]["alt2"]]
            scores = adapter.continuation_logprobs(
                prompts[s.stimulus_id], alts, add_space=bool(rec["alternative_add_space"])
            )
            n_forward += 2
            row = _continuation_row(s, rec, scores)
            append_jsonl(d / CONTINUATIONS_PARTIAL, row)
            cont_rows[s.stimulus_id] = row
            n_cont_new += 1
        adapter.assert_no_hooks()
        LOGGER.info(
            "intervene %s/%s: continuations %d new, %d resumed",
            bundle_id,
            cand,
            n_cont_new,
            len(stimuli) - n_cont_new,
        )

        # -- interventions on the selected test-split neutral stimuli ---------------------------
        if p["stale_partial"] and (d / INTERVENTIONS_PARTIAL).exists():
            stale = d / f"interventions.partial.stale-{int(time.time())}.jsonl"
            os.replace(d / INTERVENTIONS_PARTIAL, stale)
            LOGGER.warning(
                "intervene %s/%s: vectors changed since the partial rows were written; restarting (%s)",
                bundle_id,
                cand,
                stale,
            )
        write_json(
            d / INTERVENTIONS_PARTIAL_META,
            {"vectors_sha256": p["vectors_sha256"], "layer": k, "alphas": p["alphas"]},
        )
        done = set(p["done_int"])
        n_int_new = 0
        selected: list[StimulusRecord] = p["selected"]
        for si, s in enumerate(selected):
            prefix = prompts[s.stimulus_id]
            rec = rendering[s.stimulus_id]
            alts = [rec["alternatives"]["alt1"], rec["alternatives"]["alt2"]]
            add_space = bool(rec["alternative_add_space"])
            h_base = None
            for did in p["direction_ids"]:
                u = units[did]
                s_val = s_map[did]
                for alpha in p["alphas"]:
                    key = (s.stimulus_id, did, alpha)
                    if key in done:
                        continue
                    if h_base is None:
                        h_base = adapter.capture([prefix], [k]).activations[k][0].astype(np.float64)
                        n_forward += 1
                    steering = make_steering(k, u, s_val, alpha, did)
                    with adapter.steer(k, steering.displacement) as handle:
                        h_steered = adapter.capture([prefix], [k]).activations[k][0].astype(np.float64)
                        applied_capture = float(handle.last_applied_displacement_norm)
                        scores = adapter.continuation_logprobs(prefix, alts, add_space=add_space)
                        applied_cont = float(handle.last_applied_displacement_norm)
                        n_calls = int(handle.n_calls)
                    n_forward += 3
                    a1, a2 = scores
                    contrast = float(a1.total_logprob - a2.total_logprob)
                    row = {
                        **_row_base(s),
                        "direction_id": did,
                        "direction_kind": "random" if did.startswith(RANDOM_PREFIX) else "registered",
                        "matched_direction_id": matched[did],
                        "alpha": float(alpha),
                        "s": s_val,
                        "layer": int(k),
                        "hook_site": hook_site,
                        "requested_norm": float(steering.intended_norm),
                        "applied_norm": applied_capture,
                        "applied_norm_continuation": applied_cont,
                        "n_steer_hook_calls": n_calls,
                        "projection_delta": float((h_steered - h_base) @ u),
                        "projection_delta_expected": float(alpha * s_val),
                        "outcome_code": "K1" if contrast > 0 else "K2",
                        "outcome_coding": OUTCOME_CODING_ID,
                        "lik_contrast": contrast,
                        "lik_alt1_total": float(a1.total_logprob),
                        "lik_alt2_total": float(a2.total_logprob),
                        "n_tokens_alt1": int(a1.n_tokens),
                        "n_tokens_alt2": int(a2.n_tokens),
                        "rating_scale1": None,
                        "rater_panel_id": RATER_PANEL_PENDING,
                        "rendered_token_hash": s.rendered_token_hash,
                    }
                    append_jsonl(d / INTERVENTIONS_PARTIAL, row)
                    done.add(key)
                    n_int_new += 1
            adapter.assert_no_hooks()
            LOGGER.info(
                "intervene %s/%s: stimulus %d/%d done (%d rows so far, %.0fs)",
                bundle_id,
                cand,
                si + 1,
                len(selected),
                len(done),
                time.perf_counter() - t_lane,
            )
        wall = time.perf_counter() - t_lane

        # -- assemble and save the bundle --------------------------------------------------------
        acts, _ = _load_activations(d, stimuli)
        continuations = [cont_rows[s.stimulus_id] for s in stimuli]
        wanted = {
            (s.stimulus_id, did, a) for s in selected for did in p["direction_ids"] for a in p["alphas"]
        }
        interventions = sorted(
            (
                r
                for r in _read_partial_jsonl(d / INTERVENTIONS_PARTIAL)
                if (r["stimulus_id"], r["direction_id"], float(r["alpha"])) in wanted
            ),
            key=lambda r: (r["stimulus_id"], r["direction_id"], float(r["alpha"])),
        )
        if len(interventions) != len(wanted):
            raise RuntimeError(
                f"{bundle_id}/{cand}: expected {len(wanted)} intervention rows, have {len(interventions)}"
            )
        lane_hashes = _file_hashes(
            d,
            [
                STIMULI_FILE,
                RENDERING_FILE,
                ACTIVATIONS_FILE,
                VECTORS_FILE,
                CONTINUATIONS_PARTIAL,
                INTERVENTIONS_PARTIAL,
            ],
        )
        meta: dict[str, Any] = {
            "origin": origin.value,
            "primary_layer": int(k),
            "hook_site": hook_site,
            "secondary_layers": [int(x) for x in sorted(acts) if int(x) != k],
            "capture_position": CAPTURE_POSITION,
            "use_cache": False,
            "target_model_id": adapter.model_id,
            "target_model_revision": f"{adapter.model_id}@{adapter.revision or 'unpinned'}",
            "target_revision": adapter.revision,
            "target_revision_pinned": adapter.revision is not None,
            "target_dtype": adapter.dtype_name,
            "target_backend": cfg.target.backend,
            "alphas": p["alphas"],
            "n_random_directions": len(p["rand_ids"]),
            "random_directions_per_contrast": int(cfg.interventions.random_directions_per_contrast),
            "random_direction_matching": {
                rid: {"matched_direction_id": matched[rid], "s_used": s_map[rid]} for rid in p["rand_ids"]
            },
            "direction_ids": p["usable"],
            "directions_skipped_near_zero": p["skipped"],
            "projection_scales": {did: s_map[did] for did in p["usable"]},
            "projection_scale_definition": "population std of h.u over valid construction stimuli at the primary layer",
            "strength_units": cfg.interventions.strength_units,
            "position_schedule": cfg.interventions.position_schedule,
            "steering_position": "last_prefix_token",
            "exploratory": exploratory,
            "smoke_scope": smoke,
            "pilot_eligible": pilot_eligible,
            "pilot_allowed_by_registry": registry["pilot_allowed"],
            "pilot_block_reason": None
            if pilot_eligible
            else registry["reason"]
            if not registry["pilot_allowed"]
            else "exploratory/smoke bundle",
            "mode_definition_version": spec.mode_definition_version,
            "mode_registry_version": registry["registry_version"],
            "mode_registry_status": registry["status"],
            "stimulus_spec_id": spec.spec_id,
            "stimulus_spec_sha256": spec.sha256,
            "outcome_coding": OUTCOME_CODING_ID,
            "outcome_coding_note": OUTCOME_CODING_NOTE,
            "rating_status": RATER_PANEL_PENDING,
            "rater_panel_id": RATER_PANEL_PENDING,
            "n_intervention_stimuli": len(selected),
            "intervention_stimulus_rule": "test split, neutral polarity, mode-balanced seeded subsample",
            "max_intervention_stimuli": p["max_n"],
            "n_eligible_intervention_stimuli": len(p["eligible"]),
            "fit_provenance": p["derive"]["fit_provenance"],
            "identity_residual": p["prim"]["identity"],
            "provenance": provenance,
            "hashes": {
                "spec": spec.sha256,
                "config": cfg.config_hash(),
                "registry": registry["sha256"],
                **lane_hashes,
            },
            "seeds": {"study_seed": cfg.study.seed, "split_seed": split_seed_for(cfg, cand)},
            "npbench_version": __version__,
            "created_at": utc_now_iso(),
        }
        bundle = Bundle(
            bundle_id=bundle_id,
            candidate_id=cand,
            origin=origin,
            meta=meta,
            stimuli=stimuli,
            activations={int(x): acts[x] for x in acts},
            continuations=continuations,
            interventions=interventions,
        )
        out_dir = bundle_out_dir(cfg, bundle_id, cand)
        bundle_hashes = bundle.save(out_dir)
        summary = {
            "kind": "target_intervene",
            "created_at": utc_now_iso(),
            "npbench_version": __version__,
            "config_hash": cfg.config_hash(),
            "bundle_id": bundle_id,
            "candidate_id": cand,
            "origin": origin.value,
            "dir": str(d),
            "bundle_dir": str(out_dir),
            "bundle_hashes": bundle_hashes,
            "n_stimuli": len(stimuli),
            "n_continuation_rows": len(continuations),
            "n_continuation_rows_new": n_cont_new,
            "n_intervention_stimuli": len(selected),
            "intervention_stimulus_ids": [s.stimulus_id for s in selected],
            "n_intervention_rows": len(interventions),
            "n_intervention_rows_new": n_int_new,
            "n_intervention_rows_resumed": len(interventions) - n_int_new,
            "direction_ids": p["direction_ids"],
            "directions_skipped_near_zero": p["skipped"],
            "alphas": p["alphas"],
            "n_forward_passes": n_forward,
            "planned_forward_passes": p["planned_forward_passes"],
            "throughput": {
                "wall_seconds": wall,
                "forward_passes_per_second": (n_forward / wall) if wall > 0 and n_forward else None,
            },
            "peak_memory": _peak_memory(adapter),
            "exploratory": exploratory,
            "pilot_eligible": pilot_eligible,
            "preflight": preflight,
        }
        write_json(d / INTERVENE_SUMMARY, summary)
        LOGGER.info(
            "intervene %s/%s: %d rows (%d new); bundle -> %s",
            bundle_id,
            cand,
            len(interventions),
            n_int_new,
            out_dir,
        )
        out_lanes.append(summary)
    return {
        "kind": "target_intervene",
        "created_at": utc_now_iso(),
        "wall_seconds": time.perf_counter() - t0,
        "config_hash": cfg.config_hash(),
        "origin": origin.value,
        "target": _target_info(cfg),
        "provenance": provenance,
        "registry": registry,
        "exploratory": exploratory,
        "pilot_eligible": pilot_eligible,
        "preflight": preflight,
        "lanes": out_lanes,
    }
