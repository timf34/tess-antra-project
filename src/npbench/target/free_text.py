"""Resumable free continuations. Only the final prompt token is steered.

Production uses EasySteer/vLLM; the HF backend is restricted to tiny CPU fixtures.
Identical zero-strength controls share a generated sample and a rating, not fake replicates.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..bundles import Bundle
from ..config import load_config
from ..reference.build import bundles_dir, iter_bundle_dirs
from ..util import append_jsonl, read_jsonl, sha256_file, sha256_obj, write_json
from .adapter import make_steering
from .pipeline import VECTORS_FILE, load_adapter, rendered_from_record, target_root


def generation_plan(bundle, rendering, config_hash, seed):
    requests = {}
    links = []
    for kind, rows in [("continuations", bundle.continuations), ("interventions", bundle.interventions)]:
        for i, row in enumerate(rows):
            alpha = float(row.get("alpha", 0))
            direction = row.get("direction_id") if alpha else None
            sid = row["stimulus_id"]
            key = sha256_obj([config_hash, rendering[sid]["rendered_token_hash"], direction, alpha, seed])
            requests[key] = {
                "generation_id": key,
                "stimulus_id": sid,
                "direction_id": direction,
                "alpha": alpha,
                "s": float(row.get("s", 0)) if alpha else 0,
                "layer": int(row.get("layer", bundle.primary_layer())),
                "seed": seed,
            }
            links.append({"kind": kind, "row": i, "generation_id": key})
    return requests, links


def generate(cfg, backend="easysteer", engine=None):
    settings = (cfg.model_extra or {}).get("free_text", {})
    max_tokens = int(settings.get("max_tokens", 128))
    seed = int(settings.get("seed", cfg.study.seed))
    if backend == "tiny" and cfg.target.backend != "tiny":
        raise ValueError("HF free generation is restricted to the tiny engineering fixture")
    adapter = load_adapter(cfg) if backend == "tiny" else None
    if backend == "easysteer":
        from vllm import LLM, SamplingParams
        from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
        from vllm.steer_vectors.payloads import DirectionVector

        engine = engine or LLM(
            model=cfg.target.model_id,
            revision=cfg.target.revision,
            tokenizer_revision=cfg.target.tokenizer_revision,
            dtype="bfloat16",
            enable_steer_vector=True,
            steer_algorithms=["direct"],
            enable_prefix_caching=False,
            enforce_eager=True,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )
    summaries = []
    for bid, candidate, bdir in iter_bundle_dirs(bundles_dir(cfg, None)):
        b = Bundle.load(bdir)
        lane = target_root(cfg) / bid / candidate
        rendering = {r["stimulus_id"]: r for r in read_jsonl(lane / "rendering.jsonl")}
        identity = sha256_obj([cfg.config_hash(), sha256_file(lane / VECTORS_FILE), backend])
        requests, links = generation_plan(b, rendering, identity, seed)
        out = lane / "free_text.jsonl"
        done = {r["generation_id"]: r for r in read_jsonl(out)} if out.exists() else {}
        if set(done) - set(requests):
            raise ValueError("Free-text resume identity mismatch")
        pending = [r for key, r in requests.items() if key not in done]
        groups = {}
        for req in pending:
            group = (req["direction_id"], req["alpha"], req["s"], req["layer"])
            groups.setdefault(group, []).append(req)
        with np.load(lane / VECTORS_FILE, allow_pickle=False) as vecs:
            for members in groups.values():
                for start in range(0, len(members), 16):
                    batch = members[start : start + 16]
                    req = batch[0]
                    rendered_batch = [rendered_from_record(rendering[r["stimulus_id"]]) for r in batch]
                    u = (
                        None
                        if req["direction_id"] is None
                        else vecs[f"u__{req['direction_id']}__L{req['layer']}"]
                    )
                    if backend == "tiny":
                        steer = (
                            None
                            if u is None
                            else make_steering(req["layer"], u, req["s"], req["alpha"], req["direction_id"])
                        )
                        # Individual CPU generation makes fixture results independent of padding batches.
                        samples = [
                            (
                                adapter.generate([rendered], max_tokens, steering=steer, seed=seed)[0],
                                None,
                                "tiny_engineering",
                            )
                            for rendered in rendered_batch
                        ]
                    else:
                        # EasySteer's resolver accepts None (unsteered) or a SteeringSpec; False is not iterable.
                        steer = (
                            None
                            if u is None
                            else SteeringSpec(
                                vectors=[
                                    VectorSpec(
                                        data=DirectionVector({req["layer"]: np.asarray(u, dtype=np.float32)}),
                                        scale=req["alpha"] * req["s"],
                                        layers=[req["layer"]],
                                        normalize=False,
                                        apply=ApplySpec(prompt_positions=[-1]),
                                    )
                                ]
                            )
                        )
                        outputs = engine.generate(
                            [{"prompt_token_ids": r.input_ids} for r in rendered_batch],
                            sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens, seed=seed),
                            steering=steer,
                            use_tqdm=False,
                        )
                        samples = [
                            (o.outputs[0].text, list(o.outputs[0].token_ids), o.outputs[0].finish_reason)
                            for o in outputs
                        ]
                    if len(samples) != len(batch):
                        raise RuntimeError("Generation returned the wrong number of continuations")
                    for request, rendered, (text, tokens, finish) in zip(
                        batch, rendered_batch, samples, strict=True
                    ):
                        row = {
                            **request,
                            "text": text,
                            "generated_token_ids": tokens,
                            "finish_reason": finish,
                            "rendered_token_hash": rendered.rendered_token_hash,
                            "backend": backend,
                            "model_revision": cfg.target.revision,
                            "config_hash": identity,
                            "max_tokens": max_tokens,
                            "temperature": 0,
                            "position_schedule": "final_prompt_token_only",
                            "origin": b.origin.value,
                        }
                        append_jsonl(out, row)
                        done[request["generation_id"]] = row
                    print(f"free text {len(done)}/{len(requests)}", flush=True)
        for link in links:
            row = getattr(b, link["kind"])[link["row"]]
            sample = done[link["generation_id"]]
            row.update(
                generation_id=link["generation_id"],
                generated_text=sample["text"],
                generation_finish_reason=sample["finish_reason"],
            )
        b.meta["free_text"] = {
            "backend": backend,
            "identity": identity,
            "samples": len(done),
            "zero_controls_share_samples": True,
            "rating_status": "pending",
        }
        b.save(bdir)
        summary = {
            "bundle": bid,
            "candidate": candidate,
            "samples": len(done),
            "rows": len(links),
            "identity": identity,
        }
        write_json(lane / "free_text_summary.json", summary)
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=["tiny", "easysteer"], default="easysteer")
    args = parser.parse_args()
    print(json.dumps(generate(load_config(args.config), args.backend), indent=2))
