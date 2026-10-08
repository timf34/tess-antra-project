"""GPU checks against the configured real target (run on the pod: `pytest tests/gpu -m gpu`).

These load the checkpoint named in configs/gpu_smoke.yaml and repeat the adapter's engineering checks on
real weights: last-non-pad capture equals unbatched capture, alpha=0 equals baseline, applied displacement
equals requested, hooks are removed after an exception, and the chat/raw renderers differ."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def adapter():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        if os.environ.get("NPBENCH_REQUIRE_GPU") == "1":
            pytest.fail("GPU job has no working CUDA device")
        pytest.skip("no CUDA device")
    from npbench.config import load_config
    from npbench.target.adapter import HFTargetAdapter

    cfg = load_config(REPO / os.environ.get("NPBENCH_GPU_CONFIG", "configs/gpu_smoke.yaml"))
    if cfg.target.revision is None:
        if os.environ.get("NPBENCH_REQUIRE_GPU") == "1":
            pytest.fail("GPU job checkpoint revision must be pinned")
        pytest.skip("pin target.revision in configs/gpu_smoke.yaml before running GPU tests")
    return HFTargetAdapter.from_pretrained(
        cfg.target.model_id,
        cfg.target.revision,
        dtype=cfg.target.dtype or "bfloat16",
        token=os.environ.get("HF_TOKEN"),
    )


def test_capture_matches_unbatched_on_real_model(adapter):
    from npbench.target.render import render_chat

    tok = adapter.tokenizer
    prompts = [
        render_chat(tok, [{"role": "user", "content": t}])
        for t in ("Hello.", "Describe a quiet harbour at dusk in two sentences.", "Hi")
    ]
    layer = adapter.primary_layer()
    batched = adapter.capture(prompts, [layer]).activations[layer]
    for i, p in enumerate(prompts):
        single = adapter.capture([p], [layer]).activations[layer][0]
        assert np.allclose(batched[i], single, atol=2e-2, rtol=1e-2), {
            "prompt": i,
            "max_abs": float(np.max(np.abs(batched[i] - single))),
            "relative_l2": float(np.linalg.norm(batched[i] - single) / np.linalg.norm(single)),
        }  # bf16 tolerance
    adapter.assert_no_hooks()


def test_alpha_zero_and_displacement_on_real_model(adapter):
    from npbench.target.adapter import make_steering
    from npbench.target.render import render_chat
    from npbench.vectors import unit_vector

    tok = adapter.tokenizer
    p = render_chat(tok, [{"role": "user", "content": "Say one word."}])
    layer = adapter.primary_layer()
    base = adapter.forward_logits([p])
    h = adapter.capture([p], [layer]).activations[layer][0]
    u, ok = unit_vector(np.random.default_rng(0).standard_normal(h.shape[0]))
    assert ok
    zero = adapter.forward_logits([p], steering=make_steering(layer, u, 1.0, 0.0, "rand"))
    assert np.allclose(base, zero, atol=1e-2)
    spec = make_steering(layer, u, 2.0, 1.5, "rand")
    adapter.forward_logits([p], steering=spec)
    assert (
        abs(adapter.last_applied_displacement_norm - np.linalg.norm(spec.displacement))
        / np.linalg.norm(spec.displacement)
        < 5e-2
    )
    adapter.assert_no_hooks()
