"""Steering injects exactly the requested displacement at the hook site and always cleans up.

No test here asserts a monotonic behavioural response; only the mechanics of the intervention."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from npbench.target import (
    TINY_MODEL_NAME,
    HFTargetAdapter,
    SteeringSpec,
    build_tiny_llama,
    make_steering,
)
from npbench.vectors import intervention_displacement, random_directions, unit_vector

pytestmark = pytest.mark.target

S = 0.5


@pytest.fixture(scope="module")
def direction(adapter):
    rng = np.random.default_rng(42)
    u, ok = unit_vector(rng.standard_normal(adapter.model.config.hidden_size))
    assert ok
    return u


def _spec(adapter, u, alpha, layer=None):
    layer = adapter.primary_layer() if layer is None else layer
    return make_steering(layer, u, S, alpha, "test_direction")


def test_make_steering_uses_intervention_displacement(adapter, direction):
    spec = _spec(adapter, direction, 1.5)
    assert isinstance(spec, SteeringSpec)
    np.testing.assert_array_equal(spec.displacement, intervention_displacement(1.5, S, direction))
    assert spec.intended_norm == pytest.approx(1.5 * S)
    assert (spec.layer, spec.alpha, spec.s, spec.direction_id) == (
        adapter.primary_layer(),
        1.5,
        S,
        "test_direction",
    )


def test_alpha_zero_matches_baseline(adapter, prompts, direction):
    base = adapter.forward_logits(prompts)
    zero = adapter.forward_logits(prompts, steering=_spec(adapter, direction, 0.0))
    np.testing.assert_allclose(zero, base, atol=1e-6, rtol=0)
    assert adapter.last_intended_displacement_norm == 0.0
    assert adapter.last_applied_displacement_norm == 0.0
    adapter.assert_no_hooks()


def test_applied_displacement_equals_requested(adapter, prompts, direction):
    spec = _spec(adapter, direction, 2.0)
    intended = float(np.linalg.norm(spec.displacement))
    assert intended == pytest.approx(1.0)
    with adapter.steer(spec.layer, spec.displacement) as handle:
        adapter.forward_logits(prompts)
    assert handle.hook_site == adapter.hook_site(spec.layer)
    assert handle.n_calls == 1
    assert handle.intended_displacement_norm == pytest.approx(intended)
    assert len(handle.applied_displacement_norms) == len(prompts)
    for applied in handle.applied_displacement_norms:
        assert abs(applied - intended) <= 1e-4 * intended
    assert abs(handle.last_applied_displacement_norm - intended) <= 1e-4 * intended
    assert adapter.last_intended_displacement_norm == pytest.approx(intended)
    assert abs(adapter.last_applied_displacement_norm - intended) <= 1e-4 * intended
    assert adapter.last_steer_hook_calls == 1
    adapter.assert_no_hooks()


def test_opposite_alphas_give_opposite_displacements_at_hook_site(adapter, prompts, direction):
    layer = adapter.primary_layer()
    nxt = layer + 1
    positions = adapter.capture(prompts, [layer]).capture_positions

    def next_block_input(alpha):
        seen = {}
        h = adapter.decoder_blocks()[nxt].register_forward_pre_hook(
            lambda module, args: seen.__setitem__("x", args[0].detach().clone().numpy())
        )
        try:
            if alpha is None:
                adapter.forward_logits(prompts)
            else:
                adapter.forward_logits(prompts, steering=_spec(adapter, direction, alpha))
        finally:
            h.remove()
        return seen["x"]

    h_base, h_plus, h_minus = next_block_input(None), next_block_input(1.5), next_block_input(-1.5)
    rows = np.arange(len(prompts))
    pos = np.array(positions)
    d_plus = (h_plus - h_base)[rows, pos]
    d_minus = (h_minus - h_base)[rows, pos]
    np.testing.assert_allclose(d_plus, -d_minus, atol=1e-5, rtol=0)
    expected = np.broadcast_to(intervention_displacement(1.5, S, direction), d_plus.shape)
    np.testing.assert_allclose(d_plus, expected, atol=1e-5, rtol=0)
    # Every other position of the block output is untouched (the hook edits only the prefix position).
    untouched = np.ones(h_base.shape[:2], dtype=bool)
    untouched[rows, pos] = False
    np.testing.assert_array_equal(h_plus[untouched], h_base[untouched])
    np.testing.assert_array_equal(h_minus[untouched], h_base[untouched])
    adapter.assert_no_hooks()


def test_random_directions_have_matched_injected_norms(adapter, prompts, direction):
    spec = _spec(adapter, direction, 2.0)
    target = float(np.linalg.norm(spec.displacement))
    rands = random_directions(adapter.model.config.hidden_size, 3, seed=1, norm=target)
    np.testing.assert_allclose(np.linalg.norm(rands, axis=1), target, rtol=1e-12)
    for i, r in enumerate(rands):
        control = SteeringSpec(layer=spec.layer, displacement=r, alpha=2.0, s=S, direction_id=f"random_{i}")
        adapter.forward_logits(prompts, steering=control)
        assert abs(adapter.last_applied_displacement_norm - target) <= 1e-4 * target
        assert abs(adapter.last_intended_displacement_norm - target) <= 1e-12
    adapter.assert_no_hooks()


def test_hook_removed_after_exception_inside_steer(adapter, prompts, direction):
    base = adapter.forward_logits(prompts)
    spec = _spec(adapter, direction, 3.0)
    with pytest.raises(RuntimeError, match="boom"), adapter.steer(spec.layer, spec.displacement):
        adapter.forward_logits(prompts)
        raise RuntimeError("boom")
    adapter.assert_no_hooks()
    np.testing.assert_array_equal(adapter.forward_logits(prompts), base)


def test_strict_no_hooks_on_fresh_model_and_leak_detection(direction):
    model, tokenizer = build_tiny_llama(seed=5)
    fresh = HFTargetAdapter(model, tokenizer, model_id=TINY_MODEL_NAME, revision=None, dtype_name="float32")
    from npbench.target import render_chat

    p = render_chat(tokenizer, [{"role": "user", "content": "hello world"}])
    base = fresh.forward_logits([p])
    spec = make_steering(fresh.primary_layer(), direction, S, 2.0, "d")
    with pytest.raises(RuntimeError, match="boom"), fresh.steer(spec.layer, spec.displacement):
        fresh.forward_logits([p])
        raise RuntimeError("boom")
    fresh.assert_no_hooks(strict=True)
    assert all(len(b._forward_hooks) == 0 for b in fresh.decoder_blocks())
    np.testing.assert_array_equal(fresh.forward_logits([p]), base)
    leak = fresh.decoder_blocks()[0].register_forward_hook(lambda m, a, o: None)
    try:
        with pytest.raises(AssertionError, match="hooks remain"):
            fresh.assert_no_hooks()
    finally:
        leak.remove()
    fresh.assert_no_hooks(strict=True)


def test_repeated_baseline_unaffected_by_steered_calls(adapter, prompts, direction):
    base = adapter.forward_logits(prompts)
    for alpha in (1.0, -1.0, 3.0, 0.25):
        adapter.forward_logits(prompts, steering=_spec(adapter, direction, alpha))
        np.testing.assert_array_equal(adapter.forward_logits(prompts), base)
    texts = adapter.generate(prompts, max_new_tokens=3, steering=_spec(adapter, direction, 2.0))
    assert len(texts) == len(prompts)
    assert adapter.last_steer_hook_calls == 3  # one full-sequence forward per generated token
    assert abs(adapter.last_applied_displacement_norm - 1.0) <= 1e-4
    adapter.assert_no_hooks()
    np.testing.assert_array_equal(adapter.forward_logits(prompts), base)
    np.testing.assert_array_equal(adapter.forward_logits(prompts), base)


def test_steer_requires_positions_or_mask_and_validates_inputs(adapter, prompts, direction):
    spec = _spec(adapter, direction, 1.0)
    ids = torch.tensor([prompts[0].input_ids])
    mask = torch.ones_like(ids)
    with adapter.steer(spec.layer, spec.displacement), pytest.raises(RuntimeError, match="positions"):
        adapter.model(input_ids=ids, attention_mask=mask, use_cache=False)
    adapter.assert_no_hooks()
    with adapter.steer(spec.layer, spec.displacement, attention_mask=mask) as handle:
        with torch.no_grad():
            adapter.model(input_ids=ids, attention_mask=mask, use_cache=False)
    assert handle.n_calls == 1
    assert abs(handle.last_applied_displacement_norm - spec.intended_norm) <= 1e-4 * spec.intended_norm
    with pytest.raises(ValueError, match="position"):
        with adapter.steer(spec.layer, spec.displacement, position="all_tokens"):
            pass
    with pytest.raises(IndexError):
        with adapter.steer(adapter.n_blocks, spec.displacement):
            pass
    with pytest.raises(ValueError, match="dims"), adapter.steer(spec.layer, np.ones(3)):
        adapter.forward_logits(prompts)
    adapter.assert_no_hooks()


def test_generate_is_deterministic_and_restores_state(adapter, prompts, direction):
    greedy_a = adapter.generate(prompts, max_new_tokens=3)
    greedy_b = adapter.generate(prompts, max_new_tokens=3)
    assert greedy_a == greedy_b
    sampled_a = adapter.generate(prompts, max_new_tokens=3, do_sample=True, seed=11)
    sampled_b = adapter.generate(prompts, max_new_tokens=3, do_sample=True, seed=11)
    assert sampled_a == sampled_b
    adapter.assert_no_hooks()
