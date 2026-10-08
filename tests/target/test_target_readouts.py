"""Teacher-forced continuation scores and forced-choice code probabilities agree with hand calculations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from npbench.target import ActionCodeResult, ContinuationScore, make_steering
from npbench.vectors import unit_vector

pytestmark = pytest.mark.target

ALTERNATIVES = ["hello", "the cat sat", "yes ."]


def _hand_total(model, prefix_ids, cont_ids, hook=None, layer=None):
    ids = prefix_ids + cont_ids
    handle = model.model.layers[layer].register_forward_hook(hook) if hook is not None else None
    try:
        with torch.no_grad():
            logits = (
                model(
                    input_ids=torch.tensor([ids]),
                    attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    use_cache=False,
                )
                .logits[0]
                .double()
            )
    finally:
        if handle is not None:
            handle.remove()
    lp = torch.log_softmax(logits, dim=-1)
    n_prefix = len(prefix_ids)
    per_token = [float(lp[n_prefix - 1 + j, cont_ids[j]]) for j in range(len(cont_ids))]
    return sum(per_token), per_token


def test_continuation_logprobs_match_hand_calculation(adapter, tiny, prompts):
    model, tokenizer = tiny
    prefix = prompts[0]
    scores = adapter.continuation_logprobs(prefix, ALTERNATIVES)
    assert [s.n_tokens for s in scores] == [1, 3, 2]
    for alt, sc in zip(ALTERNATIVES, scores, strict=True):
        assert isinstance(sc, ContinuationScore)
        cont_ids = tokenizer(alt, add_special_tokens=False)["input_ids"]
        assert sc.alternative == alt and sc.scored_text == alt
        assert sc.token_ids == cont_ids and sc.n_tokens == len(cont_ids)
        total, per_token = _hand_total(model, prefix.input_ids, cont_ids)
        assert sc.total_logprob == pytest.approx(total, abs=1e-5)
        assert sc.per_token_logprob == pytest.approx(total / len(cont_ids), abs=1e-5)
        assert sc.per_token_logprob == pytest.approx(sc.total_logprob / sc.n_tokens, abs=1e-12)
        np.testing.assert_allclose(sc.token_logprobs, per_token, atol=1e-5, rtol=0)
        assert sc.total_logprob <= 0.0
    adapter.assert_no_hooks()


def test_continuation_logprobs_first_token_matches_forward_logits(adapter, prompts):
    prefix = prompts[2]
    logits = adapter.forward_logits([prefix])[0].astype(np.float64)
    lp = logits - np.log(np.sum(np.exp(logits - logits.max()))) - logits.max()
    (score,) = adapter.continuation_logprobs(prefix, ["hello"])
    assert score.total_logprob == pytest.approx(float(lp[score.token_ids[0]]), abs=1e-5)


def test_continuation_logprobs_add_space_and_validation(adapter, prompts):
    scores = adapter.continuation_logprobs(prompts[0], ["hello"], add_space=True)
    assert scores[0].scored_text == " hello" and scores[0].n_tokens == 1
    with pytest.raises(ValueError, match="zero tokens"):
        adapter.continuation_logprobs(prompts[0], [""])


def test_continuation_logprobs_steering_applies_at_last_prefix_position(adapter, tiny, prompts):
    model, tokenizer = tiny
    prefix = prompts[1]
    u, ok = unit_vector(np.random.default_rng(3).standard_normal(model.config.hidden_size))
    assert ok
    spec = make_steering(adapter.primary_layer(), u, 0.5, 2.0, "d")
    disp = torch.tensor(spec.displacement, dtype=torch.float32)
    n_prefix = len(prefix.input_ids)

    def manual_hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden.clone()
        steered[:, n_prefix - 1] = hidden[:, n_prefix - 1] + disp
        return (steered, *output[1:]) if isinstance(output, tuple) else steered

    scores = adapter.continuation_logprobs(prefix, ALTERNATIVES, steering=spec)
    assert adapter.last_steer_hook_calls == len(ALTERNATIVES)
    assert abs(adapter.last_applied_displacement_norm - spec.intended_norm) <= 1e-4 * spec.intended_norm
    adapter.assert_no_hooks()
    plain = adapter.continuation_logprobs(prefix, ALTERNATIVES)
    for alt, sc, pl in zip(ALTERNATIVES, scores, plain, strict=True):
        cont_ids = tokenizer(alt, add_special_tokens=False)["input_ids"]
        total, _ = _hand_total(model, prefix.input_ids, cont_ids, hook=manual_hook, layer=spec.layer)
        assert sc.total_logprob == pytest.approx(total, abs=1e-5)
        assert sc.n_tokens == pl.n_tokens and sc.token_ids == pl.token_ids


def test_action_code_probs_round_trip(adapter, tiny, prompts):
    _, tokenizer = tiny
    prefix = prompts[0]
    codes = {"1": "inspect", "2": "guess_0", "3": "guess_1"}
    res = adapter.action_code_probs(prefix, codes)
    assert isinstance(res, ActionCodeResult)
    assert res.codes == codes
    assert set(res.token_ids) == set(codes)
    assert {res.token_to_code[tid]: tid for tid in res.token_to_code} == res.token_ids
    for code, tid in res.token_ids.items():
        assert tokenizer(code, add_special_tokens=False)["input_ids"] == [tid]
        assert tokenizer.convert_ids_to_tokens(tid) == code
        assert tokenizer.decode([tid]) == code
    assert res.capture_position == prefix.token_count - 1
    assert 0.0 <= res.valid_mass <= 1.0 + 1e-12
    assert res.valid_mass == pytest.approx(sum(res.probs.values()), abs=1e-12)
    assert res.restricted_probs is None
    logits = adapter.forward_logits([prefix])[0].astype(np.float64)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    for code, tid in res.token_ids.items():
        assert res.probs[code] == pytest.approx(float(p[tid]), abs=1e-9)

    two = adapter.action_code_probs(prefix, {"1": "inspect", "2": "guess"})
    assert two.restricted_probs is not None
    assert sum(two.restricted_probs.values()) == pytest.approx(1.0, abs=1e-12)
    assert two.restricted_probs["1"] == pytest.approx(two.probs["1"] / two.valid_mass)
    assert two.probs["1"] == pytest.approx(res.probs["1"])
    adapter.assert_no_hooks()


def test_action_code_probs_rejects_bad_codes(adapter, prompts):
    with pytest.raises(ValueError, match="exactly one token"):
        adapter.action_code_probs(prompts[0], {"1 2": "x"})
    with pytest.raises(ValueError, match="unknown token"):
        adapter.action_code_probs(prompts[0], {"zzzz": "x"})
    with pytest.raises(ValueError, match="no action codes"):
        adapter.action_code_probs(prompts[0], {})
    adapter.assert_no_hooks()
