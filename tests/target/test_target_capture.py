"""Capture reads the block output at the last non-padding prefix token, never a generated token."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from npbench.target import (
    CaptureResult,
    HFTargetAdapter,
    last_nonpad_positions,
    render_chat,
)

pytestmark = pytest.mark.target


def _block_out(output):
    return output[0] if isinstance(output, tuple) else output


def test_last_nonpad_positions_general():
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
    assert last_nonpad_positions(mask).tolist() == [3, 3, 1, 1]
    with pytest.raises(ValueError):
        last_nonpad_positions(torch.tensor([[0, 0]]))


def test_batched_capture_matches_unbatched_for_mixed_lengths(adapter, prompts):
    layers = [0, adapter.primary_layer(), adapter.n_blocks - 1]
    counts = [p.token_count for p in prompts]
    assert len(set(counts)) == 3
    batched = adapter.capture(prompts, layers)
    assert isinstance(batched, CaptureResult)
    assert batched.use_cache is False
    assert batched.layers == layers
    assert batched.token_counts == counts
    assert batched.hook_sites == {k: f"decoder_block_output[{k}]" for k in layers}
    # Left padding: the last mask==1 index is the final column for every row.
    max_len = max(counts)
    assert batched.capture_positions == [max_len - 1] * len(prompts)
    for k in layers:
        assert batched.activations[k].shape == (len(prompts), adapter.model.config.hidden_size)
        assert batched.activations[k].dtype == np.float32
    for i, p in enumerate(prompts):
        single = adapter.capture([p], layers)
        assert single.capture_positions == [p.token_count - 1]
        for k in layers:
            np.testing.assert_allclose(batched.activations[k][i], single.activations[k][0], atol=1e-5, rtol=0)
    adapter.assert_no_hooks()


def test_hook_site_is_block_output_not_post_norm(adapter, prompts):
    model = adapter.model
    p = prompts[1]
    ids = torch.tensor([p.input_ids])
    mask = torch.ones_like(ids)
    norm_io = {}

    def norm_hook(module, args, output):
        norm_io["in"] = args[0].detach().clone()
        norm_io["out"] = output.detach().clone()

    h = model.model.norm.register_forward_hook(norm_hook)
    try:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
    finally:
        h.remove()
    hs = out.hidden_states
    n = adapter.n_blocks
    assert len(hs) == n + 1
    pos = p.token_count - 1
    res = adapter.capture([p], list(range(n)))
    assert res.capture_positions == [pos]
    for k in range(n):
        got = res.activations[k][0]
        if k < n - 1:
            np.testing.assert_allclose(got, hs[k + 1][0, pos].numpy(), atol=1e-6, rtol=0)
        else:
            np.testing.assert_allclose(got, norm_io["in"][0, pos].numpy(), atol=1e-6, rtol=0)
        assert not np.allclose(got, hs[k][0, pos].numpy(), atol=1e-6)
    post_norm = norm_io["out"][0, pos].numpy()
    assert not np.allclose(res.activations[n - 1][0], post_norm, atol=1e-6)
    with torch.no_grad():
        logits_from_norm = model.lm_head(norm_io["out"])[0, pos]
    np.testing.assert_allclose(logits_from_norm.numpy(), out.logits[0, pos].numpy(), atol=1e-6, rtol=0)


def test_capture_never_generates_and_ignores_generated_text(adapter, prompts, message_sets, monkeypatch):
    p = prompts[0]
    layers = [adapter.primary_layer()]
    before = adapter.capture([p], layers)

    def boom(*args, **kwargs):
        raise AssertionError("capture must never call generate")

    with monkeypatch.context() as m:
        m.setattr(type(adapter.model), "generate", boom)
        during = adapter.capture([p], layers)
    np.testing.assert_array_equal(before.activations[layers[0]], during.activations[layers[0]])

    texts = adapter.generate([p], max_new_tokens=4)
    assert len(texts) == 1 and isinstance(texts[0], str)
    after = adapter.capture([p], layers)
    np.testing.assert_array_equal(before.activations[layers[0]], after.activations[layers[0]])
    assert before.capture_positions == [p.token_count - 1]

    # A rendering that extends the prefix with an assistant turn reads its own last token instead.
    extended = render_chat(
        adapter.tokenizer,
        [*message_sets[0], {"role": "assistant", "content": "hello world"}],
        add_generation_prompt=False,
    )
    assert extended.token_count > p.token_count
    ext = adapter.capture([extended], layers)
    assert ext.capture_positions == [extended.token_count - 1]
    assert not np.allclose(ext.activations[layers[0]], before.activations[layers[0]])
    adapter.assert_no_hooks()


def test_capture_matches_independent_hook(adapter, prompts):
    k = adapter.primary_layer()
    seen = {}

    def hook(module, args, output):
        seen["out"] = _block_out(output).detach().clone()

    h = adapter.decoder_blocks()[k].register_forward_hook(hook)
    try:
        res = adapter.capture(prompts, [k])
    finally:
        h.remove()
    rows = np.arange(len(prompts))
    pos = np.array(res.capture_positions)
    np.testing.assert_array_equal(res.activations[k], seen["out"][rows, pos].numpy())
    adapter.assert_no_hooks()


def test_capture_validation_and_layer_conventions(adapter, prompts):
    assert isinstance(adapter.decoder_blocks(), torch.nn.ModuleList)
    assert adapter.n_blocks == 4
    assert (adapter.quarter_layer(), adapter.primary_layer(), adapter.three_quarter_layer()) == (1, 2, 3)
    assert adapter.hook_site(2) == "decoder_block_output[2]"
    with pytest.raises(IndexError):
        adapter.hook_site(4)
    with pytest.raises(IndexError):
        adapter.capture(prompts, [4])
    with pytest.raises(ValueError):
        adapter.capture([], [0])
    with pytest.raises(ValueError):
        adapter.capture(prompts, [])
    adapter.assert_no_hooks()


def test_decoder_block_path_resolution_and_vision_refusal(tiny):
    _, tokenizer = tiny

    class VisionBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

    class Inner(torch.nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.layers = torch.nn.ModuleList(blocks)

    class Nested(torch.nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.language_model = Inner(blocks)

    class Bare(torch.nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.layers = torch.nn.ModuleList(blocks)

    nested = HFTargetAdapter(
        Nested([Block(), Block()]), tokenizer, model_id="x", revision=None, dtype_name="float32"
    )
    assert nested.n_blocks == 2 and nested.provenance()["decoder_blocks_path"] == "language_model.layers"
    bare = HFTargetAdapter(Bare([Block()]), tokenizer, model_id="x", revision=None, dtype_name="float32")
    assert bare.provenance()["decoder_blocks_path"] == "layers"
    with pytest.raises(ValueError, match="vision"):
        HFTargetAdapter(Bare([VisionBlock()]), tokenizer, model_id="x", revision=None, dtype_name="float32")
    with pytest.raises(AttributeError, match="decoder blocks"):
        HFTargetAdapter(Block(), tokenizer, model_id="x", revision=None, dtype_name="float32")
    with pytest.raises(ValueError, match="dtype"):
        HFTargetAdapter(Bare([Block()]), tokenizer, model_id="x", revision=None, dtype_name="bfloat16")


def test_left_padding_does_not_shift_position_ids(adapter, prompts):
    seen = []
    handle = adapter.model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["position_ids"].detach().cpu()), with_kwargs=True
    )
    try:
        adapter.capture(prompts, [0])
    finally:
        handle.remove()
    positions = seen[0]
    for row, prompt in enumerate(prompts):
        assert positions[row, -prompt.token_count :].tolist() == list(range(prompt.token_count))
