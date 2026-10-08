"""Provenance carries everything a stimulus record needs to name the measurement site exactly."""

from __future__ import annotations

import inspect

import pytest
import torch
import transformers

from npbench.target import CAPTURE_POSITION, HOOK_SITE_TEMPLATE, TINY_MODEL_NAME, HFTargetAdapter

pytestmark = pytest.mark.target

REQUIRED_KEYS = {
    "model_id",
    "revision",
    "dtype",
    "n_blocks",
    "primary_layer",
    "quarter_layer",
    "three_quarter_layer",
    "hook_site_template",
    "transformers_version",
    "torch_version",
    "device",
    "padding_side",
    "capture_position",
    "use_cache",
}


def test_provenance_has_required_keys_and_values(adapter):
    prov = adapter.provenance()
    missing = REQUIRED_KEYS - set(prov)
    assert not missing, missing
    assert prov["model_id"] == TINY_MODEL_NAME
    assert prov["revision"] == "unpinned"
    assert prov["dtype"] == "float32"
    assert prov["n_blocks"] == 4
    assert (prov["quarter_layer"], prov["primary_layer"], prov["three_quarter_layer"]) == (1, 2, 3)
    assert prov["hook_site_template"] == HOOK_SITE_TEMPLATE == "decoder_block_output[{layer}]"
    assert prov["hook_site_template"].format(layer=2) == adapter.hook_site(2)
    assert "not the post-final-norm" in prov["hook_site_definition"]
    assert prov["transformers_version"] == transformers.__version__
    assert prov["torch_version"] == torch.__version__
    assert prov["device"] == "cpu"
    assert prov["padding_side"] == "left"
    assert prov["capture_position"] == CAPTURE_POSITION == "registered_final_nonpadding_prefix_token"
    assert prov["use_cache"] is False
    assert prov["decoder_blocks_path"] == "model.layers"
    assert prov["tokenizer_revision"] == "npbench-tiny-llama@unpinned"
    assert prov["model_class"] == "LlamaForCausalLM"


def test_provenance_pins_revision(tiny):
    model, tokenizer = tiny
    pinned = HFTargetAdapter(
        model, tokenizer, model_id="org/model", revision="deadbeef", dtype_name="float32"
    )
    prov = pinned.provenance()
    assert prov["revision"] == "deadbeef"
    assert prov["tokenizer_revision"] == "npbench-tiny-llama@deadbeef"


def test_from_pretrained_is_a_thin_pinned_wrapper():
    sig = inspect.signature(HFTargetAdapter.from_pretrained)
    assert list(sig.parameters)[:2] == ["model_id", "revision"]
    assert sig.parameters["dtype"].default == "bfloat16"
    assert sig.parameters["attn_implementation"].default == "sdpa"
    assert "token" in sig.parameters
