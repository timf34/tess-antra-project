"""The offline tiny model factory: deterministic, special tokens are single ids, template renders exactly."""

from __future__ import annotations

import pytest
import torch
from jinja2 import TemplateError

from npbench.target import (
    CHAT_SPECIAL_TOKENS,
    DEFAULT_VOCAB,
    SPECIAL_TOKENS,
    TINY_MODEL_NAME,
    build_tiny_llama,
    build_tiny_tokenizer,
)

pytestmark = pytest.mark.target


def test_build_is_deterministic_in_seed():
    model_a, _ = build_tiny_llama(seed=0, n_layers=2)
    model_b, _ = build_tiny_llama(seed=0, n_layers=2)
    model_c, _ = build_tiny_llama(seed=1, n_layers=2)
    sa, sb, sc = model_a.state_dict(), model_b.state_dict(), model_c.state_dict()
    assert sa.keys() == sb.keys() == sc.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def test_build_does_not_disturb_global_rng():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    build_tiny_llama(seed=7, n_layers=1)
    assert torch.equal(torch.rand(3), expected)


def test_model_shape_and_mode(tiny):
    model, tokenizer = tiny
    assert not model.training
    assert next(model.parameters()).dtype == torch.float32
    assert model.config.num_hidden_layers == 4
    assert model.config.hidden_size == 32
    assert model.config.vocab_size == len(tokenizer) == len(DEFAULT_VOCAB) + len(SPECIAL_TOKENS)
    assert model.config.tie_word_embeddings is False
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    assert len(model.model.layers) == 4


def test_every_special_token_is_one_id(tiny):
    _, tokenizer = tiny
    ids = []
    for tok in SPECIAL_TOKENS:
        enc = tokenizer(tok, add_special_tokens=False)["input_ids"]
        assert enc == [tokenizer.convert_tokens_to_ids(tok)], tok
        assert tokenizer.convert_ids_to_tokens(enc[0]) == tok
        ids.append(enc[0])
    assert len(set(ids)) == len(SPECIAL_TOKENS)
    assert set(ids) <= set(tokenizer.all_special_ids)
    assert tokenizer.bos_token == "<s>" and tokenizer.eos_token == "</s>"
    assert tokenizer.pad_token == "<pad>" and tokenizer.unk_token == "<unk>"
    assert set(CHAT_SPECIAL_TOKENS) <= set(tokenizer.all_special_tokens)
    assert tokenizer.padding_side == "left"
    assert tokenizer.name_or_path == TINY_MODEL_NAME


def test_chat_template_renders_exactly(tiny):
    _, tokenizer = tiny
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "the cat"},
        {"role": "user", "content": "sat on"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert (
        text
        == "<|user|> hello world <|end|> <|assistant|> the cat <|end|> <|user|> sat on <|end|> <|assistant|> "
    )
    text_no_gen = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert (
        text_no_gen == "<|user|> hello world <|end|> <|assistant|> the cat <|end|> <|user|> sat on <|end|> "
    )
    with pytest.raises(TemplateError, match="supports only"):
        tokenizer.apply_chat_template([{"role": "system", "content": "x"}], tokenize=False)


def test_words_tokenize_without_unk_and_unknown_words_hit_unk(tiny):
    _, tokenizer = tiny
    ids = tokenizer(" ".join(DEFAULT_VOCAB), add_special_tokens=False)["input_ids"]
    assert tokenizer.unk_token_id not in ids
    assert len(ids) == len(DEFAULT_VOCAB)
    assert tokenizer("zzzz", add_special_tokens=False)["input_ids"] == [tokenizer.unk_token_id]


def test_custom_vocab_validation():
    with pytest.raises(ValueError, match="duplicate"):
        build_tiny_tokenizer(["a", "a"])
    with pytest.raises(ValueError, match="clash"):
        build_tiny_tokenizer(["a", "<s>"])
    with pytest.raises(ValueError, match="single"):
        build_tiny_tokenizer(["don't"])
    tok = build_tiny_tokenizer(["x", "y"])
    assert len(tok) == 2 + len(SPECIAL_TOKENS)
    with pytest.raises(ValueError, match="divisible"):
        build_tiny_llama(d_model=30)
