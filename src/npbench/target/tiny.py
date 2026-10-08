"""Offline test-model factory: a tiny randomly initialised Llama plus an in-memory word-level tokenizer.

Nothing here touches the network or the Hugging Face cache. The model is deterministic in ``seed``,
runs in float32 on CPU, and goes through the real ``transformers`` code paths (decoder blocks, SDPA
attention with left padding, chat templates, ``generate``), so the target-lane engineering tests can
check hook sites, padding, steering and readouts without downloading any weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

TINY_MODEL_NAME = "npbench-tiny-llama"

BOS, EOS, PAD, UNK = "<s>", "</s>", "<pad>", "<unk>"
USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL_TOKENS: tuple[str, ...] = (BOS, EOS, PAD, UNK, USER, ASSISTANT, END)
CHAT_SPECIAL_TOKENS: tuple[str, ...] = (USER, ASSISTANT, END)

# Small word list; every entry must be a single unit under ``pre_tokenizers.Whitespace`` (``\\w+`` or a
# run of punctuation), otherwise it could never be produced by tokenization.
DEFAULT_VOCAB: tuple[str, ...] = (
    "hello", "world", "the", "a", "an", "cat", "dog", "sat", "on", "mat", "ran", "runs", "is", "was",
    "are", "and", "or", "not", "yes", "no", "what", "why", "how", "who", "which", "do", "does", "did",
    "go", "went", "see", "saw", "think", "know", "tell", "ask", "answer", "question", "please", "thanks",
    "you", "i", "we", "they", "it", "this", "that", "one", "two", "three", "1", "2", "3", "A", "B",
    "inspect", "guess", "red", "blue", "green", "big", "small", "good", "bad", "happy", "sad", "feel",
    "fine", "okay", "sorry", "help", "me", "my", "your", "in", "to", "of", ",", ".", "?", "!",
)  # fmt: skip

# Renders ``<|user|> {content} <|end|> `` / ``<|assistant|> {content} <|end|> `` per message and appends
# ``<|assistant|> `` when ``add_generation_prompt`` is true. Unknown roles raise instead of vanishing.
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}<|user|> {{ m['content'] }} <|end|> "
    "{% elif m['role'] == 'assistant' %}<|assistant|> {{ m['content'] }} <|end|> "
    "{% else %}{{ raise_exception('tiny chat template supports only user/assistant roles, got: ' ~ m['role']) }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)


def build_tiny_tokenizer(vocab: Sequence[str] | None = None) -> PreTrainedTokenizerFast:
    """Word-level tokenizer over ``vocab`` (default :data:`DEFAULT_VOCAB`) plus the seven special tokens.

    Ids are assigned in order: special tokens first, then the words. Every special token tokenizes to
    exactly one id (verified before returning). Padding side is left and the chat template is set."""
    words = list(DEFAULT_VOCAB if vocab is None else vocab)
    if not words:
        raise ValueError("vocab must contain at least one word")
    if len(set(words)) != len(words):
        raise ValueError("vocab contains duplicate words")
    clash = sorted(set(words) & set(SPECIAL_TOKENS))
    if clash:
        raise ValueError(f"vocab words clash with special tokens: {clash}")
    pre = pre_tokenizers.Whitespace()
    for w in words:
        pieces = [p for p, _ in pre.pre_tokenize_str(w)]
        if pieces != [w]:
            raise ValueError(f"vocab word {w!r} is not a single Whitespace pre-tokenization unit: {pieces}")

    vocab_map = {t: i for i, t in enumerate([*SPECIAL_TOKENS, *words])}
    core = Tokenizer(models.WordLevel(vocab=vocab_map, unk_token=UNK))
    core.pre_tokenizer = pre
    core.add_special_tokens(list(SPECIAL_TOKENS))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        bos_token=BOS,
        eos_token=EOS,
        pad_token=PAD,
        unk_token=UNK,
        additional_special_tokens=list(CHAT_SPECIAL_TOKENS),
        name_or_path=TINY_MODEL_NAME,
        model_max_length=256,
    )
    tokenizer.padding_side = "left"
    tokenizer.chat_template = CHAT_TEMPLATE

    for t in SPECIAL_TOKENS:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if ids != [vocab_map[t]]:
            raise RuntimeError(f"special token {t!r} does not tokenize to its single id: {ids}")
    return tokenizer


def build_tiny_llama(
    seed: int = 0,
    n_layers: int = 4,
    d_model: int = 32,
    vocab: Sequence[str] | None = None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizerFast]:
    """Deterministic, randomly initialised ``LlamaForCausalLM`` (float32, eval mode) and its tokenizer.

    The global torch RNG is forked so building a model does not disturb the caller's random state."""
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if d_model % 4 != 0:
        raise ValueError("d_model must be divisible by the 4 attention heads")
    tokenizer = build_tiny_tokenizer(vocab)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=d_model,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="sdpa",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LlamaForCausalLM(config)
    model = model.to(torch.float32).eval()
    model.requires_grad_(False)
    return model, tokenizer
