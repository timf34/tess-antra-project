"""Open-weight target adapter: pure PyTorch + transformers, explicit hook sites, no KV cache.

Hook-site convention
    ``hook_site(layer) == "decoder_block_output[layer]"`` names the forward output of decoder block
    ``layer`` (zero-based): ``output[0]`` when the block returns a tuple, the bare tensor otherwise
    (transformers >= 5). That is the residual stream *after* block ``layer``, i.e.
    ``hidden_states[layer + 1]`` in ``output_hidden_states`` terms. It is never the post-final-norm
    tensor that feeds ``lm_head``.

Capture-position convention
    Every readout and every steering injection happens at the registered final non-padding prefix
    token of each sequence (left padding; the last index where ``attention_mask == 1``). Nothing
    generated is ever read back into an activation, and every forward runs with ``use_cache=False`` so a
    block sees the whole sequence exactly once per call.

Steering composition
    ``steer`` registers one forward hook on the chosen block. The hook needs the per-sequence prefix
    positions, which the adapter registers whenever a forward goes through it (``capture``,
    ``forward_logits``, ``continuation_logprobs``, ``action_code_probs``, ``generate``); a direct call to
    ``model(...)`` inside a ``steer`` block must pass ``attention_mask`` to ``steer`` instead. Hooks are
    always removed in ``finally``; ``assert_no_hooks`` verifies the clean state.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import transformers

from npbench.vectors import intervention_displacement

from .render import RenderedPrompt, tokenizer_revision_string

HOOK_SITE_TEMPLATE = "decoder_block_output[{layer}]"
HOOK_SITE_DEFINITION = (
    "forward output[0] (or bare tensor) of decoder block `layer`: the residual stream after the block, "
    "equal to hidden_states[layer+1]; not the post-final-norm tensor"
)
CAPTURE_POSITION = "registered_final_nonpadding_prefix_token"
STEER_POSITION_LAST_PREFIX = "last_prefix_token"

# Attribute paths (relative to the HF model object) tried in order when locating the decoder blocks.
DECODER_BLOCK_PATHS: tuple[str, ...] = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "layers",
)
_FORBIDDEN_PATH_WORDS = ("vision", "vision_tower")
# Hooks transformers itself installs (lazily, on the first ``output_hidden_states=True`` forward) and
# never removes. They are inert unless a thread-local collector is active and do not alter outputs.
_INERT_LIBRARY_HOOK_MODULES = ("transformers.utils.output_capturing",)


# ----------------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------------


@dataclass
class SteeringSpec:
    """A registered intervention: add ``displacement`` (= alpha * s * u) at the hook site of ``layer``."""

    layer: int
    displacement: np.ndarray
    alpha: float
    s: float
    direction_id: str

    @property
    def intended_norm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.displacement, dtype=np.float64)))


def make_steering(layer: int, u: np.ndarray, s: float, alpha: float, direction_id: str) -> SteeringSpec:
    """Build a ``SteeringSpec`` from a unit direction ``u``, projection scale ``s`` and ``alpha``."""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1:
        raise ValueError("u must be a 1-D direction")
    return SteeringSpec(
        layer=int(layer),
        displacement=intervention_displacement(alpha, s, u),
        alpha=float(alpha),
        s=float(s),
        direction_id=str(direction_id),
    )


@dataclass
class CaptureResult:
    activations: dict[int, np.ndarray]  # layer -> (batch, d_model) float32, at the capture position
    capture_positions: list[int]  # per sequence, the token index actually read
    hook_sites: dict[int, str]
    layers: list[int]
    token_counts: list[int]
    use_cache: bool = False


@dataclass
class ContinuationScore:
    alternative: str  # as given by the caller
    scored_text: str  # what was tokenized (``alternative`` with the optional leading space)
    token_ids: list[int]
    n_tokens: int
    total_logprob: float
    per_token_logprob: float  # total / n_tokens
    token_logprobs: list[float]


@dataclass
class ActionCodeResult:
    codes: dict[str, str]  # code -> action
    token_ids: dict[str, int]  # code -> single token id
    token_to_code: dict[int, str]
    probs: dict[str, float]  # code -> next-token probability
    valid_mass: float  # sum of the code probabilities
    restricted_probs: dict[str, float] | None  # p / valid_mass, only when exactly two codes
    capture_position: int


@dataclass
class SteerHandle:
    """Yielded by ``HFTargetAdapter.steer``; updated by the hook on every forward it touches."""

    layer: int
    hook_site: str
    position: str
    intended_displacement_norm: float
    last_applied_displacement_norm: float | None = None  # mean over the batch of ||steered - original||
    applied_displacement_norms: list[float] = field(default_factory=list)  # per sequence, last call
    n_calls: int = 0


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------


def last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence index of the last token with ``attention_mask == 1`` (general, not left-pad only)."""
    mask = torch.as_tensor(attention_mask).long()
    if mask.ndim != 2:
        raise ValueError("attention_mask must be (batch, seq)")
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("a sequence has no non-padding tokens")
    idx = torch.arange(mask.shape[1], device=mask.device)
    return (mask * idx).argmax(dim=1)


def _block_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _with_block_output(output: Any, hidden: torch.Tensor) -> Any:
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def _is_inert_library_hook(fn: Any) -> bool:
    return getattr(fn, "__module__", "") in _INERT_LIBRARY_HOOK_MODULES


# ----------------------------------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------------------------------


class HFTargetAdapter:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        model_id: str,
        revision: str | None,
        dtype_name: str,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        actual_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        if dtype_name != actual_dtype:
            raise ValueError(f"dtype_name={dtype_name!r} but the model parameters are {actual_dtype!r}")
        self.dtype_name = dtype_name
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has neither pad nor eos token; cannot batch with padding")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.transformers_version = transformers.__version__
        self.torch_version = torch.__version__

        self._blocks_path: str | None = None
        self._blocks = self._resolve_decoder_blocks()
        self._steer_positions: torch.Tensor | None = None
        self._hook_handles: list[Any] = []

        self.last_intended_displacement_norm: float | None = None
        self.last_applied_displacement_norm: float | None = None
        self.last_applied_displacement_norms: list[float] = []
        self.last_steer_hook_calls: int = 0

    # -- construction ------------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str | None,
        dtype: str = "bfloat16",
        device_map: str | Mapping[str, Any] | None = "auto",
        attn_implementation: str = "sdpa",
        token: str | None = None,
    ) -> HFTargetAdapter:
        """Load a Hub checkpoint pinned to ``revision`` (left padding, pad := eos if missing, eval mode)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            token=token,
        )
        model.eval()
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        param = next(model.parameters())
        return cls(
            model,
            tokenizer,
            model_id=model_id,
            revision=revision,
            dtype_name=str(param.dtype).removeprefix("torch."),
            device=str(param.device),
        )

    # -- decoder blocks and layer conventions ------------------------------------------------

    def _resolve_decoder_blocks(self) -> torch.nn.ModuleList:
        tried: list[str] = []
        for path in DECODER_BLOCK_PATHS:
            obj: Any = self.model
            found = True
            for part in path.split("."):
                if any(w in part.lower() for w in _FORBIDDEN_PATH_WORDS):
                    raise ValueError(f"refusing decoder-block path {path!r}: it names a vision module")
                if not hasattr(obj, part):
                    found = False
                    break
                obj = getattr(obj, part)
            if found and isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
                vision_blocks = [type(b).__name__ for b in obj if "vision" in type(b).__name__.lower()]
                if vision_blocks:
                    raise ValueError(
                        f"refusing decoder-block path {path!r}: it resolves to vision blocks {vision_blocks}"
                    )
                self._blocks_path = path
                return obj
            tried.append(path)
        raise AttributeError(
            f"could not locate the decoder blocks of {type(self.model).__name__}; tried paths {tried}"
        )

    def decoder_blocks(self) -> torch.nn.ModuleList:
        """The language-model decoder blocks (never a vision tower)."""
        return self._blocks

    @property
    def n_blocks(self) -> int:
        return len(self._blocks)

    def primary_layer(self) -> int:
        return self.n_blocks // 2

    def quarter_layer(self) -> int:
        return self.n_blocks // 4

    def three_quarter_layer(self) -> int:
        return (3 * self.n_blocks) // 4

    def _check_layer(self, layer: int) -> int:
        layer = int(layer)
        if not 0 <= layer < self.n_blocks:
            raise IndexError(f"layer {layer} out of range for {self.n_blocks} decoder blocks")
        return layer

    def hook_site(self, layer: int) -> str:
        """``"decoder_block_output[layer]"``: the block's forward output ``output[0]`` (residual stream
        after block ``layer`` == ``hidden_states[layer+1]``), NOT the post-final-norm tensor."""
        return HOOK_SITE_TEMPLATE.format(layer=self._check_layer(layer))

    # -- batching and forward ----------------------------------------------------------------

    def _collate(self, rendered: Sequence[RenderedPrompt]) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad the registered ``input_ids`` (never re-tokenized) into a batch."""
        if not rendered:
            raise ValueError("no rendered prompts")
        pad_id = int(self.tokenizer.pad_token_id)
        lengths = []
        for r in rendered:
            if len(r.input_ids) != r.token_count or r.token_count == 0:
                raise ValueError("rendered prompt has inconsistent or empty input_ids")
            lengths.append(r.token_count)
        n, max_len = len(rendered), max(lengths)
        input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        for i, r in enumerate(rendered):
            k = r.token_count
            input_ids[i, max_len - k :] = torch.tensor(r.input_ids, dtype=torch.long)
            attention_mask[i, max_len - k :] = 1
        return input_ids.to(self.device), attention_mask.to(self.device)

    @contextlib.contextmanager
    def _registered_positions(self, positions: torch.Tensor) -> Iterator[None]:
        previous = self._steer_positions
        self._steer_positions = positions
        try:
            yield
        finally:
            self._steer_positions = previous

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        steer_positions: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> Any:
        """One full-sequence forward (``use_cache=False``) with the prefix positions registered for
        any active steering hook. ``steer_positions`` defaults to the last non-padding index of
        ``attention_mask``; ``continuation_logprobs`` passes the prefix positions explicitly."""
        if steer_positions is None:
            steer_positions = last_nonpad_positions(attention_mask)
        with self._registered_positions(steer_positions), torch.no_grad():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )

    @staticmethod
    def _logits_at(logits: torch.Tensor, seq_len: int, positions: torch.Tensor) -> torch.Tensor:
        """Index ``logits`` at absolute sequence ``positions`` whether the model returned every position
        or only the last ``logits.shape[1]`` of them (``logits_to_keep``)."""
        kept = logits.shape[1]
        offset = seq_len - kept
        rel = positions.to(logits.device) - offset
        if offset < 0 or bool((rel < 0).any()) or bool((rel >= kept).any()):
            raise RuntimeError(
                f"model returned logits for {kept} of {seq_len} positions; cannot index {positions}"
            )
        rows = torch.arange(logits.shape[0], device=logits.device)
        return logits[rows, rel]

    def _maybe_steer(self, steering: SteeringSpec | None) -> contextlib.AbstractContextManager[Any]:
        if steering is None:
            return contextlib.nullcontext()
        return self.steer(steering.layer, steering.displacement)

    # -- capture -----------------------------------------------------------------------------

    def _make_capture_hook(self, layer: int, positions: torch.Tensor, store: dict[int, torch.Tensor]):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
            hidden = _block_output(output)
            if layer in store:
                raise RuntimeError(
                    f"decoder block {layer} ran twice in one capture forward (is a cache active?)"
                )
            pos = positions.to(hidden.device)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            store[layer] = hidden[rows, pos].detach().to(device="cpu", dtype=torch.float32)

        return hook

    def capture(self, rendered: Sequence[RenderedPrompt], layers: Sequence[int]) -> CaptureResult:
        """Block-output activations at the last non-padding prefix token of each sequence.

        Hooks are registered on the requested blocks for exactly one ``use_cache=False`` forward and
        removed in ``finally``. ``generate`` is never called."""
        layers = [self._check_layer(x) for x in layers]
        if not layers:
            raise ValueError("no layers requested")
        input_ids, attention_mask = self._collate(rendered)
        positions = last_nonpad_positions(attention_mask)
        blocks = self.decoder_blocks()
        store: dict[int, torch.Tensor] = {}
        handles = []
        try:
            for layer in dict.fromkeys(layers):
                handles.append(
                    blocks[layer].register_forward_hook(self._make_capture_hook(layer, positions, store))
                )
            self._forward(input_ids, attention_mask, steer_positions=positions, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        missing = [layer for layer in layers if layer not in store]
        if missing:
            raise RuntimeError(f"capture hooks never fired for layers {missing}")
        return CaptureResult(
            activations={layer: store[layer].numpy() for layer in layers},
            capture_positions=[int(p) for p in positions.tolist()],
            hook_sites={layer: self.hook_site(layer) for layer in layers},
            layers=list(layers),
            token_counts=[r.token_count for r in rendered],
            use_cache=False,
        )

    def assert_no_hooks(self, *, strict: bool = False) -> None:
        """Verify that no forward hook remains on any decoder block.

        By default transformers' own inert ``output_capturing_hook`` entries (installed lazily by the
        library on the first ``output_hidden_states=True`` forward and never removed) are ignored; with
        ``strict=True`` the check is literally ``len(block._forward_hooks) == 0`` for every block."""
        leaked: list[str] = []
        for i, block in enumerate(self.decoder_blocks()):
            for hook_id, fn in block._forward_hooks.items():
                if strict or not _is_inert_library_hook(fn):
                    leaked.append(f"block {i}: hook {hook_id} {getattr(fn, '__qualname__', repr(fn))}")
        if self._hook_handles:
            leaked.append(f"adapter still tracks {len(self._hook_handles)} steering hook handle(s)")
        if leaked:
            raise AssertionError("forward hooks remain on decoder blocks:\n  " + "\n  ".join(leaked))

    # -- steering ----------------------------------------------------------------------------

    @contextlib.contextmanager
    def steer(
        self,
        layer: int,
        displacement: np.ndarray,
        position: str = STEER_POSITION_LAST_PREFIX,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> Iterator[SteerHandle]:
        """Add ``displacement`` to the output of decoder block ``layer`` at the last non-padding prefix
        position of every sequence, for every forward run inside the block.

        Positions come from the forward registered by the adapter (``capture``, ``forward_logits``,
        ``continuation_logprobs``, ``generate``) or from ``attention_mask`` when given. The hook records
        the intended norm and the actually applied norm ``||steered - original||`` at that position. The
        hook is removed in ``finally`` even if the body raises."""
        if position != STEER_POSITION_LAST_PREFIX:
            raise ValueError(
                f"unsupported steering position {position!r}; only {STEER_POSITION_LAST_PREFIX!r}"
            )
        layer = self._check_layer(layer)
        disp = np.asarray(displacement, dtype=np.float64)
        if disp.ndim != 1:
            raise ValueError("displacement must be a 1-D vector of size d_model")
        fixed_positions = (
            last_nonpad_positions(torch.as_tensor(attention_mask)) if attention_mask is not None else None
        )
        disp_t = torch.as_tensor(disp)
        handle = SteerHandle(
            layer=layer,
            hook_site=self.hook_site(layer),
            position=position,
            intended_displacement_norm=float(np.linalg.norm(disp)),
        )
        self.last_intended_displacement_norm = handle.intended_displacement_norm
        self.last_applied_displacement_norm = None
        self.last_applied_displacement_norms = []
        self.last_steer_hook_calls = 0

        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> Any:
            hidden = _block_output(output)
            positions = fixed_positions if fixed_positions is not None else self._steer_positions
            if positions is None:
                raise RuntimeError(
                    "steering hook fired without registered prefix positions: run the model through the "
                    "adapter (capture/forward_logits/continuation_logprobs/generate) or pass attention_mask "
                    "to steer()"
                )
            positions = positions.to(hidden.device)
            if positions.shape[0] != hidden.shape[0]:
                raise RuntimeError(
                    f"steering positions cover {positions.shape[0]} sequences but the batch has {hidden.shape[0]}"
                )
            if int(positions.max()) >= hidden.shape[1]:
                raise RuntimeError(
                    "hidden states are shorter than the registered prefix position; steering requires "
                    "full-sequence forwards (use_cache=False)"
                )
            if disp_t.numel() != hidden.shape[-1]:
                raise ValueError(
                    f"displacement has {disp_t.numel()} dims but the block output has {hidden.shape[-1]}"
                )
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            add = disp_t.to(device=hidden.device, dtype=hidden.dtype)
            original = hidden[rows, positions].detach().clone()
            steered = hidden.clone()
            steered[rows, positions] = original + add
            applied = (steered[rows, positions] - original).detach().float().norm(dim=-1).cpu().tolist()
            handle.n_calls += 1
            handle.applied_displacement_norms = applied
            handle.last_applied_displacement_norm = float(np.mean(applied))
            self.last_steer_hook_calls = handle.n_calls
            self.last_applied_displacement_norms = applied
            self.last_applied_displacement_norm = handle.last_applied_displacement_norm
            return _with_block_output(output, steered)

        h = self.decoder_blocks()[layer].register_forward_hook(hook)
        self._hook_handles.append(h)
        try:
            yield handle
        finally:
            h.remove()
            self._hook_handles.remove(h)

    # -- readouts ----------------------------------------------------------------------------

    def _next_token_logits(
        self, rendered: Sequence[RenderedPrompt], steering: SteeringSpec | None
    ) -> tuple[np.ndarray, list[int]]:
        input_ids, attention_mask = self._collate(rendered)
        positions = last_nonpad_positions(attention_mask)
        seq_len = input_ids.shape[1]
        keep = 1 if bool((positions == seq_len - 1).all()) else 0
        with self._maybe_steer(steering):
            out = self._forward(input_ids, attention_mask, steer_positions=positions, logits_to_keep=keep)
        logits = self._logits_at(out.logits, seq_len, positions)
        return logits.detach().float().cpu().numpy(), [int(p) for p in positions.tolist()]

    def forward_logits(
        self, rendered: Sequence[RenderedPrompt], steering: SteeringSpec | None = None
    ) -> np.ndarray:
        """Next-token logits (batch, vocab) at the last prefix position of each sequence."""
        return self._next_token_logits(rendered, steering)[0]

    def continuation_logprobs(
        self,
        prefix: RenderedPrompt,
        alternatives: Sequence[str],
        *,
        add_space: bool = False,
        steering: SteeringSpec | None = None,
    ) -> list[ContinuationScore]:
        """Teacher-forced log-likelihood of each alternative after ``prefix`` (one forward each, no cache).

        Steering, when given, is registered once for the whole call and applied only at the registered
        last-prefix position (the position that predicts the first continuation token) of every forward."""
        prefix_ids = [int(i) for i in prefix.input_ids]
        n_prefix = len(prefix_ids)
        if n_prefix == 0:
            raise ValueError("empty prefix")
        tokenized: list[tuple[str, str, list[int]]] = []
        for alt in alternatives:
            scored_text = f" {alt}" if add_space else alt
            cont_ids = [int(i) for i in self.tokenizer(scored_text, add_special_tokens=False)["input_ids"]]
            if not cont_ids:
                raise ValueError(f"alternative {alt!r} tokenizes to zero tokens")
            tokenized.append((alt, scored_text, cont_ids))
        prefix_positions = torch.tensor([n_prefix - 1], dtype=torch.long)
        out_scores: list[ContinuationScore] = []
        with self._maybe_steer(steering):
            for alt, scored_text, cont_ids in tokenized:
                n = len(cont_ids)
                ids = prefix_ids + cont_ids
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                attention_mask = torch.ones_like(input_ids)
                out = self._forward(
                    input_ids, attention_mask, steer_positions=prefix_positions, logits_to_keep=n + 1
                )
                wanted = torch.arange(n_prefix - 1, n_prefix - 1 + n)
                logits = self._logits_at(out.logits.expand(n, -1, -1), len(ids), wanted)  # (n, vocab)
                logprobs = torch.log_softmax(logits.detach().double(), dim=-1)
                tok_lp = logprobs[torch.arange(n), torch.tensor(cont_ids, device=logprobs.device)]
                total = float(tok_lp.sum())
                out_scores.append(
                    ContinuationScore(
                        alternative=alt,
                        scored_text=scored_text,
                        token_ids=cont_ids,
                        n_tokens=n,
                        total_logprob=total,
                        per_token_logprob=total / n,
                        token_logprobs=[float(x) for x in tok_lp.cpu().tolist()],
                    )
                )
        return out_scores

    def action_code_probs(
        self,
        prefix: RenderedPrompt,
        codes: Mapping[str, str],
        *,
        add_space: bool = False,
        steering: SteeringSpec | None = None,
    ) -> ActionCodeResult:
        """Next-token probability of each forced-choice code (each must be exactly one token)."""
        if not codes:
            raise ValueError("no action codes")
        token_ids: dict[str, int] = {}
        unk = self.tokenizer.unk_token_id
        for code in codes:
            text = f" {code}" if add_space else code
            ids = [int(i) for i in self.tokenizer(text, add_special_tokens=False)["input_ids"]]
            if len(ids) != 1:
                raise ValueError(
                    f"action code {code!r} (scored as {text!r}) tokenizes to {len(ids)} tokens "
                    f"{self.tokenizer.convert_ids_to_tokens(ids)}; forced-choice codes must be exactly one token"
                )
            if unk is not None and ids[0] == unk:
                raise ValueError(
                    f"action code {code!r} maps to the unknown token; it is not in the vocabulary"
                )
            token_ids[code] = ids[0]
        if len(set(token_ids.values())) != len(token_ids):
            raise ValueError(f"action codes share token ids: {token_ids}")
        logits, positions = self._next_token_logits([prefix], steering)
        probs_all = torch.softmax(torch.as_tensor(logits[0], dtype=torch.float64), dim=-1)
        probs = {code: float(probs_all[tid]) for code, tid in token_ids.items()}
        valid_mass = float(sum(probs.values()))
        restricted = None
        if len(codes) == 2 and valid_mass > 0:
            restricted = {code: p / valid_mass for code, p in probs.items()}
        return ActionCodeResult(
            codes=dict(codes),
            token_ids=token_ids,
            token_to_code={tid: code for code, tid in token_ids.items()},
            probs=probs,
            valid_mass=valid_mass,
            restricted_probs=restricted,
            capture_position=positions[0],
        )

    # -- generation --------------------------------------------------------------------------

    def generate(
        self,
        rendered: Sequence[RenderedPrompt],
        max_new_tokens: int,
        steering: SteeringSpec | None = None,
        do_sample: bool = False,
        seed: int | None = None,
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode continuations with ``model.generate(use_cache=False)``; the steering hook (if any) is
        active for every full-sequence step and removed afterwards. ``seed`` reseeds torch for sampling."""
        input_ids, attention_mask = self._collate(rendered)
        positions = last_nonpad_positions(attention_mask)
        if seed is not None:
            torch.manual_seed(seed)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "use_cache": False,
            "pad_token_id": int(self.tokenizer.pad_token_id),
        }
        if not do_sample:
            gen_kwargs["num_beams"] = 1
        with self._maybe_steer(steering), self._registered_positions(positions), torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        new_tokens = out[:, input_ids.shape[1] :].cpu().tolist()
        return [self.tokenizer.decode(row, skip_special_tokens=skip_special_tokens) for row in new_tokens]

    # -- provenance --------------------------------------------------------------------------

    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision or "unpinned",
            "dtype": self.dtype_name,
            "model_class": type(self.model).__name__,
            "attn_implementation": getattr(getattr(self.model, "config", None), "_attn_implementation", None),
            "n_blocks": self.n_blocks,
            "decoder_blocks_path": self._blocks_path,
            "primary_layer": self.primary_layer(),
            "quarter_layer": self.quarter_layer(),
            "three_quarter_layer": self.three_quarter_layer(),
            "hook_site_template": HOOK_SITE_TEMPLATE,
            "hook_site_definition": HOOK_SITE_DEFINITION,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "device": str(self.device),
            "padding_side": self.tokenizer.padding_side,
            "tokenizer_revision": tokenizer_revision_string(self.tokenizer, self.revision),
            "capture_position": CAPTURE_POSITION,
            "use_cache": False,
        }
