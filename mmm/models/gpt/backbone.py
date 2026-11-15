"""
GPT-style backbone candidate: a causal transformer with a KV cache, explored
as a possible alternative to `mmm.models.mamba.backbone.MambaBackbone` --
not because it is faster (at this parameter count, both should land in the
same microsecond range once exported, since the dominant runtime cost is
framework overhead, not architecture), but because attention over a bounded
window of recent frames may capture the joystick/candidate/history
relationship differently than Mamba's single recurrent state does. That is
the actual question this candidate exists to test, decided empirically
(vs-teacher loss, jitter, sliding -- see `scripts/eval_checkpoints.py`), not
assumed from either architecture's reputation.

Adapted from a char-level language model sketch: the attention block and the
causal masking idea carry over, but everything about tokens does not. There is
no vocabulary, no token embedding table and no `generate()` -- a "token" here
is one frame's already-continuous input-layer output (see
`mmm.models.io_layers`), the same `(batch, ..., d_model)` tensor
`MambaBackbone` consumes, not a discrete symbol to look up or sample.

Interface contract, identical to every other backbone so `MmmPolicy` and
`mmm.models.runtime.build_policy` need no knowledge of which one they hold:

    forward(x, state=None) -> (y, state)   # a whole sequence, training
    step(x, state)         -> (y, state)   # one frame, inference
    initial_state(batch)   -> state

`state` here is a `GptState`: one key/value cache per layer, holding at most
`window` past frames, plus a running position counter. Unlike Mamba's fixed
size state, this one grows with context until `window` is reached, then
slides -- the one real difference `step()`'s caller
(`mmm.models.runtime.ModelController`) does not need to know about, but this
module does.

Positions use rotary embeddings (RoPE) rather than a learned table indexed by
position-in-window: a learned table would need re-indexing every time the
window slides and the oldest cached frame is evicted, since what was "slot 5"
yesterday is "slot 4" today. RoPE encodes each frame's *absolute* stream
position once, when it enters the cache, and the attention score between any
two positions depends only on their difference -- so a sliding window needs
no cache surgery at all, only dropping the oldest entries.

`mmm.models.baselines.transformer` exists for a related but different
question -- whether attention's cost growing with history (plan section 4.5's
argument for an SSM) is visible in practice, not only in theory. It should
import `GptBackbone` from here rather than keep a second implementation; the
two questions share one answer.

No part of the C++ runtime engine depends on this: the engine reads an
exported weight blob and runs inference, with no attention, KV cache or
sampling in it. This backbone is a training-time and PyTorch-time-only
candidate, same as every other backbone, until one is chosen to export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GptConfig", "GptState", "GptBackbone"]


@dataclass
class GptConfig:
    """Backbone shape. Defaults sized to land near `MambaBackbone`'s default
    parameter count (plan section 4.5's reasoning for the GRU baseline
    applies here too: a capacity-matched comparison, not a generous one).
    """

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    window: int = 64
    mlp_ratio: int = 4
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )
        return self.d_model // self.n_heads


@dataclass
class GptState:
    """Per character runtime state: a KV cache per layer, plus where in the
    stream the next frame sits.

    Attributes:
        keys: list of `(batch, n_heads, t, head_dim)`, `t <= window`.
        values: same shape as `keys`.
        position: frames seen so far in this stream. Keeps counting past
            `window` even once the cache itself has started sliding -- RoPE
            needs each frame's true absolute position, not its slot in the
            cache, to keep relative angles (and so attention scores) correct.
    """

    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)
    position: int = 0

    @classmethod
    def empty(cls, config: GptConfig, batch: int, device=None, dtype=torch.float32) -> "GptState":
        shape = (batch, config.n_heads, 0, config.head_dim)
        return cls(
            keys=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)],
            values=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)],
            position=0,
        )

    def detach(self) -> "GptState":
        """Cut the graph while keeping the values, for carrying state across
        training windows without backpropagating through every earlier one."""
        return GptState(
            keys=[k.detach() for k in self.keys],
            values=[v.detach() for v in self.values],
            position=self.position,
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _rope_angles(positions: torch.Tensor, head_dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)`, each `(len(positions), head_dim)`, for `apply_rope`.

    Standard RoPE: `head_dim / 2` frequencies, each used twice (once per half
    of the rotated pair), so a dot product between two rotated vectors ends up
    depending only on the positions' difference -- the property that makes a
    sliding KV cache need no re-indexing when the oldest entry is evicted.
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    freqs = positions.to(dtype)[:, None] * inv_freq[None, :]
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x`: `(batch, n_heads, t, head_dim)`. `cos`/`sin`: `(t, head_dim)`."""
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and an explicit KV cache.

    One module handles both modes `MambaBlock` splits across `forward`/`step`:
    with `cache=None` it is a plain causal pass over a whole window (training);
    with a cache it attends the new frame(s) against the cache's past frames
    plus themselves, then returns the updated (and, if needed, slid) cache.
    """

    def __init__(self, config: GptConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, t, _ = x.shape
        return x.view(batch, t, self.config.n_heads, self.config.head_dim).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, position: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: `(batch, t, d_model)`, the new frame(s) only.
            keys, values: the incoming cache, `(batch, n_heads, t_cache, head_dim)`,
                `t_cache` possibly 0.
            position: the absolute stream position of `x`'s first frame.

        Returns:
            `(output (batch, t, d_model), new_keys, new_values)`, the cache
            already extended with this call's frames but NOT yet trimmed to
            `window` -- trimming is the caller's job, since only the caller
            (`GptBackbone`) knows whether this was a `forward` or a `step`
            call and what "oldest" should mean across both.
        """
        batch, t, _ = x.shape
        q, k, v = self.qkv(x).split(self.config.d_model, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        new_positions = torch.arange(position, position + t, device=x.device)
        cos, sin = _rope_angles(new_positions, self.config.head_dim, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        all_keys = torch.cat([keys, k], dim=2)
        all_values = torch.cat([values, v], dim=2)

        # Masked on absolute stream position, not on index within this call,
        # so one big forward() over many frames agrees exactly with stepping
        # through them one at a time: a query may only see a key at or before
        # its own position, and no further back than `window` -- the window
        # has to be enforced here, inside the attention a single call
        # computes, not only between calls by trimming the outgoing cache,
        # or a forward() spanning more than `window` frames would let an
        # early query see a late key's whole unwindowed history while step()
        # never could.
        t_cache = keys.shape[2]
        cache_positions = torch.arange(position - t_cache, position, device=x.device)
        key_positions = torch.cat([cache_positions, new_positions])
        causal = key_positions[None, :] <= new_positions[:, None]
        windowed = (new_positions[:, None] - key_positions[None, :]) < self.config.window
        mask = causal & windowed

        scores = q @ all_keys.transpose(-2, -1) * (self.config.head_dim**-0.5)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = weights @ all_values
        out = out.transpose(1, 2).reshape(batch, t, self.config.d_model)
        return self.resid_dropout(self.proj(out)), all_keys, all_values


class FeedForward(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        hidden = config.mlp_ratio * config.d_model
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: norm, attend, residual; norm, feed
    forward, residual -- the stabler ordering for training from scratch,
    and the same shape `MambaBackbone`'s own blocks use (`norm(x)` feeds the
    block, the block's output is added back as a residual)."""

    def __init__(self, config: GptConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffwd = FeedForward(config)

    def forward(
        self, x: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, position: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attended, new_keys, new_values = self.attn(self.ln1(x), keys, values, position)
        x = x + attended
        x = x + self.ffwd(self.ln2(x))
        return x, new_keys, new_values


class GptBackbone(nn.Module):
    """A stack of causal self-attention blocks with a sliding KV cache."""

    def __init__(self, config: GptConfig | None = None, **overrides):
        super().__init__()
        if config is None:
            config = GptConfig(**overrides)
        elif overrides:
            raise TypeError("pass either a GptConfig or keyword overrides, not both")

        self.config = config
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)

    def initial_state(self, batch: int, device=None, dtype=torch.float32) -> GptState:
        return GptState.empty(self.config, batch, device=device, dtype=dtype)

    def _trimmed(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.config.window
        if keys.shape[2] <= window:
            return keys, values
        return keys[:, :, -window:], values[:, :, -window:]

    def forward(self, x: torch.Tensor, state: GptState | None = None):
        """Parallel mode: a whole window at once, training.

        Args:
            x: `(batch, seq, d_model)`.
            state: optional incoming cache, which makes consecutive training
                windows equivalent to one long stream, the same role it plays
                for `MambaBackbone`.

        Returns:
            `(output, new_state)`.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (batch, seq, d_model); got {tuple(x.shape)}")

        batch = x.shape[0]
        if state is None:
            state = self.initial_state(batch, device=x.device, dtype=x.dtype)

        new_keys, new_values = [], []
        for block, keys, values in zip(self.blocks, state.keys, state.values):
            x, k, v = block(x, keys, values, state.position)
            k, v = self._trimmed(k, v)
            new_keys.append(k)
            new_values.append(v)

        new_state = GptState(keys=new_keys, values=new_values, position=state.position + x.shape[1])
        return self.final_norm(x), new_state

    def step(self, x: torch.Tensor, state: GptState):
        """Recurrent mode: one frame, inference.

        Attends only the new frame's query against the cached keys/values --
        the cache is never recomputed from scratch, which is the entire point
        of carrying one. Verified against `forward` in
        `tests/test_gpt_recurrent_equals_parallel.py`, the same check
        `tests/test_mamba_recurrent_equals_parallel.py` makes for Mamba.

        Args:
            x: `(batch, d_model)`.
            state: the carried `GptState`.

        Returns:
            `(output, new_state)`, output `(batch, d_model)`.
        """
        if x.dim() != 2:
            raise ValueError(f"expected (batch, d_model); got {tuple(x.shape)}")

        output, new_state = self.forward(x.unsqueeze(1), state)
        return output.squeeze(1), new_state

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
