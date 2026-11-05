"""
GRU backbone baseline. Required by plan section 4.5.

At runtime Mamba behaves like a strong RNN, so the plan requires a GRU baseline
to establish whether Mamba is actually needed rather than merely fashionable.
This is the control for that experiment, so it is sized to a comparable
parameter and multiply-add count: the comparison has to be about the
architecture, not about capacity.

Exposes exactly `MambaBackbone`'s interface -- `forward(x, state)`,
`step(x, state)`, `initial_state(batch)` -- so only the config changes between
runs and `MmmPolicy` needs no knowledge of which backbone it holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = ["GruState", "GruBackbone"]


@dataclass
class GruState:
    """The GRU's hidden state, wrapped to match `MambaState`'s interface."""

    hidden: torch.Tensor

    def detach(self) -> "GruState":
        return GruState(hidden=self.hidden.detach())

    def clone(self) -> "GruState":
        return GruState(hidden=self.hidden.clone())


class GruBackbone(nn.Module):
    """A stacked GRU with the same interface as `MambaBackbone`.

    Args:
        d_model: width, matching the Mamba backbone's `d_model`.
        n_layers: stacked GRU layers.
        hidden_size: defaults to `d_model`. A GRU layer has roughly
            `3 * (d_model * hidden + hidden^2)` parameters against a Mamba
            block's roughly `4 * d_model * d_inner`, so leaving both at 256
            puts the two within the same order and keeps the comparison fair
            without contriving the numbers.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 4, hidden_size: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.hidden_size = hidden_size or d_model

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=self.hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        # Projects back to d_model so the output layer is shared with the Mamba
        # path. Identity when the widths already agree, but kept as a Linear
        # unconditionally so the parameter count does not change meaning with
        # hidden_size.
        self.out_proj = nn.Linear(self.hidden_size, d_model)
        self.final_norm = nn.LayerNorm(d_model)

    @property
    def config(self):
        """Mirror `MambaConfig`'s attribute names, so `MmmPolicy.describe`
        reports something sensible for either backbone."""

        @dataclass
        class _Config:
            d_model: int
            n_layers: int
            d_state: int | None = None
            d_conv: int | None = None
            d_inner: int | None = None

        return _Config(
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_inner=self.hidden_size,
        )

    def initial_state(self, batch: int, device=None, dtype=torch.float32) -> GruState:
        return GruState(
            hidden=torch.zeros(
                self.n_layers, batch, self.hidden_size, device=device, dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor, state: GruState | None = None):
        """Parallel mode. `x` is `(batch, seq, d_model)`."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, seq, d_model); got {tuple(x.shape)}")

        hidden = state.hidden if state is not None else None
        out, new_hidden = self.gru(x, hidden)
        return self.final_norm(self.out_proj(out)), GruState(hidden=new_hidden)

    def step(self, x: torch.Tensor, state: GruState):
        """Recurrent mode: one frame. `x` is `(batch, d_model)`."""
        if x.dim() != 2:
            raise ValueError(f"expected (batch, d_model); got {tuple(x.shape)}")

        out, new_hidden = self.gru(x.unsqueeze(1), state.hidden)
        return self.final_norm(self.out_proj(out.squeeze(1))), GruState(hidden=new_hidden)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def multiply_adds_per_frame(self) -> int:
        """Rough multiply-add count for one frame, comparable to
        `MambaBackbone.multiply_adds_per_frame`.

        A GRU layer computes three gates, each with an input and a recurrent
        matrix multiply.
        """
        first = 3 * (self.d_model * self.hidden_size + self.hidden_size**2)
        later = 3 * (self.hidden_size * self.hidden_size + self.hidden_size**2)
        return first + (self.n_layers - 1) * later + self.hidden_size * self.d_model
