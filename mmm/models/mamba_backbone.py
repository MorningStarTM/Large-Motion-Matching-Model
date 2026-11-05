"""
The Mamba backbone: 4 layers at width 256, in two equivalent modes.

Pure PyTorch, with no `mamba-ssm` dependency, so it runs on CPU and on the
project's 4 GB GPU without a CUDA build step.

Two modes, and they must agree:

- **Parallel** (`forward`) processes a whole sequence at once. Used for
  training.
- **Recurrent** (`step`) processes one frame, carrying a conv buffer and an SSM
  state. Used at runtime, and the form plan section 13.2 specifies for the C++
  engine.

`tests/test_mamba_recurrent_equals_parallel.py` enforces the equivalence.
Debugging that disagreement in C++ is far more expensive than in Python, and a
model that trains well in parallel mode but behaves differently in recurrent
mode would fail in the game for reasons invisible during training.

Provenance
----------
The parallel path, the `dt_proj` log-uniform initialization and the `A_log`
parameterisation follow `src/gpsm/ssm/motion_mamba.py` from the author's GPSM-1
project, which in turn follows Algorithm 2 of the Mamba paper (Gu and Dao,
2023). Two things are new here:

1. **`step`**, the incremental recurrence. GPSM-1's `SelectiveScan` loops over
   time but materialises `(B, T, d_inner, d_state)` up front and keeps no state
   between calls, so it cannot run one frame at a time. MMM needs that for the
   runtime loop and for the engine.
2. **The scan is written as an in-place state update** over a preallocated
   output list rather than a stack of per-timestep tensors, which keeps the
   memory flat in `d_state` and matches what the engine does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MambaConfig", "MambaState", "RMSNorm", "SelectiveScan", "MambaBlock", "MambaBackbone"]

# dt_proj bias initialization range, from the Mamba reference implementation.
_DT_MIN = 0.001
_DT_MAX = 0.1
_DT_SCALE = 1.0
_DT_INIT_FLOOR = 1e-4


@dataclass
class MambaConfig:
    """Backbone shape. The defaults are plan section 4.3's model.

    `d_inner` is derived rather than configured, so every layer reads one
    source of truth instead of recomputing `expand * d_model`.
    """

    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def dt_rank_resolved(self) -> int:
        if self.dt_rank is not None:
            return self.dt_rank
        return max(1, math.ceil(self.d_model / 16))

    def state_bytes(self) -> int:
        """Per character runtime state size, the number plan section 13.3 step
        3 quotes: a conv buffer plus an SSM state, per layer, float32."""
        conv = self.d_inner * (self.d_conv - 1)
        ssm = self.d_inner * self.d_state
        return self.n_layers * (conv + ssm) * 4


@dataclass
class MambaState:
    """Per character recurrent state: exactly what the C++ engine keeps.

    Attributes:
        conv: list of `(batch, d_inner, d_conv - 1)` -- each layer's last few
            inputs, feeding the causal convolution.
        ssm: list of `(batch, d_inner, d_state)` -- each layer's SSM state.
    """

    conv: list = field(default_factory=list)
    ssm: list = field(default_factory=list)

    @classmethod
    def zeros(cls, config: MambaConfig, batch: int, device=None, dtype=torch.float32):
        return cls(
            conv=[
                torch.zeros(batch, config.d_inner, config.d_conv - 1, device=device, dtype=dtype)
                for _ in range(config.n_layers)
            ],
            ssm=[
                torch.zeros(batch, config.d_inner, config.d_state, device=device, dtype=dtype)
                for _ in range(config.n_layers)
            ],
        )

    def detach(self) -> "MambaState":
        """Cut the graph while keeping the values, for carrying state across
        training windows without backpropagating through every earlier one."""
        return MambaState(
            conv=[c.detach() for c in self.conv],
            ssm=[s.detach() for s in self.ssm],
        )

    def clone(self) -> "MambaState":
        return MambaState(conv=[c.clone() for c in self.conv], ssm=[s.clone() for s in self.ssm])


class RMSNorm(nn.Module):
    """Root mean square layer normalization, as Mamba's architecture uses."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def _init_dt_proj(dt_proj: nn.Linear, dt_rank: int) -> None:
    """Targeted initialization for `dt_proj`.

    Left at a generic N(0, 0.02) with zero bias, every channel would start with
    nearly the same near-zero `delta`, so every SSM channel would update its
    state at the same very slow rate. Instead the bias is set so that
    `softplus(bias)` is spread log-uniformly over `[_DT_MIN, _DT_MAX]`: some
    channels react quickly, others slowly.

    `softplus` cannot be inverted algebraically into "assign this bias", so the
    target `delta` is sampled first and the bias solved for:
    `bias = delta + log(1 - exp(-delta))`, in the numerically stable form that
    never evaluates `exp(delta)` directly.
    """
    std = dt_rank**-0.5 * _DT_SCALE
    nn.init.uniform_(dt_proj.weight, -std, std)

    d_inner = dt_proj.out_features
    log_min, log_max = math.log(_DT_MIN), math.log(_DT_MAX)
    target_dt = torch.exp(torch.rand(d_inner) * (log_max - log_min) + log_min).clamp(
        min=_DT_INIT_FLOOR
    )
    inverse_softplus = target_dt + torch.log(-torch.expm1(-target_dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inverse_softplus)

    dt_proj.weight._no_reinit = True
    dt_proj.bias._no_reinit = True


class SelectiveScan(nn.Module):
    """The selective state space recurrence at the heart of a Mamba block.

    Per timestep, for channel `i` and state `n`:

        s[i][n] = exp(dt[i] * A[i][n]) * s[i][n] + dt[i] * B[n] * u[i]
        y[i]    = sum_n C[n] * s[i][n] + D[i] * u[i]

    `A` and `D` are fixed learned parameters; `dt`, `B` and `C` are computed
    from the input by the block around this one. That input dependence is the
    "selective" part, and is what Mamba adds over the fixed-parameter S4 models
    it builds on: the model decides, from what it currently sees, how fast the
    state decays (`dt`), how strongly to write into it (`B`) and how to read it
    out (`C`).

    A plain loop of differentiable ops, so autograd handles the backward pass
    and no custom kernel is needed. This is the obviously correct version
    rather than the fast one; a parallel associative scan is a later
    optimisation and would have to be checked against this.
    """

    def forward(self, u, delta, A, B, C, D, state=None):
        """Scan a whole sequence.

        Args:
            u: `(batch, seq, d_inner)` block input.
            delta: `(batch, seq, d_inner)` positive step size.
            A: `(d_inner, d_state)` fixed, learned, negative.
            B, C: `(batch, seq, d_state)` input dependent.
            D: `(d_inner,)` learned skip weight.
            state: `(batch, d_inner, d_state)` initial state, or None for zeros.

        Returns:
            `y` `(batch, seq, d_inner)` and the final state.
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        if state is None:
            state = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)

        # Discretize: A_bar = exp(dt * A) is exact for a linear system under a
        # zero order hold; B_bar = dt * B is the Euler approximation, which the
        # Mamba paper notes is sufficient because A dominates.
        A_bar = torch.exp(delta.unsqueeze(-1) * A)
        B_bar_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        outputs = []
        for t in range(seq_len):
            state = A_bar[:, t] * state + B_bar_u[:, t]
            outputs.append((state * C[:, t].unsqueeze(1)).sum(dim=-1))

        y = torch.stack(outputs, dim=1) + u * D
        return y, state


class MambaBlock(nn.Module):
    """One Mamba block: projections, causal convolution, gate, around the scan.

    Order: `in_proj` splits the input into a main path and a gate path; a short
    causal depthwise convolution gives the main path local context; `x_proj`
    reads `delta`, `B` and `C` off it; the scan runs; the result is gated by
    `SiLU(gate)`; `out_proj` projects back to `d_model`.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_model, d_inner = config.d_model, config.d_inner
        d_state, d_conv = config.d_state, config.d_conv
        dt_rank = config.dt_rank_resolved

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        _init_dt_proj(self.dt_proj, dt_rank)

        # A is stored as its log so that A = -exp(A_log) is negative for any
        # real A_log the optimizer produces, which keeps the state decaying and
        # the recurrence stable. Initialized to -1, -2, ..., -d_state, the
        # standard S4 starting point.
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.scan = SelectiveScan()
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _project(self, x):
        """Shared by both modes: split `in_proj` and compute `delta`, `B`, `C`.

        Keeping this in one place is what keeps the parallel and recurrent
        paths from drifting apart.
        """
        config = self.config
        dt_rank = config.dt_rank_resolved
        x_and_gate = self.in_proj(x)
        return x_and_gate.chunk(2, dim=-1)

    def _ssm_parameters(self, u):
        dt_rank = self.config.dt_rank_resolved
        d_state = self.config.d_state
        projected = self.x_proj(u)
        delta, B, C = projected.split([dt_rank, d_state, d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return delta, B, C

    def forward(self, x, state: torch.Tensor | None = None, conv_state: torch.Tensor | None = None):
        """Parallel mode over a sequence.

        Args:
            x: `(batch, seq, d_model)`.
            state: optional initial SSM state.
            conv_state: optional initial conv buffer, `(batch, d_inner, d_conv-1)`.
                Supplying it is what makes a sequence processed in consecutive
                windows identical to the same sequence processed in one go.

        Returns:
            `(output, new_ssm_state, new_conv_state)`.
        """
        batch, seq_len, _ = x.shape
        u, gate = self._project(x)

        # Conv expects channels before time.
        u = u.transpose(1, 2)
        if conv_state is not None:
            # Prepend the carried buffer instead of zero padding, then take
            # exactly seq_len outputs, so the first frames of this window see
            # the real previous frames.
            padded = torch.cat([conv_state, u], dim=-1)
            convolved = F.conv1d(
                padded,
                self.conv1d.weight,
                self.conv1d.bias,
                groups=self.config.d_inner,
            )
            u_conv = convolved[:, :, :seq_len]
        else:
            u_conv = self.conv1d(u)[:, :, :seq_len]

        new_conv_state = u[:, :, -(self.config.d_conv - 1) :]
        if new_conv_state.shape[-1] < self.config.d_conv - 1:
            pad = self.config.d_conv - 1 - new_conv_state.shape[-1]
            new_conv_state = F.pad(new_conv_state, (pad, 0))

        u_conv = F.silu(u_conv.transpose(1, 2))

        A = -torch.exp(self.A_log)
        delta, B, C = self._ssm_parameters(u_conv)
        y, new_state = self.scan(u_conv, delta, A, B, C, self.D, state)
        y = y * F.silu(gate)
        return self.out_proj(y), new_state, new_conv_state

    def step(self, x, ssm_state: torch.Tensor, conv_state: torch.Tensor):
        """Recurrent mode: one frame.

        Mirrors plan section 13.2 one line at a time, so the C++ engine can be
        written straight from this and checked against it.

        Args:
            x: `(batch, d_model)`.
            ssm_state: `(batch, d_inner, d_state)`.
            conv_state: `(batch, d_inner, d_conv - 1)`.

        Returns:
            `(output, new_ssm_state, new_conv_state)`, output `(batch, d_model)`.
        """
        config = self.config
        u, gate = self._project(x)

        # conv1d_step: the buffer holds the previous d_conv-1 inputs, so the
        # window is those plus this frame.
        window = torch.cat([conv_state, u.unsqueeze(-1)], dim=-1)
        weight = self.conv1d.weight.squeeze(1)
        u_conv = (window * weight).sum(dim=-1) + self.conv1d.bias
        new_conv_state = window[:, :, 1:]

        u_conv = F.silu(u_conv)

        A = -torch.exp(self.A_log)
        delta, B, C = self._ssm_parameters(u_conv)

        # s = exp(dt * A) * s + dt * B * u, then y = C . s + D * u
        A_bar = torch.exp(delta.unsqueeze(-1) * A)
        B_bar_u = delta.unsqueeze(-1) * B.unsqueeze(1) * u_conv.unsqueeze(-1)
        new_ssm_state = A_bar * ssm_state + B_bar_u
        y = (new_ssm_state * C.unsqueeze(1)).sum(dim=-1) + u_conv * self.D

        y = y * F.silu(gate)
        return self.out_proj(y), new_ssm_state, new_conv_state


class MambaBackbone(nn.Module):
    """A stack of Mamba blocks, each with RMSNorm and a residual connection.

    Mamba blocks are self-contained enough that they need no separate feed
    forward sub-layer the way a transformer block does, so one block per layer
    is the whole stack.
    """

    def __init__(self, config: MambaConfig | None = None, **overrides):
        super().__init__()
        if config is None:
            config = MambaConfig(**overrides)
        elif overrides:
            raise TypeError("pass either a MambaConfig or keyword overrides, not both")

        self.config = config
        self.blocks = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.norms = nn.ModuleList([RMSNorm(config.d_model) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)

    def initial_state(self, batch: int, device=None, dtype=torch.float32) -> MambaState:
        return MambaState.zeros(self.config, batch, device=device, dtype=dtype)

    def forward(self, x, state: MambaState | None = None):
        """Parallel mode.

        Args:
            x: `(batch, seq, d_model)`.
            state: optional incoming state, which makes consecutive windows
                equivalent to one long sequence.

        Returns:
            `(output, new_state)`.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (batch, seq, d_model); got {tuple(x.shape)}")

        new_conv, new_ssm = [], []
        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            ssm_in = state.ssm[i] if state is not None else None
            conv_in = state.conv[i] if state is not None else None
            delta, ssm_out, conv_out = block(norm(x), ssm_in, conv_in)
            x = x + delta
            new_ssm.append(ssm_out)
            new_conv.append(conv_out)

        return self.final_norm(x), MambaState(conv=new_conv, ssm=new_ssm)

    def step(self, x, state: MambaState):
        """Recurrent mode: one frame.

        Args:
            x: `(batch, d_model)`.
            state: the carried state.

        Returns:
            `(output, new_state)`, output `(batch, d_model)`.
        """
        if x.dim() != 2:
            raise ValueError(f"expected (batch, d_model); got {tuple(x.shape)}")

        new_conv, new_ssm = [], []
        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            delta, ssm_out, conv_out = block.step(norm(x), state.ssm[i], state.conv[i])
            x = x + delta
            new_ssm.append(ssm_out)
            new_conv.append(conv_out)

        return self.final_norm(x), MambaState(conv=new_conv, ssm=new_ssm)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def multiply_adds_per_frame(self) -> int:
        """Rough multiply-add count for one recurrent frame.

        The number plan section 4.5 budgets at 1 to 2 million. Counts the four
        matrix multiplies and the SSM update per layer, and ignores the
        element-wise activations, which are negligible beside them.
        """
        c = self.config
        dt_rank = c.dt_rank_resolved
        per_layer = (
            c.d_model * c.d_inner * 2          # in_proj
            + c.d_inner * c.d_conv             # conv step
            + c.d_inner * (dt_rank + 2 * c.d_state)  # x_proj
            + dt_rank * c.d_inner              # dt_proj
            + 3 * c.d_inner * c.d_state        # SSM update and readout
            + c.d_inner * c.d_model            # out_proj
        )
        return c.n_layers * per_layer
