"""
The assembled model: input assembly, backbone, head, and one frame of state.

This is what training, the environment and the exporter all talk to, so the
exact input layout is defined in one place and cannot drift between the training
loop and the runtime.

Input assembly, plan section 4.3:

    joystick        12         desired future root positions and facing at 20,
                               40 and 60 frames, in character space
    previous pose   pose_dim   the model's own previous output

There is no retrieval in this input. A mechanism that concatenated raw
candidate poses from the Animation Database straight onto this vector used to
live here, for both training and inference; it has been removed. Retrieval is
coming back as an encoder feeding the backbone through cross-attention
instead, not by widening this vector again.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmm.models.heads import FullPoseHead, ResidualHead
from mmm.models.io_layers import InputLayer, IoSpec, OutputLayer
from mmm.models.mamba.backbone import MambaBackbone, MambaConfig, MambaState

__all__ = ["MmmPolicy"]


class MmmPolicy(nn.Module):
    """Linear in, backbone, linear out, plus a pose head.

    Args:
        spec: the input and output widths.
        backbone: a `MambaBackbone`, or any module exposing the same
            `forward(x, state)` / `step(x, state)` / `initial_state(batch)`
            interface. The GRU and transformer baselines of plan section 9 drop
            in here unchanged, which is the point of the interface.
        head: "full" for direct pose output, "residual" for retrieve then
            refine (plan section 4.4).
    """

    def __init__(
        self,
        spec: IoSpec,
        backbone: nn.Module | None = None,
        head: str = "full",
        d_model: int = 256,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.spec = spec
        # Plain dropout, off by default. The strongest overfitting controls
        # here are the clip-disjoint split and early stopping; these are for
        # when the validation curve turns up before the training curve flattens.
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.hidden_dropout = nn.Dropout(hidden_dropout) if hidden_dropout > 0 else nn.Identity()

        if backbone is None:
            backbone = MambaBackbone(MambaConfig(d_model=d_model))
        self.backbone = backbone
        self.d_model = d_model

        self.input_layer = InputLayer(spec, d_model)
        self.output_layer = OutputLayer(spec, d_model)

        if head == "full":
            self.head = FullPoseHead(spec)
        elif head == "residual":
            self.head = ResidualHead(spec)
        else:
            raise ValueError(f"head must be 'full' or 'residual'; got {head!r}")
        self.head_kind = head

    # -- input assembly ---------------------------------------------------

    def build_input(
        self,
        joystick: torch.Tensor,
        previous_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate the blocks in the fixed documented order.

        Args:
            joystick: `(..., 12)`.
            previous_pose: `(..., pose_dim)`.

        Returns:
            `(..., input_dim)`.
        """
        return torch.cat([joystick, previous_pose], dim=-1)

    # -- forward ----------------------------------------------------------

    def initial_state(self, batch: int, device=None, dtype=torch.float32):
        return self.backbone.initial_state(batch, device=device, dtype=dtype)

    def forward(self, joystick: torch.Tensor, previous_pose: torch.Tensor, state=None):
        """Parallel mode over a sequence, used for training.

        All pose arguments are `(batch, seq, ...)`. `previous_pose` is the
        teacher forced or scheduled sampled previous frame, supplied by the
        caller rather than fed back here, so this stays a pure function of its
        inputs and the whole window trains in one pass.

        Returns:
            `(pose, contacts, new_state)`.
        """
        x = self.build_input(joystick, previous_pose)
        hidden, new_state = self.backbone(self.input_dropout(self.input_layer(x)), state)
        raw_pose, contacts = self.output_layer(self.hidden_dropout(hidden))
        pose = self.head(raw_pose)
        return pose, contacts, new_state

    def step(self, joystick: torch.Tensor, previous_pose: torch.Tensor, state=None):
        """Recurrent mode: one frame. What the runtime loop calls.

        All pose arguments are `(batch, ...)` with no sequence axis.

        Returns:
            `(pose, contacts, new_state)`.
        """
        if state is None:
            state = self.initial_state(joystick.shape[0], device=joystick.device, dtype=joystick.dtype)
        x = self.build_input(joystick, previous_pose)
        hidden, new_state = self.backbone.step(self.input_dropout(self.input_layer(x)), state)
        raw_pose, contacts = self.output_layer(self.hidden_dropout(hidden))
        pose = self.head(raw_pose)
        return pose, contacts, new_state

    # -- reporting --------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict:
        """The frozen architecture description the exporter and the C++ engine
        read, so the engine is built against an explicit contract."""
        config = getattr(self.backbone, "config", None)
        return {
            "pose_dim": self.spec.pose_dim,
            "joystick_dim": self.spec.joystick_dim,
            "input_dim": self.spec.input_dim,
            "output_dim": self.spec.output_dim,
            "predict_contacts": self.spec.predict_contacts,
            "d_model": self.d_model,
            "head": self.head_kind,
            "backbone": type(self.backbone).__name__,
            "n_layers": getattr(config, "n_layers", None),
            "d_state": getattr(config, "d_state", None),
            "d_conv": getattr(config, "d_conv", None),
            "d_inner": getattr(config, "d_inner", None),
            "n_heads": getattr(config, "n_heads", None),
            "window": getattr(config, "window", None),
            "parameters": self.n_parameters(),
        }
