"""
The assembled model: input assembly, backbone, head, and one frame of state.

This is what training, the environment and the exporter all talk to, so the
exact input layout is defined in one place and cannot drift between the training
loop and the runtime.

Input assembly, plan section 4.3:

    joystick        12         desired future root positions and facing at 20,
                               40 and 60 frames, in character space
    previous pose   pose_dim   the model's own previous output
    candidates      K x pose   the top K retrieved poses from the Animation
                               Database
    candidate mask  K          1 where a candidate is real

The retrieved candidates are plain pose numbers: no encoder sits between the
Animation Database and the model, which is precisely what allows a clip
appended to the KB to be usable immediately.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmm.models.heads import FullPoseHead, ResidualHead
from mmm.models.io_layers import InputLayer, IoSpec, OutputLayer
from mmm.models.mamba_backbone import MambaBackbone, MambaConfig, MambaState

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
        retrieval_dropout: probability of blanking all candidates for a
            training sequence. This is one of the three Phase 1 mechanisms that
            force the model to actually use retrieval; it lives here, next to
            the layout it perturbs.
    """

    def __init__(
        self,
        spec: IoSpec,
        backbone: nn.Module | None = None,
        head: str = "full",
        d_model: int = 256,
        retrieval_dropout: float = 0.1,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.spec = spec
        self.retrieval_dropout = retrieval_dropout
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
        candidates: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        apply_retrieval_dropout: bool = False,
    ) -> torch.Tensor:
        """Concatenate the blocks in the fixed documented order.

        Args:
            joystick: `(..., 12)`.
            previous_pose: `(..., pose_dim)`.
            candidates: `(..., K, pose_dim)`.
            candidate_mask: `(..., K)` with 1 for a real candidate. Defaults to
                all ones.
            apply_retrieval_dropout: whether to randomly blank the candidates.
                Only meaningful in training; the caller decides, so evaluation
                can never silently drop retrieval.

        Returns:
            `(..., input_dim)`.
        """
        spec = self.spec
        batch_shape = joystick.shape[:-1]

        if not spec.use_retrieval:
            # Phase 1: motion only. Candidates are accepted and ignored, so a
            # loader that still produces them costs nothing and the same
            # training loop serves both settings.
            return torch.cat([joystick, previous_pose], dim=-1)

        if candidates is None:
            raise ValueError(
                "this policy was built with use_retrieval=True, so candidates "
                "are required; got None"
            )

        if candidates.shape[-2:] != (spec.n_candidates, spec.pose_dim):
            raise ValueError(
                f"candidates must be (..., {spec.n_candidates}, {spec.pose_dim}); "
                f"got {tuple(candidates.shape)}"
            )

        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch_shape + (spec.n_candidates,), device=joystick.device, dtype=joystick.dtype
            )

        if apply_retrieval_dropout and self.retrieval_dropout > 0.0:
            # One decision per sequence, not per frame: dropping retrieval for
            # a scattering of individual frames would let the model interpolate
            # across the gaps and learn nothing about coping without it.
            keep = (
                torch.rand(batch_shape[:1] + (1,) * (len(batch_shape) - 1) + (1,),
                           device=joystick.device)
                > self.retrieval_dropout
            ).to(joystick.dtype)
            candidate_mask = candidate_mask * keep

        masked_candidates = candidates * candidate_mask.unsqueeze(-1)

        return torch.cat(
            [
                joystick,
                previous_pose,
                masked_candidates.reshape(batch_shape + (spec.n_candidates * spec.pose_dim,)),
                candidate_mask,
            ],
            dim=-1,
        )

    # -- forward ----------------------------------------------------------

    def initial_state(self, batch: int, device=None, dtype=torch.float32):
        return self.backbone.initial_state(batch, device=device, dtype=dtype)

    def forward(
        self,
        joystick: torch.Tensor,
        previous_pose: torch.Tensor,
        candidates: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        state=None,
        apply_retrieval_dropout: bool = False,
    ):
        """Parallel mode over a sequence, used for training.

        All pose arguments are `(batch, seq, ...)`. `previous_pose` is the
        teacher forced or scheduled sampled previous frame, supplied by the
        caller rather than fed back here, so this stays a pure function of its
        inputs and the whole window trains in one pass.

        Returns:
            `(pose, contacts, new_state)`.
        """
        x = self.build_input(
            joystick, previous_pose, candidates, candidate_mask, apply_retrieval_dropout
        )
        hidden, new_state = self.backbone(self.input_dropout(self.input_layer(x)), state)
        raw_pose, contacts = self.output_layer(self.hidden_dropout(hidden))
        pose = self.head(raw_pose, candidates, candidate_mask)
        return pose, contacts, new_state

    def step(
        self,
        joystick: torch.Tensor,
        previous_pose: torch.Tensor,
        candidates: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        state=None,
    ):
        """Recurrent mode: one frame. What the runtime loop calls.

        All pose arguments are `(batch, ...)` with no sequence axis.

        Returns:
            `(pose, contacts, new_state)`.
        """
        if state is None:
            state = self.initial_state(joystick.shape[0], device=joystick.device, dtype=joystick.dtype)
        x = self.build_input(joystick, previous_pose, candidates, candidate_mask)
        hidden, new_state = self.backbone.step(self.input_dropout(self.input_layer(x)), state)
        raw_pose, contacts = self.output_layer(self.hidden_dropout(hidden))
        pose = self.head(raw_pose, candidates, candidate_mask)
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
            "n_candidates": self.spec.n_candidates,
            "use_retrieval": self.spec.use_retrieval,
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
            "parameters": self.n_parameters(),
        }
