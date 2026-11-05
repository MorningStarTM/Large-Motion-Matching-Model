"""
The trainable linear input and output layers.

Plan section 4.3: inputs go through one linear map to width 256, and the
backbone's output through one linear map to pose size. These two layers are
also the first and last matrix multiply of the C++ engine, so their shapes and
weight order are part of the export contract.

Skeleton specific adapters (plan section 15.3): both layers are swappable while
the backbone is kept, which is what allows the backbone to be pretrained on the
SMPL skeleton and then fine tuned on Geno with fresh linear layers. They are
therefore kept free of any other state, so that swap stays trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["IoSpec", "InputLayer", "OutputLayer"]


@dataclass
class IoSpec:
    """Widths of everything entering and leaving the model.

    Attributes:
        pose_dim: the stored pose width, `n_joints * 6 + 7`.
        joystick_dim: 12, the desired future root positions and facing.
        n_candidates: K, how many retrieved poses the model reads. Ignored
            when `use_retrieval` is False, but kept so the same spec can be
            extended for a later phase without re-deciding K.
        use_retrieval: whether the candidate block is part of the input at all.

            False is the Phase 1 setting. Pre-training predicts the next mocap
            frame from the true previous one, and at 60Hz that is nearly solved
            by continuity: measured on LaFAN1, copying the previous pose scores
            2.29 against 16.24 for copying the nearest retrieved candidate, and
            the retrieval ablation came out at -0.0013. So the candidate block
            -- 604 of 771 input numbers, 78% of the width -- carries no signal
            the objective rewards, and training with it risks teaching the
            model that its knowledge base is noise, which Phase 2 would then
            have to unlearn.

            Retrieval is introduced in Phase 2, where the target is the KNN
            teacher's own output and the candidates are therefore causally
            load-bearing. The handoff extends this layer rather than replacing
            it; see `mmm.training.checkpoints.extend_input_layer`.
        predict_contacts: whether the output carries foot contact logits.
        n_feet: how many contact logits, if it does.
    """

    pose_dim: int
    joystick_dim: int = 12
    n_candidates: int = 4
    use_retrieval: bool = True
    predict_contacts: bool = True
    n_feet: int = 2

    @property
    def input_dim(self) -> int:
        """Total width entering the input layer.

        Joystick, the model's own previous pose, and -- when retrieval is in
        use -- K retrieved poses plus one validity flag each, so a masked-out
        candidate is distinguishable from a genuinely zero pose.
        """
        width = self.joystick_dim + self.pose_dim
        if self.use_retrieval:
            width += self.n_candidates * self.pose_dim + self.n_candidates
        return width

    @property
    def output_dim(self) -> int:
        return self.pose_dim + (self.n_feet if self.predict_contacts else 0)

    def block_offsets(self) -> dict[str, slice]:
        """Where each block sits in the input vector.

        Declared here so the C++ engine, the ONNX export and the policy all
        read one definition instead of three agreeing by luck.
        """
        blocks = [
            ("joystick", self.joystick_dim),
            ("previous_pose", self.pose_dim),
        ]
        if self.use_retrieval:
            blocks += [
                ("candidates", self.n_candidates * self.pose_dim),
                ("candidate_mask", self.n_candidates),
            ]

        offsets = {}
        cursor = 0
        for name, size in blocks:
            offsets[name] = slice(cursor, cursor + size)
            cursor += size
        return offsets


class InputLayer(nn.Module):
    """One linear map from the assembled input vector to the model width."""

    def __init__(self, spec: IoSpec, d_model: int = 256):
        super().__init__()
        self.spec = spec
        self.d_model = d_model
        self.proj = nn.Linear(spec.input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.spec.input_dim:
            raise ValueError(
                f"input has width {x.shape[-1]}, expected {self.spec.input_dim}. "
                "The layout is defined by IoSpec.block_offsets()."
            )
        return self.proj(x)


class OutputLayer(nn.Module):
    """One linear map from the model width to pose size, plus contact logits.

    The pose and the contacts come out of a single matrix multiply and are then
    split, rather than from two heads, so the engine has exactly one output
    projection to implement.
    """

    def __init__(self, spec: IoSpec, d_model: int = 256):
        super().__init__()
        self.spec = spec
        self.proj = nn.Linear(d_model, spec.output_dim)

        # A near-zero output layer makes the first predictions small rather
        # than arbitrary, which matters most for the residual variant, where
        # a small correction means "stay close to the retrieved pose".
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor):
        """Returns `(pose_output, contact_logits)`; contacts are None if the
        spec does not ask for them."""
        out = self.proj(hidden)
        pose = out[..., : self.spec.pose_dim]
        contacts = out[..., self.spec.pose_dim :] if self.spec.predict_contacts else None
        return pose, contacts
