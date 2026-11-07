"""
Pose losses for Phases 1 and 2 (plan section 6.2), plus the residual term.

Notation: `y_t` true pose, `yhat_t` predicted pose, `R(.)` joint rotations as
matrices, `FK(.)` joint positions in character space, `r_t` root velocity,
`c_t` foot contacts, `dt` the frame time of 1/60 s.

    L_rot     = mean | R(yhat) - R(y) |                             (L1)
    L_pos     = mean | FK(yhat) - FK(y) |
    L_vel     = mean | d FK(yhat) - d FK(y) | / dt, and the same on rotations
    L_root    = mean | rhat - r |
    L_contact = BCE(chat, c)
    L_res     = w_res * | correction | ^ 2

The velocity term is not optional: plain MSE gave jittery motion in LMM, and
the velocity term is what removes it. The position term is measured after
forward kinematics, where the eye sees the error, rather than in rotation
space, where a small error at the hips and a small error at the wrist count the
same.

The target `y` is mocap in Phase 1 and the KNN teacher's output in Phase 2. The
loss code does not care which, which is why it is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from mmm.data import rotations
from mmm.data.fk import forward_kinematics_torch
from mmm.data.skeleton import Skeleton

__all__ = ["LossWeights", "PoseLoss"]


@dataclass
class LossWeights:
    """Per term weights, set so each term has roughly equal magnitude.

    The defaults are a starting point, not a validated set. Use
    `PoseLoss.calibrate` to measure the actual magnitudes on a batch and get
    suggested weights, rather than leaving them to guesswork -- the terms are
    in different units (radians, centimetres, centimetres per second) and so
    differ by orders of magnitude out of the box.
    """

    rotation: float = 1.0
    position: float = 1.0
    velocity: float = 1.0
    root: float = 1.0
    hips_height: float = 1.0
    contact: float = 0.1
    residual: float = 0.0

    def as_dict(self) -> dict:
        return {
            "rotation": self.rotation,
            "position": self.position,
            "velocity": self.velocity,
            "root": self.root,
            "hips_height": self.hips_height,
            "contact": self.contact,
            "residual": self.residual,
        }


class PoseLoss:
    """The combined pose loss.

    Args:
        skeleton: needed for forward kinematics inside the loss.
        n_joints: joints in the pose. Taken from the skeleton if omitted.
        frame_time: seconds per frame, for the velocity term.
        weights: per term weights.
        position_scale: divides the position and velocity terms, to bring
            centimetres onto roughly the same scale as the rotation term before
            the weights are applied. LaFAN1 is in centimetres, so joint offsets
            are tens to hundreds; without this the position term dominates by
            two orders of magnitude and the weights have to absorb it.
    """

    def __init__(
        self,
        skeleton: Skeleton,
        frame_time: float = 1.0 / 60.0,
        weights: LossWeights | None = None,
        n_joints: int | None = None,
        position_scale: float = 100.0,
    ):
        self.skeleton = skeleton
        self.n_joints = n_joints if n_joints is not None else skeleton.n_joints
        self.frame_time = frame_time
        self.weights = weights or LossWeights()
        self.position_scale = position_scale

    # -- helpers ----------------------------------------------------------

    def _split(self, pose: torch.Tensor):
        """Pose vector to `(rotation_matrices, hips_height, root_velocity)`.

        Mirrors `mmm.data.database`'s layout. The 6D block is passed through
        `sixd_to_matrix`, which re-orthonormalises, so a raw network output
        becomes a valid rotation before FK sees it.
        """
        base = self.n_joints * 6
        sixd = pose[..., :base].reshape(pose.shape[:-1] + (self.n_joints, 6))
        matrices = rotations.sixd_to_matrix(sixd)
        hips_height = pose[..., base : base + 1]
        root_velocity = pose[..., base + 1 : base + 7]
        return matrices, hips_height, root_velocity

    def _fk(self, matrices: torch.Tensor, hips_height: torch.Tensor) -> torch.Tensor:
        """Joint positions in character space.

        The root is placed at `(0, hips_height, 0)`, which is where a stored
        pose lives: character space with the ground at y = 0. Heights are
        therefore comparable between prediction and target, which is what makes
        the foot position error meaningful.
        """
        batch_shape = matrices.shape[:-3]
        root = torch.zeros(batch_shape + (3,), device=matrices.device, dtype=matrices.dtype)
        root = torch.cat(
            [root[..., :1], hips_height, root[..., 2:3]], dim=-1
        )
        positions, _ = forward_kinematics_torch(self.skeleton, matrices, root)
        return positions

    # -- the loss ---------------------------------------------------------

    def __call__(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        predicted_contacts: torch.Tensor | None = None,
        target_contacts: torch.Tensor | None = None,
        correction: torch.Tensor | None = None,
    ):
        """Compute the total loss and every term.

        Args:
            predicted: `(batch, seq, pose_dim)`.
            target: `(batch, seq, pose_dim)`.
            predicted_contacts: `(batch, seq, n_feet)` logits, optional.
            target_contacts: `(batch, seq, n_feet)` in `{0, 1}`, optional.
            correction: `(batch, seq, pose_dim)` for the residual variant.

        Returns:
            `(total, terms)` where `terms` is a dict of detached floats, so a
            caller can log every term separately. A single scalar hides which
            term the model is actually improving.
        """
        if predicted.shape != target.shape:
            raise ValueError(
                f"predicted {tuple(predicted.shape)} and target {tuple(target.shape)} "
                "must have the same shape"
            )
        if predicted.dim() != 3:
            raise ValueError(f"expected (batch, seq, pose_dim); got {tuple(predicted.shape)}")

        pred_rot, pred_height, pred_root = self._split(predicted)
        true_rot, true_height, true_root = self._split(target)

        # Rotation: L1 on the matrices, not on the 6D vectors. Two different 6D
        # vectors can denote the same rotation, so comparing them directly
        # penalises a correct prediction.
        l_rot = (pred_rot - true_rot).abs().mean()

        pred_pos = self._fk(pred_rot, pred_height)
        true_pos = self._fk(true_rot, true_height)
        l_pos = (pred_pos - true_pos).abs().mean() / self.position_scale

        # Velocity, on positions and on rotations. Needs at least two frames.
        if predicted.shape[1] > 1:
            pred_dp = (pred_pos[:, 1:] - pred_pos[:, :-1]) / self.frame_time
            true_dp = (true_pos[:, 1:] - true_pos[:, :-1]) / self.frame_time
            l_vel_pos = (pred_dp - true_dp).abs().mean() / self.position_scale

            pred_dr = (pred_rot[:, 1:] - pred_rot[:, :-1]) / self.frame_time
            true_dr = (true_rot[:, 1:] - true_rot[:, :-1]) / self.frame_time
            l_vel_rot = (pred_dr - true_dr).abs().mean()
            l_vel = l_vel_pos + l_vel_rot
        else:
            l_vel = predicted.new_zeros(())

        # Root velocity and hips height, as two separate terms.
        #
        # They used to be summed into one. That was a mistake, and a costly
        # one: velocity carries numbers around 27 (cm/s) while the height
        # carries about 0.8 (centimetres over `position_scale`), a ratio of
        # 33 to 1. `calibrate` then sized the single weight from the combined
        # magnitude, which velocity dominates, and the height half was scaled
        # down with it. Measured on the first trained run, a 10 cm error in
        # the character's height cost 1.6% of the loss while a 5 cm/s error in
        # its velocity cost 41%.
        #
        # Splitting them lets `calibrate` give each the weight its own scale
        # deserves. Note this changes what `root` means: it is now velocity
        # only, so the term is not comparable with runs made before the split.
        l_root = (pred_root - true_root).abs().mean()
        l_height = (pred_height - true_height).abs().mean() / self.position_scale

        w = self.weights
        total = (
            w.rotation * l_rot
            + w.position * l_pos
            + w.velocity * l_vel
            + w.root * l_root
            + w.hips_height * l_height
        )
        terms = {
            "rotation": float(l_rot.detach()),
            "position": float(l_pos.detach()),
            "velocity": float(l_vel.detach()),
            "root": float(l_root.detach()),
            "hips_height": float(l_height.detach()),
        }

        if predicted_contacts is not None and target_contacts is not None:
            l_contact = F.binary_cross_entropy_with_logits(
                predicted_contacts, target_contacts.to(predicted_contacts.dtype)
            )
            total = total + w.contact * l_contact
            terms["contact"] = float(l_contact.detach())

        if correction is not None and w.residual > 0.0:
            l_res = correction.pow(2).mean()
            total = total + w.residual * l_res
            terms["residual"] = float(l_res.detach())

        if not torch.isfinite(total):
            raise ValueError(f"non-finite loss: terms were {terms}")

        terms["total"] = float(total.detach())
        return total, terms

    def calibrate(self, predicted: torch.Tensor, target: torch.Tensor) -> dict:
        """Measure each term's unweighted magnitude and suggest weights.

        Plan section 6.2 asks for weights "set so each term has roughly equal
        magnitude". This measures them on one batch and returns the reciprocals
        normalized to the rotation term, which is a defensible starting point
        rather than a guess.
        """
        saved = self.weights
        self.weights = LossWeights(
            rotation=1.0, position=1.0, velocity=1.0, root=1.0,
            hips_height=1.0, contact=0.0,
        )
        try:
            with torch.no_grad():
                _, terms = self(predicted, target)
        finally:
            self.weights = saved

        reference = max(terms["rotation"], 1e-12)
        return {
            name: reference / max(terms[name], 1e-12)
            for name in ("rotation", "position", "velocity", "root", "hips_height")
        }
