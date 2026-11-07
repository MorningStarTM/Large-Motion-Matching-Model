"""
Driving the trained model in the simulation.

`KnnController` plays back frames from a database. `ModelController` generates
them: every frame it hands the policy the joystick and the pose it produced
last time, takes the pose that comes back, and integrates the root. Nothing is
retrieved and nothing is played back, so this is the first time the model has
to stand on its own.

That is the point. Every number in the Phase 1 report is computed against
mocap the model never has to live with -- even `val_rollout_total` is 64
frames with a known starting pose and a known joystick. A viewer session is
minutes of free running with a human changing their mind, and it is the only
way to find out whether the motion is actually usable. A model can score well
and still drift into a crouch after four seconds.

The same interface as `KnnController`
-------------------------------------
Both viewers already drive a controller through `step`, `reset`, `state` and
`world_positions`, so this exposes exactly those and the viewers do not care
which one they have. `mmm/teacher/controller.py` is untouched; this is not a
subclass of it, because almost nothing is shared -- there is no index, no
inertializer, no transition cost, and no row to advance.

What `StepRecord` carries is deliberately the same shape, with `searched`,
`jumped` and `candidates` left empty. An overlay that draws retrieval
candidates then simply draws none, instead of needing to know the difference.

Root integration
----------------
`root_source` means the same three things it does for the teacher, and the
default is again `"hybrid"`: facing from the spring damper so the player can
steer, translation from the model's own predicted velocity so the feet do not
slide. `"animation"` here means the model's prediction owns both, which is the
honest test of whether it learned to turn -- and, on the Phase 1 checkpoint,
the setting most likely to wander, since nothing in pre-training rewarded
following a joystick.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from mmm.data import features as feat
from mmm.data import spaces
from mmm.data.database import (
    pose_to_hips_height,
    pose_to_root_velocity,
    pose_to_rotations,
)
from mmm.data.fk import forward_kinematics
from mmm.data.skeleton import HIPS, LEFT_FOOT, RIGHT_FOOT, Skeleton
from mmm.models.io_layers import IoSpec
from mmm.models.policy import MmmPolicy
from mmm.teacher.controller import ControllerState, StepRecord
from mmm.teacher.trajectory import TrajectoryPredictor
from mmm.training.checkpoints import load_checkpoint

__all__ = ["ModelController", "load_policy_from_manifest", "build_policy"]


def build_policy(architecture: dict) -> MmmPolicy:
    """Rebuild a policy from the architecture block a run recorded.

    The manifest is the contract. Guessing any of these -- `d_model`,
    `n_layers`, whether retrieval is in the input -- produces a state dict that
    either refuses to load or, worse, loads partially under `strict=False` and
    runs with random weights in the layers that did not match.
    """
    from mmm.models.mamba_backbone import MambaBackbone, MambaConfig

    spec = IoSpec(
        pose_dim=int(architecture["pose_dim"]),
        joystick_dim=int(architecture.get("joystick_dim", 12)),
        n_candidates=int(architecture.get("n_candidates", 4)),
        use_retrieval=bool(architecture.get("use_retrieval", False)),
        predict_contacts=bool(architecture.get("predict_contacts", True)),
    )

    d_model = int(architecture.get("d_model", 256))
    backbone = MambaBackbone(
        MambaConfig(
            d_model=d_model,
            n_layers=int(architecture.get("n_layers", 4)),
            d_state=int(architecture.get("d_state", 16)),
            d_conv=int(architecture.get("d_conv", 4)),
        )
    )
    return MmmPolicy(
        spec=spec,
        backbone=backbone,
        head=str(architecture.get("head", "full")),
        d_model=d_model,
    )


def load_policy_from_manifest(
    checkpoint: str, manifest: str | None = None, device: str = "cpu"
):
    """Load a published checkpoint, using the manifest beside it.

    Args:
        checkpoint: a `.safetensors` file.
        manifest: `manifest.json`. Defaults to the one in the same folder.
        device: where to put the weights.

    Returns:
        `(policy, architecture, metadata)`.
    """
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"{checkpoint} not found")

    manifest = manifest or os.path.join(os.path.dirname(checkpoint), "manifest.json")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(
            f"{manifest} not found. A checkpoint alone does not say how wide "
            "the model is or whether retrieval is in its input, so it cannot "
            "be rebuilt without the manifest that was published with it."
        )

    with open(manifest, encoding="utf-8") as handle:
        blob = json.load(handle)
    architecture = blob.get("architecture")
    if not architecture:
        raise ValueError(f"{manifest} has no 'architecture' block")

    policy = build_policy(architecture)
    metadata = load_checkpoint(checkpoint, policy=policy, device=device, strict=True)
    policy.to(device).eval()

    expected = architecture.get("parameters")
    actual = policy.n_parameters()
    if expected is not None and int(expected) != actual:
        raise ValueError(
            f"{checkpoint}: rebuilt {actual:,} parameters but the manifest "
            f"records {int(expected):,}. The architecture block does not "
            "describe these weights."
        )

    return policy, architecture, metadata


@dataclass
class ModelDiagnostics:
    """Per frame numbers worth putting on a HUD.

    Drift is the thing to watch. A model that is fine for two seconds and then
    sinks, leans or speeds up has a problem no validation loss reports, and
    the hips height against its starting value is the cheapest way to see it.
    """

    hips_height: float = 0.0
    hips_drift: float = 0.0
    speed: float = 0.0
    requested_speed: float = 0.0
    contact: tuple = (0.0, 0.0)


class ModelController:
    """Drives the trained policy, frame by frame, from a joystick.

    Args:
        policy: a loaded `MmmPolicy` in eval mode.
        skeleton: the joint tree the pose describes.
        seed_pose: `(pose_dim,)` the pose to start from and to feed as the
            first `previous_pose`. There has to be one: the model predicts the
            *next* pose, so frame zero has no input otherwise. Use a real
            mocap frame, not zeros -- a zero pose is not a valid rotation and
            the model has never seen anything like it.
        frame_time: seconds per frame.
        device: torch device.
        root_source: "hybrid", "simulation" or "animation"; see the module
            docstring.
        trajectory: the spring damper. One is built if not given.
    """

    def __init__(
        self,
        policy: MmmPolicy,
        skeleton: Skeleton,
        seed_pose: np.ndarray,
        frame_time: float = 1.0 / 60.0,
        device: str = "cpu",
        root_source: str = "hybrid",
        trajectory: TrajectoryPredictor | None = None,
    ):
        if root_source not in ("hybrid", "simulation", "animation"):
            raise ValueError(
                "root_source must be 'hybrid', 'simulation' or 'animation'; "
                f"got {root_source!r}"
            )

        seed_pose = np.asarray(seed_pose, np.float32).reshape(-1)
        expected = policy.spec.pose_dim
        if seed_pose.shape[0] != expected:
            raise ValueError(
                f"seed_pose is {seed_pose.shape[0]} wide but the policy's pose "
                f"is {expected}. These are different skeletons or different "
                "pose layouts."
            )

        self.policy = policy
        self.skeleton = skeleton
        self.n_joints = skeleton.n_joints
        self.frame_time = frame_time
        self.device = torch.device(device)
        self.root_source = root_source
        self.seed_pose = seed_pose

        self.left_foot = skeleton.index(LEFT_FOOT)
        self.right_foot = skeleton.index(RIGHT_FOOT)
        self.hips = skeleton.index(HIPS)

        self.trajectory = (
            trajectory
            if trajectory is not None
            else TrajectoryPredictor(frame_time=frame_time)
        )

        self.diagnostics = ModelDiagnostics()
        self.state: ControllerState | None = None
        self._hidden = None
        self._previous_pose = None
        self._start_height = 0.0
        self.reset()

    # -- construction -----------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        skeleton: Skeleton,
        seed_pose: np.ndarray,
        manifest: str | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> "ModelController":
        policy, architecture, _ = load_policy_from_manifest(
            checkpoint, manifest=manifest, device=device
        )
        if architecture.get("use_retrieval"):
            raise ValueError(
                f"{checkpoint} was trained with retrieval in its input, so it "
                "needs candidate poses this controller does not supply. That "
                "is a Phase 2 checkpoint; drive it with a controller that "
                "carries a database."
            )
        return cls(policy, skeleton, seed_pose, device=device, **kwargs)

    # -- lifecycle --------------------------------------------------------

    def reset(self, origin=None, forward=None, seed_pose=None) -> None:
        """Put the character back at the start, with a fresh hidden state.

        The hidden state matters. A Mamba backbone carries everything it knows
        about the recent past in it, so resetting the position without
        resetting the state leaves the model mid-stride at the origin and it
        lurches.
        """
        pose = self.seed_pose if seed_pose is None else np.asarray(seed_pose, np.float32)
        origin = np.zeros(3, np.float32) if origin is None else np.asarray(origin, np.float32)
        forward = (
            np.array([0.0, 1.0], np.float32) if forward is None
            else np.asarray(forward, np.float32)
        )

        rotations = pose_to_rotations(pose, self.n_joints)
        linear, angular = pose_to_root_velocity(pose, self.n_joints)
        height = float(pose_to_hips_height(pose, self.n_joints))

        self.state = ControllerState(
            row=-1,                      # no database row: nothing is played back
            origin=origin.copy(),
            forward=forward.copy(),
            local_rotations=np.asarray(rotations, np.float32),
            hips_height=height,
            linear_velocity=np.asarray(linear, np.float32),
            angular_velocity=np.asarray(angular, np.float32),
        )
        self._start_height = height
        self._previous_pose = torch.from_numpy(pose.copy()).to(self.device)[None]
        self._hidden = None
        self.trajectory.reset() if hasattr(self.trajectory, "reset") else None

    # -- the loop ---------------------------------------------------------

    def _joystick(self, state: ControllerState) -> np.ndarray:
        """The 12 numbers the model reads, in the training layout.

        Identical to the first two blocks of the teacher's query, and built by
        the same spring damper, so what the model sees at runtime has the same
        meaning as what it saw during training. Any divergence here is
        invisible and fatal: the model would be steering by a signal that
        means something slightly different.
        """
        desired_positions, desired_directions = self.trajectory.predict(state.forward)
        return np.concatenate(
            [
                np.asarray(desired_positions, np.float32).reshape(-1),
                np.asarray(desired_directions, np.float32).reshape(-1),
            ]
        ).astype(np.float32)

    @torch.no_grad()
    def step(self, stick: np.ndarray, facing_stick=None) -> StepRecord:
        """Advance one frame.

        Args:
            stick: `(2,)` movement request in world XZ, magnitude `[0, 1]`.
            facing_stick: `(2,)` optional separate facing request.

        Returns:
            A `StepRecord` with the generated pose. `searched`, `jumped` and
            `candidates` are always empty -- nothing is retrieved -- so an
            overlay written for the teacher draws no candidates rather than
            breaking.
        """
        state = self.state
        self.trajectory.update(stick, facing_stick)

        joystick = torch.from_numpy(self._joystick(state)).to(self.device)[None]
        pose_tensor, contact_logits, self._hidden = self.policy.step(
            joystick, self._previous_pose, state=self._hidden
        )
        # Fed straight back in next frame. This is the whole difference from
        # the teacher: there is no ground truth to fall back on, so an error
        # here is an error the model has to recover from itself.
        self._previous_pose = pose_tensor

        pose = pose_tensor[0].detach().cpu().numpy().astype(np.float32)
        contacts = (
            torch.sigmoid(contact_logits)[0].detach().cpu().numpy()
            if contact_logits is not None
            else np.zeros(2, np.float32)
        )

        # `pose_to_rotations` re-orthonormalises, so a raw 6D output is always
        # a valid rotation even before the model has learned to produce one.
        state.local_rotations = np.asarray(
            pose_to_rotations(pose, self.n_joints), np.float32
        )
        state.hips_height = float(pose_to_hips_height(pose, self.n_joints))
        linear, angular = pose_to_root_velocity(pose, self.n_joints)
        state.linear_velocity = np.asarray(linear, np.float32)
        state.angular_velocity = np.asarray(angular, np.float32)

        self._integrate_root(state)
        self._update_diagnostics(state, contacts)

        return StepRecord(
            pose=pose,
            local_rotations=state.local_rotations,
            origin=state.origin.copy(),
            forward=state.forward.copy(),
            searched=False,
            jumped=False,
            query=None,
            candidates=None,
            distances=None,
            chosen_row=-1,
        )

    def _integrate_root(self, state: ControllerState) -> None:
        """Advance the root frame, by whichever of the three rules is set."""
        if self.root_source == "simulation":
            state.origin, state.forward = self.trajectory.step_root(
                state.origin, state.forward
            )
            state.linear_velocity, state.angular_velocity = (
                self.trajectory.realised_root_velocity(state.forward)
            )
            return

        if self.root_source == "animation":
            # The model's own predicted root velocity owns both position and
            # facing. The honest test of whether it learned to turn, and the
            # setting most likely to wander on a Phase 1 checkpoint.
            state.origin, state.forward = spaces.integrate_root(
                state.origin,
                state.forward,
                state.linear_velocity,
                state.angular_velocity,
                self.frame_time,
            )
            return

        # "hybrid": facing from the spring damper so it steers, translation
        # from the model's velocity so the feet keep up with the body.
        new_forward = self.trajectory.facing.copy()
        step = spaces._rotate_out_of_frame(
            np.array(
                [state.linear_velocity[0], state.linear_velocity[2]], np.float32
            )[None]
            * self.frame_time,
            new_forward[None],
        )[0]
        state.origin = state.origin.copy()
        state.origin[0] += float(step[0])
        state.origin[2] += float(step[1])
        state.origin[1] = 0.0
        state.forward = new_forward
        state.angular_velocity = np.array(
            [0.0, self.trajectory.facing_rate, 0.0], np.float32
        )

    def _update_diagnostics(self, state: ControllerState, contacts) -> None:
        planar = np.array([state.linear_velocity[0], state.linear_velocity[2]])
        self.diagnostics = ModelDiagnostics(
            hips_height=state.hips_height,
            hips_drift=state.hips_height - self._start_height,
            speed=float(np.linalg.norm(planar)),
            requested_speed=float(np.linalg.norm(self.trajectory.velocity[:2]))
            if hasattr(self.trajectory, "velocity")
            else 0.0,
            contact=(float(contacts[0]), float(contacts[-1])),
        )

    # -- geometry, matching KnnController ---------------------------------

    def character_space_positions(self, state: ControllerState) -> np.ndarray:
        root = np.array([[0.0, state.hips_height, 0.0]], np.float32)
        positions, _ = forward_kinematics(
            self.skeleton, state.local_rotations[None], root
        )
        return positions[0]

    def world_positions(self, state: ControllerState) -> np.ndarray:
        local = self.character_space_positions(state)[None]
        frame = spaces.RootFrame(origin=state.origin[None], forward=state.forward[None])
        return spaces.from_character_space(local, frame)[0]
