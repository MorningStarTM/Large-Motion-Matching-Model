"""
Inertialization: smooth blending when playback jumps between clips.

Without it, every 10 frame jump is a visible pop. With it, the *offset* between
the pose before the jump and the pose after it is recorded at the moment of the
jump and then decayed towards zero over a short half life. Playback follows the
new clip immediately; only the difference is faded out, which is why this costs
nothing per frame beyond one decay and keeps the new motion's timing intact.

Rotation offsets are decayed in scaled angle axis space. A rotation vector can
be multiplied by a scalar, which is exactly what a decay needs and what neither
matrices nor quaternions permit directly.

Follows Holden's Spring Roll Call. The same object serves the teacher and, at
runtime, the residual variant of the model, which is why it lives here rather
than inside the controller.
"""

from __future__ import annotations

import numpy as np

from mmm.data import rotations

__all__ = ["Inertializer"]


def _decay_spring_damper(
    x: np.ndarray, v: np.ndarray, halflife: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Decay an offset and its velocity towards zero, critically damped.

    This is the spring of `trajectory.py` with a goal of zero, written out
    separately because it is applied to offsets of several different shapes and
    needs no goal argument.
    """
    damping = 4.0 * 0.6931471805599453 / (halflife + 1e-5)
    c = damping * 0.5
    j = v + x * c
    decay = np.exp(-c * dt)
    return decay * (x + j * dt), decay * (v - j * c * dt)


class Inertializer:
    """Decaying pose offset, applied on top of whatever the source is playing.

    Usage, per frame:

        if jumping:
            inertializer.transition(current_rotations, target_rotations, ...)
        rotations, linear, angular = inertializer.apply(source_rotations, ...)
        inertializer.step()

    Args:
        n_joints: joints in the skeleton.
        frame_time: seconds per frame.
        rotation_halflife: how quickly the rotation offset fades. Short enough
            to be invisible, long enough to hide the discontinuity; 0.1 to 0.2
            seconds is the usual range.
        velocity_halflife: the same for the root velocity offset.
    """

    def __init__(
        self,
        n_joints: int,
        frame_time: float = 1.0 / 60.0,
        rotation_halflife: float = 0.15,
        velocity_halflife: float = 0.2,
    ):
        self.n_joints = n_joints
        self.frame_time = frame_time
        self.rotation_halflife = rotation_halflife
        self.velocity_halflife = velocity_halflife

        # Offsets live in scaled angle axis space, one 3-vector per joint.
        self.rotation_offset = np.zeros((n_joints, 3), np.float32)
        self.rotation_offset_rate = np.zeros((n_joints, 3), np.float32)
        self.linear_offset = np.zeros(3, np.float32)
        self.linear_offset_rate = np.zeros(3, np.float32)
        self.angular_offset = np.zeros(3, np.float32)
        self.angular_offset_rate = np.zeros(3, np.float32)

    def reset(self) -> None:
        self.rotation_offset[:] = 0.0
        self.rotation_offset_rate[:] = 0.0
        self.linear_offset[:] = 0.0
        self.linear_offset_rate[:] = 0.0
        self.angular_offset[:] = 0.0
        self.angular_offset_rate[:] = 0.0

    @property
    def is_blending(self) -> bool:
        """Whether any offset is still large enough to be visible."""
        return bool(
            np.abs(self.rotation_offset).max() > 1e-4
            or np.abs(self.linear_offset).max() > 1e-3
        )

    def transition(
        self,
        current_rotations: np.ndarray,
        target_rotations: np.ndarray,
        current_linear: np.ndarray | None = None,
        target_linear: np.ndarray | None = None,
        current_angular: np.ndarray | None = None,
        target_angular: np.ndarray | None = None,
    ) -> None:
        """Record the offset at a jump.

        Called with the pose that *was* playing -- the blended, visible one --
        and the pose about to play. The offset is **assigned**, not added to the
        offset still decaying from a previous jump.

        That distinction is the whole correctness of this class. `current_*` is
        already the blended value, so it *contains* the old offset; back to back
        transitions compose automatically. Adding instead would count the old
        offset twice, doubling it at every jump. At a 10 frame search interval
        that is exponential: the velocity offset reaches millions within a few
        hundred frames, and because the controller's next query is built from
        the resulting positions, the blow-up feeds straight back into the search
        and the character tears off to infinity.

        Args:
            current_rotations: `(n_joints, 3, 3)` local rotations currently
                visible, i.e. after the previous offset was applied.
            target_rotations: `(n_joints, 3, 3)` local rotations about to play.
            current_linear, target_linear: `(3,)` root linear velocity.
            current_angular, target_angular: `(3,)` root angular velocity.
        """
        relative = np.swapaxes(target_rotations, -1, -2) @ current_rotations
        self.rotation_offset = rotations.quat_to_scaled_angle_axis(
            rotations.matrix_to_quat(relative)
        ).astype(np.float32)

        if current_linear is not None and target_linear is not None:
            self.linear_offset = np.asarray(current_linear - target_linear, np.float32)
        if current_angular is not None and target_angular is not None:
            self.angular_offset = np.asarray(current_angular - target_angular, np.float32)

    def apply(
        self,
        source_rotations: np.ndarray,
        source_linear: np.ndarray | None = None,
        source_angular: np.ndarray | None = None,
    ):
        """Add the current offset to the source pose.

        Args:
            source_rotations: `(n_joints, 3, 3)` what the source is playing now.
            source_linear, source_angular: `(3,)` root velocities, optional.

        Returns:
            `(rotations, linear, angular)`, with the unsupplied velocities
            returned as None.
        """
        offset_quat = rotations.scaled_angle_axis_to_quat(self.rotation_offset)
        offset_matrix = rotations.quat_to_matrix(offset_quat)
        blended = source_rotations @ offset_matrix

        linear = None if source_linear is None else (source_linear + self.linear_offset)
        angular = None if source_angular is None else (source_angular + self.angular_offset)
        return blended.astype(np.float32), linear, angular

    def step(self) -> None:
        """Decay every offset by one frame. Call once per frame, after apply."""
        self.rotation_offset, self.rotation_offset_rate = _decay_spring_damper(
            self.rotation_offset, self.rotation_offset_rate, self.rotation_halflife, self.frame_time
        )
        self.linear_offset, self.linear_offset_rate = _decay_spring_damper(
            self.linear_offset, self.linear_offset_rate, self.velocity_halflife, self.frame_time
        )
        self.angular_offset, self.angular_offset_rate = _decay_spring_damper(
            self.angular_offset, self.angular_offset_rate, self.velocity_halflife, self.frame_time
        )
