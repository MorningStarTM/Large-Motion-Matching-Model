"""
World, character and local space conversions.

- World space: relative to the game world.
- Character space: relative to the character's root, with horizontal position
  and facing removed but height kept. This is the model's input normalization:
  every feature and every pose the network sees is in character space.
- Local space: each joint relative to its parent, which is what BVH stores and
  what forward kinematics chains together.

The root is a ground plane frame, not a full 3D transform: an origin on the
floor under the hips and a forward direction in the XZ plane. Pitch and roll
stay in the pose, because leaning is motion the model must reproduce, while
"where the character stands and which way it faces" is exactly what should be
factored out.

Root velocity and its inverse
-----------------------------
`root_velocities` and `integrate_root` are an exact pair: integrating a clip's
own stored velocities must retrace that clip's own path. That is what
`tests/test_root_velocity_roundtrip.py` asserts, and it is the only thing
keeping the runtime's dead reckoning honest.

Forward direction
-----------------
Determined empirically from LaFAN1 resolved rather than assumed: the hips
joint's third basis vector (its local Z axis) projected onto the ground plane
aligns with the direction of travel with a median cosine of 0.99 over fast
walking frames, while its second (Y) axis is vertical with |Y| = 0.985. Hence
`FORWARD_AXIS = 2`, declared once as a constant so a different skeleton only
needs this changed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmm.data import rotations

__all__ = [
    "RootFrame",
    "FORWARD_AXIS",
    "HEADING_SMOOTH_SIGMA",
    "yaw_matrix",
    "strip_yaw",
    "apply_yaw",
    "extract_root_frames",
    "to_character_space",
    "direction_to_character_space",
    "from_character_space",
    "root_velocities",
    "integrate_root",
]

# Which column of the hips rotation matrix points forward. See the module
# docstring: measured, not guessed.
FORWARD_AXIS = 2

# Gaussian sigma, in frames, applied to the heading before it becomes the root
# frame -- the "simulation bone" of Holden's motion matching.
#
# The raw hips heading carries the pelvis's own yaw swing: measured on LaFAN1,
# an oscillation of 4.6 to 8.9 degrees standard deviation about the smooth
# path. That swing is real motion and belongs in the pose, but it must not be
# in the *frame*, because the runtime frame is driven by a spring damper and is
# smooth by construction. A pose authored against a wobbling frame and replayed
# in a smooth one has its feet rotated by the difference, which reads as foot
# sliding.
#
# 10 frames at 60Hz leaves 2.5 degrees of residual oscillation for the pose to
# carry and preserves 98% of the heading a clip genuinely turns through, which
# is the balance measured across walking, running and sprinting clips.
HEADING_SMOOTH_SIGMA = 10.0


@dataclass
class RootFrame:
    """The character's ground plane frame, per frame.

    Attributes:
        origin: `(n_frames, 3)` the point on the floor under the hips. Y is
            always 0: height belongs to the pose, not to the frame.
        forward: `(n_frames, 2)` unit forward direction in the XZ plane.
    """

    origin: np.ndarray
    forward: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.origin.shape[0])

    @property
    def yaw(self) -> np.ndarray:
        """Facing angle in radians, `(n_frames,)`, measured from +Z toward +X.

        Useful for reporting a facing error as an angle; the conversions
        themselves never need it, since they use the forward vector directly
        and so avoid a wrap-around discontinuity.
        """
        return np.arctan2(self.forward[:, 0], self.forward[:, 1])

    def __getitem__(self, index) -> "RootFrame":
        origin = np.atleast_2d(self.origin[index])
        forward = np.atleast_2d(self.forward[index])
        return RootFrame(origin=origin, forward=forward)


def extract_root_frames(
    global_rotations: np.ndarray,
    root_positions: np.ndarray,
    smooth_sigma: float = HEADING_SMOOTH_SIGMA,
) -> RootFrame:
    """Build the ground plane frame from the hips.

    Args:
        global_rotations: `(n_frames, n_joints, 3, 3)`, or `(n_frames, 3, 3)`
            for the hips alone. Only the hips (joint 0) is read.
        root_positions: `(n_frames, 3)` world space root translation.
        smooth_sigma: Gaussian sigma in frames applied to the heading. See
            `HEADING_SMOOTH_SIGMA`. Pass 0 for the raw hips heading, which is
            what the pelvis swing measurements are taken against.

    Returns:
        A `RootFrame`.
    """
    if global_rotations.ndim == 4:
        hips = global_rotations[:, 0]
    else:
        hips = global_rotations

    forward_3d = hips[:, :, FORWARD_AXIS]
    forward = forward_3d[:, [0, 2]]
    norm = np.linalg.norm(forward, axis=-1, keepdims=True)

    # If the hips' forward axis is near vertical the ground projection is
    # undefined -- a character lying down or doing a handstand. Fall back to
    # the previous usable frame rather than emitting a NaN that would silently
    # poison every feature downstream.
    degenerate = norm[:, 0] < 1e-4
    forward = np.where(norm > 1e-4, forward / np.maximum(norm, 1e-12), np.array([0.0, 1.0]))
    if degenerate.any():
        last_good = np.array([0.0, 1.0], dtype=forward.dtype)
        for i in range(len(forward)):
            if degenerate[i]:
                forward[i] = last_good
            else:
                last_good = forward[i]

    if smooth_sigma and len(forward) > 1:
        # Smooth the *unwrapped angle*, not the vector components. Filtering the
        # components and renormalising barely attenuates the swing once the
        # heading itself has turned a long way, because the two components are
        # then far from independent.
        from scipy.ndimage import gaussian_filter1d

        angle = np.unwrap(np.arctan2(forward[:, 0], forward[:, 1]))
        angle = gaussian_filter1d(angle, float(smooth_sigma), mode="nearest")
        forward = np.stack([np.sin(angle), np.cos(angle)], axis=-1)

    origin = root_positions.copy()
    origin[:, 1] = 0.0

    return RootFrame(origin=origin.astype(np.float32), forward=forward.astype(np.float32))


def yaw_matrix(forward: np.ndarray) -> np.ndarray:
    """The rotation about the world up axis that maps +Z onto `forward`.

    Args:
        forward: `(..., 2)` unit forward in XZ.

    Returns:
        `(..., 3, 3)`.
    """
    forward = np.asarray(forward, dtype=np.float32)
    fx = forward[..., 0]
    fz = forward[..., 1]
    zero = np.zeros_like(fx)
    one = np.ones_like(fx)
    rows = [
        [fz, zero, fx],
        [zero, one, zero],
        [-fx, zero, fz],
    ]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def strip_yaw(rotation: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Express a world rotation in the character frame, removing the facing.

    This is what makes a stored pose purely character space: the hips rotation
    keeps its lean and twist but loses "which way the character faces", so the
    world orientation lives only in the integrated root frame and is never
    counted twice.

    Args:
        rotation: `(..., 3, 3)` a world space rotation.
        forward: `(..., 2)` the frame's forward.
    """
    yaw = yaw_matrix(forward)
    return np.swapaxes(yaw, -1, -2) @ rotation


def apply_yaw(rotation: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Inverse of `strip_yaw`: character frame rotation back to world."""
    return yaw_matrix(forward) @ rotation


def _rotate_into_frame(xz: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Rotate XZ vectors so `forward` becomes +Z.

    `forward = (fx, fz)`. The frame's right axis is `(fz, -fx)`, so the
    rotated coordinates are the dot products with right and forward.
    """
    fx = forward[..., 0]
    fz = forward[..., 1]
    x = xz[..., 0] * fz - xz[..., 1] * fx
    z = xz[..., 0] * fx + xz[..., 1] * fz
    return np.stack([x, z], axis=-1)


def _rotate_out_of_frame(xz: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Inverse of `_rotate_into_frame`."""
    fx = forward[..., 0]
    fz = forward[..., 1]
    x = xz[..., 0] * fz + xz[..., 1] * fx
    z = -xz[..., 0] * fx + xz[..., 1] * fz
    return np.stack([x, z], axis=-1)


def to_character_space(points: np.ndarray, frame: RootFrame) -> np.ndarray:
    """World space points to character space.

    Args:
        points: `(n_frames, ..., 3)`.
        frame: the frame of each of those frames.

    Returns:
        Same shape. Horizontal position and facing are removed; Y is unchanged.
    """
    origin = frame.origin.reshape((frame.n_frames,) + (1,) * (points.ndim - 2) + (3,))
    forward = frame.forward.reshape((frame.n_frames,) + (1,) * (points.ndim - 2) + (2,))

    local = points - origin
    xz = _rotate_into_frame(local[..., [0, 2]], forward)
    return np.stack([xz[..., 0], local[..., 1], xz[..., 1]], axis=-1)


def direction_to_character_space(directions: np.ndarray, frame: RootFrame) -> np.ndarray:
    """Like `to_character_space` but without the translation, for vectors such
    as velocities and facing directions."""
    forward = frame.forward.reshape((frame.n_frames,) + (1,) * (directions.ndim - 2) + (2,))
    xz = _rotate_into_frame(directions[..., [0, 2]], forward)
    return np.stack([xz[..., 0], directions[..., 1], xz[..., 1]], axis=-1)


def from_character_space(points: np.ndarray, frame: RootFrame) -> np.ndarray:
    """Character space points back to world space."""
    origin = frame.origin.reshape((frame.n_frames,) + (1,) * (points.ndim - 2) + (3,))
    forward = frame.forward.reshape((frame.n_frames,) + (1,) * (points.ndim - 2) + (2,))

    xz = _rotate_out_of_frame(points[..., [0, 2]], forward)
    world = np.stack([xz[..., 0], points[..., 1], xz[..., 1]], axis=-1)
    return world + origin


def root_velocities(
    frame: RootFrame, hips_rotations: np.ndarray | None, frame_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """The root velocity the model predicts, in character space.

    Expressing it in the *current* frame is what makes it usable at runtime:
    the model never sees a world position, so its output has to be a local
    increment the viewer can integrate.

    Yaw rate is differenced from the **ground frame's forward**, not taken from
    the hips rotation's angular velocity. This is the exact inverse of
    `integrate_root`, and it has to be, or replaying a clip's own stored
    velocities does not reproduce that clip's own path.

    Taking component 1 of the hips rotation's angular velocity vector -- the
    obvious thing, and what this did first -- is wrong twice over. A single
    component of a 3D rotation vector is only the yaw rate when pitch and roll
    are zero, and a running pelvis oscillates in all three every stride.
    Measured on `run1_subject2`, which runs in a straight line to within 0.2
    degrees over ten seconds, that quantity averaged 29.5 deg/s and peaked at
    179 deg/s: pure pelvis wobble. Integrating it swung the facing by +-64
    degrees. Worse, the search jumps every 10 frames and so resamples the
    oscillation at an arbitrary stride phase instead of completing a cycle, so
    the error does not average out -- it becomes a biased random walk and the
    character turns when nothing asked it to.

    Components 0 and 2 are therefore left at zero. The root frame is a ground
    plane frame by construction; the pelvis's real pitch and roll are motion
    the pose must reproduce and they live in the hips rotation, where the loss
    can see them. Leaving them in here as well would both double count them
    and invite the model to steer the frame with wobble.

    Args:
        frame: the root frames.
        hips_rotations: unused, kept so existing callers need no change. The
            yaw rate now comes from `frame.forward`.
        frame_time: seconds per frame.

    Returns:
        `linear` `(n_frames, 3)` in character space units per second, and
        `angular` `(n_frames, 3)` as a rotation vector per second, of which
        only the vertical component is non zero. Both have their last entry
        duplicated, so they align with the pose at the same index and the
        arrays stay the same length.
    """
    n = frame.n_frames

    delta_world = np.diff(frame.origin, axis=0)
    delta_local = _rotate_into_frame(delta_world[:, [0, 2]], frame.forward[:-1])
    linear = np.zeros((n, 3), dtype=np.float32)
    linear[:-1, 0] = delta_local[:, 0] / frame_time
    linear[:-1, 2] = delta_local[:, 1] / frame_time

    angular = np.zeros((n, 3), dtype=np.float32)
    if n > 1:
        angle = np.arctan2(frame.forward[:, 0], frame.forward[:, 1])
        # Wrap the difference, not the angles: a heading crossing +-pi between
        # two frames is a small turn, and differencing the raw angles would
        # report it as a full revolution.
        delta = (np.diff(angle) + np.pi) % (2.0 * np.pi) - np.pi
        angular[:-1, 1] = delta / frame_time

        linear[-1] = linear[-2]
        angular[-1] = angular[-2]

    return linear, angular


def integrate_root(
    origin: np.ndarray,
    forward: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    frame_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one root frame by one step. The inverse of `root_velocities`,
    and what the runtime loop calls every frame.

    Args:
        origin: `(3,)` current floor origin.
        forward: `(2,)` current unit forward in XZ.
        linear_velocity: `(3,)` character space, units per second.
        angular_velocity: `(3,)` rotation vector per second. Only the
            component about the world up axis turns the frame; pitch and roll
            belong to the pose.
        frame_time: seconds per frame.

    Returns:
        The new `(origin, forward)`.
    """
    step = _rotate_out_of_frame(
        np.array([linear_velocity[0], linear_velocity[2]], dtype=np.float32) * frame_time,
        forward,
    )
    new_origin = origin.copy()
    new_origin[0] += step[0]
    new_origin[2] += step[1]
    new_origin[1] = 0.0

    yaw_rate = float(angular_velocity[1])
    angle = yaw_rate * frame_time
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    fx, fz = float(forward[0]), float(forward[1])
    new_forward = np.array([fx * cos_a + fz * sin_a, -fx * sin_a + fz * cos_a], dtype=np.float32)
    new_forward /= max(float(np.linalg.norm(new_forward)), 1e-8)

    return new_origin, new_forward
