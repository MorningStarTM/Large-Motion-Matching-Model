"""
Forward kinematics: chain local joint rotations into 3D joint positions.

FK is differentiable, so it sits inside the loss. The position and velocity
terms of plan section 6.2 are what make the error measurable where the eye sees
it rather than in rotation space.

Two implementations with identical semantics: a NumPy one for data preparation
and the teacher, and a torch one that keeps the graph intact for the losses.
`tests/test_fk.py` checks they agree and that gradients match finite
differences.

The loop runs over joints, not over time or batch: 24 iterations of batched
matrix products, rather than a Python loop over thousands of frames.
"""

from __future__ import annotations

import numpy as np

from mmm.data.skeleton import Skeleton

__all__ = ["forward_kinematics", "forward_kinematics_torch", "joint_velocities"]


def forward_kinematics(
    skeleton: Skeleton,
    local_rotations: np.ndarray,
    root_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy forward kinematics.

    Args:
        skeleton: the joint tree. `skeleton.parents` must be topologically
            sorted, which `Skeleton` enforces on construction -- that is what
            lets a single forward pass work.
        local_rotations: `(..., n_joints, 3, 3)` each joint relative to its
            parent.
        root_positions: `(..., 3)` world root translation. Defaults to the
            origin, which is what the character space variant wants.

    Returns:
        `positions` `(..., n_joints, 3)` and `global_rotations`
        `(..., n_joints, 3, 3)`.
    """
    n_joints = skeleton.n_joints
    if local_rotations.shape[-3] != n_joints:
        raise ValueError(
            f"local_rotations has {local_rotations.shape[-3]} joints, "
            f"skeleton has {n_joints}"
        )

    batch_shape = local_rotations.shape[:-3]
    offsets = skeleton.offsets.astype(local_rotations.dtype)

    global_rotations = np.empty_like(local_rotations)
    positions = np.empty(batch_shape + (n_joints, 3), dtype=local_rotations.dtype)

    if root_positions is None:
        positions[..., 0, :] = 0.0
    else:
        positions[..., 0, :] = root_positions
    global_rotations[..., 0, :, :] = local_rotations[..., 0, :, :]

    for joint in range(1, n_joints):
        parent = int(skeleton.parents[joint])
        parent_rotation = global_rotations[..., parent, :, :]
        global_rotations[..., joint, :, :] = parent_rotation @ local_rotations[..., joint, :, :]
        # The offset is a column vector rotated by the parent's frame.
        rotated = (parent_rotation @ offsets[joint][:, None])[..., 0]
        positions[..., joint, :] = positions[..., parent, :] + rotated

    return positions, global_rotations


def forward_kinematics_torch(
    skeleton: Skeleton,
    local_rotations,
    root_positions=None,
):
    """Torch forward kinematics, differentiable.

    Same arguments and returns as `forward_kinematics`, with tensors. Built by
    appending to lists and stacking once at the end rather than writing into a
    preallocated tensor, because in-place writes into a tensor that requires
    grad break autograd.
    """
    import torch

    n_joints = skeleton.n_joints
    if local_rotations.shape[-3] != n_joints:
        raise ValueError(
            f"local_rotations has {local_rotations.shape[-3]} joints, "
            f"skeleton has {n_joints}"
        )

    device = local_rotations.device
    dtype = local_rotations.dtype
    offsets = torch.as_tensor(skeleton.offsets, device=device, dtype=dtype)

    batch_shape = local_rotations.shape[:-3]
    if root_positions is None:
        root = torch.zeros(batch_shape + (3,), device=device, dtype=dtype)
    else:
        root = root_positions

    rotation_list = [local_rotations[..., 0, :, :]]
    position_list = [root]

    for joint in range(1, n_joints):
        parent = int(skeleton.parents[joint])
        parent_rotation = rotation_list[parent]
        rotation_list.append(parent_rotation @ local_rotations[..., joint, :, :])
        rotated = (parent_rotation @ offsets[joint].reshape(3, 1)).squeeze(-1)
        position_list.append(position_list[parent] + rotated)

    positions = torch.stack(position_list, dim=-2)
    global_rotations = torch.stack(rotation_list, dim=-3)
    return positions, global_rotations


def joint_velocities(positions: np.ndarray, frame_time: float) -> np.ndarray:
    """Per joint velocity from a position sequence, `(n_frames, n_joints, 3)`.

    The first frame has no predecessor, so it copies the second frame's
    velocity rather than being zero: a spurious zero at the clip start would
    show up in the matching features as a stationary foot and attract bad
    retrievals.
    """
    velocities = np.zeros_like(positions)
    velocities[1:] = (positions[1:] - positions[:-1]) / frame_time
    if len(positions) > 1:
        velocities[0] = velocities[1]
    return velocities
