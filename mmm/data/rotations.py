"""
Rotation format conversions.

6D rotation, the first two columns of a rotation matrix, is the network facing
format: it is continuous, which is what neural networks need. Everything the
model reads or writes is 6D; everything on disk is euler angles.

Every function here works on both NumPy arrays and torch tensors. The data
pipeline uses NumPy and the losses use torch, and duplicating this maths in two
places is how the two silently drift apart. The dispatch is deliberately tiny:
`_xp` returns the array module, and only operations present in both namespaces
under the same name are used.

Conventions
-----------
- Rotation matrices are applied to column vectors: `p' = R @ p`.
- Euler angles are in degrees, and an order string such as "ZYX" means the
  matrix is the product in that order, `R = Rz @ Ry @ Rx`. That is the BVH
  convention: the channel order in the file is the multiplication order.
- Quaternions are `(w, x, y, z)` with the scalar first.
- 6D vectors are the first two columns stacked, `[m00, m10, m20, m01, m11, m21]`.

Relationship to GenoViewPython
------------------------------
Written from scratch against rotation matrices, not adapted from
GenoViewPython's `quat.py`, which works in quaternions. The project's phase
plan describes this module as adapted from it; it is not, because that
repository was unavailable at the time. Instead the two are cross-validated in
`tests/test_rotations_vs_reference.py`, where a vendored copy of `quat.py`
(MIT, Copyright (c) 2024 Daniel Holden) serves as an independent reference and
agrees to 1e-9 or better on every shared conversion. See NOTICE.md.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "euler_to_matrix",
    "matrix_to_euler",
    "matrix_to_sixd",
    "sixd_to_matrix",
    "matrix_to_quat",
    "quat_to_matrix",
    "quat_mul",
    "quat_inv",
    "quat_rotate",
    "quat_to_scaled_angle_axis",
    "scaled_angle_axis_to_quat",
    "quat_normalize",
    "quat_between_vectors",
]


def _xp(x):
    """Return the array module (`numpy` or `torch`) that owns `x`."""
    if isinstance(x, np.ndarray):
        return np
    # Imported lazily so this module stays usable in Phase 0, which has no
    # torch dependency at all.
    import torch

    if isinstance(x, torch.Tensor):
        return torch
    return np


def _zeros_like_scalar(x, value):
    """An array of `value` shaped like `x`, in `x`'s module and dtype."""
    xp = _xp(x)
    if xp is np:
        return np.full_like(x, value)
    return xp.full_like(x, value)


def _cross(a, b):
    """Cross product over the last axis, written out so it works in both
    NumPy and torch without their differing `cross` signatures."""
    xp = _xp(a)
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack(
        [a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1
    ) if xp is np else xp.stack(
        [a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], dim=-1
    )


def _stack(xp, arrays, axis=-1):
    """`stack` with the keyword each module actually accepts."""
    if xp is np:
        return np.stack(arrays, axis=axis)
    return xp.stack(arrays, dim=axis)


def _norm(x, axis=-1, keepdims=True):
    xp = _xp(x)
    if xp is np:
        return np.linalg.norm(x, axis=axis, keepdims=keepdims)
    return xp.linalg.norm(x, dim=axis, keepdim=keepdims)


# ---------------------------------------------------------------------------
# euler <-> matrix
# ---------------------------------------------------------------------------

def _axis_matrix(axis: str, angle):
    """Rotation matrix about one coordinate axis. `angle` is in radians and may
    have any leading shape; the result is `(..., 3, 3)`."""
    xp = _xp(angle)
    c = xp.cos(angle)
    s = xp.sin(angle)
    one = _zeros_like_scalar(angle, 1.0)
    zero = _zeros_like_scalar(angle, 0.0)

    if axis == "X":
        rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
    elif axis == "Y":
        rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
    elif axis == "Z":
        rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
    else:
        raise ValueError(f"axis must be one of X, Y, Z; got {axis!r}")

    return _stack(xp, [_stack(xp, row, axis=-1) for row in rows], axis=-2)


def euler_to_matrix(angles_deg, order: str = "ZYX"):
    """Euler angles to rotation matrices.

    Args:
        angles_deg: `(..., 3)`. Angle `i` belongs to axis `order[i]`.
        order: e.g. "ZYX", meaning `R = Rz @ Ry @ Rx`. This is the BVH
            convention: pass the channel order exactly as the file lists it.

    Returns:
        `(..., 3, 3)`.
    """
    if len(order) != 3 or set(order) != {"X", "Y", "Z"}:
        raise ValueError(f"order must be a permutation of XYZ; got {order!r}")

    xp = _xp(angles_deg)
    radians = angles_deg * (np.pi / 180.0)

    result = None
    for i, axis in enumerate(order):
        m = _axis_matrix(axis, radians[..., i])
        result = m if result is None else (result @ m)
    return result


def matrix_to_euler(matrix, order: str = "ZYX"):
    """Rotation matrices to euler angles in degrees, inverting
    `euler_to_matrix` for the same `order`.

    Only "ZYX" is implemented, because that is the channel order LaFAN1
    resolved uses and a half-correct general solver is worse than an explicit
    refusal. Add other orders when a dataset actually needs one.
    """
    if order != "ZYX":
        raise NotImplementedError(
            f"matrix_to_euler currently supports order='ZYX' only; got {order!r}"
        )

    xp = _xp(matrix)
    # Writing out R = Rz(z) @ Ry(y) @ Rx(x) gives
    #   row 0: [ cz cy,  -sz cx + cz sy sx,   sz sx + cz sy cx ]
    #   row 1: [ sz cy,   cz cx + sz sy sx,  -cz sx + sz sy cx ]
    #   row 2: [   -sy,              cy sx,              cy cx ]
    # so the three angles come straight off the last row and first column:
    #   y from R[2,0] = -sin(y)
    #   z from R[0,0], R[1,0]  (both scaled by cos(y))
    #   x from R[2,1], R[2,2]  (both scaled by cos(y))
    m20 = xp.clip(matrix[..., 2, 0], -1.0, 1.0)
    y = xp.arcsin(-m20)
    z = xp.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    x = xp.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])

    # Gimbal lock: cos(y) == 0 zeroes all four entries used above, leaving z
    # and x degenerate -- only their combination is determined. Pin z to 0 and
    # fold the whole rotation into x, the usual convention, which keeps the
    # round trip exact rather than merely close. With z = 0 and sy = -m20 the
    # first column of row 0/1 is unusable but R[0,1] = sy*sx and R[1,1] = cx.
    cos_y_small = xp.abs(xp.cos(y)) < 1e-6
    z = xp.where(cos_y_small, _zeros_like_scalar(z, 0.0), z)
    x = xp.where(
        cos_y_small,
        xp.arctan2(-m20 * matrix[..., 0, 1], matrix[..., 1, 1]),
        x,
    )

    return _stack(xp, [z, y, x], axis=-1) * (180.0 / np.pi)


# ---------------------------------------------------------------------------
# 6D <-> matrix
# ---------------------------------------------------------------------------

def matrix_to_sixd(matrix):
    """Rotation matrices `(..., 3, 3)` to 6D `(..., 6)`: the first two columns.

    The third column is dropped because it is recoverable as the cross product
    of the other two, which is exactly what makes this representation
    continuous.
    """
    xp = _xp(matrix)
    col0 = matrix[..., :, 0]
    col1 = matrix[..., :, 1]
    if xp is np:
        return np.concatenate([col0, col1], axis=-1)
    return xp.cat([col0, col1], dim=-1)


def sixd_to_matrix(sixd):
    """6D `(..., 6)` to rotation matrices `(..., 3, 3)`, by Gram-Schmidt.

    A raw network output is not orthonormal, so this re-orthonormalises it:
    the result is always a valid rotation, whatever the input. That is why
    every head calls this before FK rather than leaving it to the loss.
    """
    xp = _xp(sixd)
    a0 = sixd[..., 0:3]
    a1 = sixd[..., 3:6]

    b0 = a0 / xp.clip(_norm(a0), 1e-8, None)
    # Remove b0's component from a1, so the two are orthogonal.
    dot = (b0 * a1).sum(axis=-1, keepdims=True) if xp is np else (b0 * a1).sum(dim=-1, keepdim=True)
    a1 = a1 - dot * b0
    b1 = a1 / xp.clip(_norm(a1), 1e-8, None)
    b2 = _cross(b0, b1)

    return _stack(xp, [b0, b1, b2], axis=-1)


# ---------------------------------------------------------------------------
# quaternions -- used by inertialization, which needs to decay a rotation
# offset smoothly and cannot do that in matrix or euler form
# ---------------------------------------------------------------------------

def quat_normalize(q):
    xp = _xp(q)
    return q / xp.clip(_norm(q), 1e-8, None)


def quat_mul(a, b):
    """Hamilton product `a * b`, both `(..., 4)` as `(w, x, y, z)`."""
    xp = _xp(a)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return _stack(
        xp,
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_inv(q):
    """Inverse of a unit quaternion: negate the vector part."""
    xp = _xp(q)
    return _stack(xp, [q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_rotate(q, v):
    """Rotate vectors `v` `(..., 3)` by unit quaternions `q` `(..., 4)`."""
    xp = _xp(q)
    qv = q[..., 1:4]
    w = q[..., 0:1]
    t = 2.0 * _cross(qv, v)
    return v + w * t + _cross(qv, t)


def quat_to_matrix(q):
    xp = _xp(q)
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return _stack(xp, [_stack(xp, r, axis=-1) for r in rows], axis=-2)


def matrix_to_quat(matrix):
    """Rotation matrices to quaternions, via the branch with the largest
    denominator so the division never loses precision."""
    xp = _xp(matrix)
    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    # The four candidate reconstructions, one per component being largest.
    def _safe_sqrt(v):
        return xp.sqrt(xp.clip(v, 1e-12, None))

    qw = _safe_sqrt(1.0 + m00 + m11 + m22) * 0.5
    qx = _safe_sqrt(1.0 + m00 - m11 - m22) * 0.5
    qy = _safe_sqrt(1.0 - m00 + m11 - m22) * 0.5
    qz = _safe_sqrt(1.0 - m00 - m11 + m22) * 0.5

    cand_w = _stack(xp, [qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)], axis=-1)
    cand_x = _stack(xp, [(m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)], axis=-1)
    cand_y = _stack(xp, [(m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)], axis=-1)
    cand_z = _stack(xp, [(m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz], axis=-1)

    trace = m00 + m11 + m22
    use_w = (trace > 0)[..., None]
    x_largest = ((m00 > m11) & (m00 > m22))[..., None]
    y_largest = (m11 > m22)[..., None]

    result = xp.where(y_largest, cand_y, cand_z)
    result = xp.where(x_largest, cand_x, result)
    result = xp.where(use_w, cand_w, result)
    return quat_normalize(result)


def quat_to_scaled_angle_axis(q, eps: float = 1e-8):
    """Unit quaternion to a rotation vector: the axis scaled by the angle in
    radians.

    This is the form inertialization decays: a rotation offset in this space
    can be multiplied by a scalar, which is exactly what a spring needs and
    what neither matrices nor quaternions allow directly.
    """
    xp = _xp(q)
    q = quat_normalize(q)
    # Keep the shorter arc, so a 359 degree offset is treated as -1 degree.
    sign = xp.where(q[..., 0:1] < 0.0, _zeros_like_scalar(q[..., 0:1], -1.0),
                    _zeros_like_scalar(q[..., 0:1], 1.0))
    q = q * sign

    w = xp.clip(q[..., 0:1], -1.0, 1.0)
    v = q[..., 1:4]
    v_norm = _norm(v)

    # result = axis * angle = 2 * v * arctan2(|v|, w) / |v|.
    #
    # Written as that single factor rather than as a normalised axis times an
    # angle. Normalising first needs a clipped divisor, and once the clip binds
    # the "axis" is no longer unit, so the product comes out scaled by
    # |v| / eps instead of being merely imprecise: at a rotation vector of
    # 1e-9 rad this returned 5e-13, three orders of magnitude small, and an
    # earlier comment here asserted the angle factor made that harmless. It
    # does not. The quotient has a finite limit of 1/w as |v| -> 0, so the
    # small case needs no division at all.
    small = v_norm < eps
    quotient = xp.arctan2(v_norm, xp.clip(w, eps, None)) / xp.clip(v_norm, eps, None)
    factor = xp.where(small, _zeros_like_scalar(v_norm, 1.0), quotient)
    return 2.0 * v * factor


def scaled_angle_axis_to_quat(v, eps: float = 1e-8):
    """Inverse of `quat_to_scaled_angle_axis`."""
    xp = _xp(v)
    angle = _norm(v)
    half = angle * 0.5
    w = xp.cos(half)

    # xyz = axis * sin(angle/2) = v * sin(angle/2) / angle, whose limit as
    # angle -> 0 is v / 2. Taken as one factor for the same reason as in
    # `quat_to_scaled_angle_axis`: normalising the axis through a clipped
    # divisor silently rescales the result for angles below `eps` rather than
    # just rounding it.
    small = angle < eps
    quotient = xp.sin(half) / xp.clip(angle, eps, None)
    factor = xp.where(small, _zeros_like_scalar(angle, 0.5), quotient)
    xyz = v * factor
    if xp is np:
        return np.concatenate([w, xyz], axis=-1)
    return xp.cat([w, xyz], dim=-1)


def quat_between_vectors(a, b):
    """The shortest rotation taking unit vector `a` onto unit vector `b`."""
    xp = _xp(a)
    a = a / xp.clip(_norm(a), 1e-8, None)
    b = b / xp.clip(_norm(b), 1e-8, None)
    dot = (a * b).sum(axis=-1, keepdims=True) if xp is np else (a * b).sum(dim=-1, keepdim=True)
    axis = _cross(a, b)
    w = 1.0 + dot
    if xp is np:
        q = np.concatenate([w, axis], axis=-1)
    else:
        q = xp.cat([w, axis], dim=-1)
    return quat_normalize(q)
