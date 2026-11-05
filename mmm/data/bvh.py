"""
BVH reading and writing. MMM's working format (plan section 14).

Written against LaFAN1 resolved, whose files have three quirks worth knowing
about, all handled here and all verified against the data:

1. **Every joint carries 6 channels**, `Xposition Yposition Zposition
   Zrotation Yrotation Xrotation`, not just the root. For non-root joints the
   position channels are constant and equal to the joint's `OFFSET`, so they
   carry no information. In practice 69 of the 75 joints match their offset
   exactly and the other six sit within about 1% of bone length of it, which
   is residue from Holden's IK resolve; `Motion.position_drift` records the
   worst case and the loader warns only past 2%, which would mean genuine
   per-frame bone scaling.
2. **The root's `OFFSET` duplicates frame 0's position channels.** Adding both
   would double the character's world position, so the root offset is ignored
   and the root's position is taken from the channels alone.
3. **Units are centimetres and the up axis is Y**, which matches raylib and
   GenoViewPython, so no conversion is applied. `scale` is available for data
   that needs it.

Parsing is done by hand rather than with a library: the format is small, and a
dependency here would have to be understood before it could be trusted with
the three quirks above.

Relationship to GenoViewPython
------------------------------
Written from scratch, not adapted from GenoViewPython's `bvh.py`, which the
project's phase plan credits: that repository was unavailable at the time. The
reference parses line by line with regular expressions; this parses the
hierarchy block by block and reads the whole motion section with one
`np.array(block.split(), dtype=np.float32)`, which is the difference between
about 2 seconds and several minutes on a 15,679 frame clip. See NOTICE.md.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass

import numpy as np

from mmm.data import rotations
from mmm.data.skeleton import Skeleton

__all__ = ["Motion", "load", "save", "load_cached"]

# Non-root position channels in LaFAN1 resolved sit within about 1% of bone
# length of the OFFSET (IK solve residue). Warn only beyond this, which would
# mean genuine per-frame bone scaling.
_POSITION_DRIFT_TOLERANCE = 0.02

_ROTATION_CHANNELS = ("Xrotation", "Yrotation", "Zrotation")
_POSITION_CHANNELS = ("Xposition", "Yposition", "Zposition")


@dataclass
class Motion:
    """One parsed clip.

    Attributes:
        skeleton: the joint tree, with rest offsets.
        root_positions: `(n_frames, 3)` root translation in world space.
        rotations: `(n_frames, n_joints, 3, 3)` LOCAL joint rotations, each
            relative to its parent. Matrices rather than euler angles, because
            every consumer wants matrices and converting once here keeps the
            channel order confined to this module.
        frame_time: seconds per frame (1/60 for LaFAN1).
        name: clip name, normally the filename stem.
        channel_order: the rotation channel order the file declared, kept so
            `save` can round trip it.
        position_drift: the largest observed `|position channel - OFFSET|` on a
            non-root joint, as a fraction of that joint's bone length. In
            LaFAN1 resolved this is about 1% on the forearms, legs and neck --
            residual noise from the IK solve, not real bone scaling -- and
            exactly 0 on the other 69 joints. Forward kinematics uses the rest
            offsets, so this is the error that choice introduces.
    """

    skeleton: Skeleton
    root_positions: np.ndarray
    rotations: np.ndarray
    frame_time: float
    name: str = ""
    channel_order: str = "ZYX"
    position_drift: float = 0.0

    @property
    def n_frames(self) -> int:
        return int(self.root_positions.shape[0])

    @property
    def n_joints(self) -> int:
        return self.skeleton.n_joints

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    def __len__(self) -> int:
        return self.n_frames

    def slice(self, start: int, stop: int) -> "Motion":
        return Motion(
            skeleton=self.skeleton,
            root_positions=self.root_positions[start:stop],
            rotations=self.rotations[start:stop],
            frame_time=self.frame_time,
            name=f"{self.name}[{start}:{stop}]",
            channel_order=self.channel_order,
            position_drift=self.position_drift,
        )

    def subset(self, joint_names) -> "Motion":
        """Restrict to a joint subset, e.g. the 24 locomotion joints.

        Dropped intermediate joints have their rotation folded into the
        nearest kept descendant's parent relationship, so the visible pose is
        preserved: the kept joint's local rotation becomes the product along
        the dropped span.
        """
        new_skeleton, source = self.skeleton.subset(joint_names)

        # Global rotations in the original skeleton, so a kept joint's new
        # local rotation can be computed as (parent_global)^-1 @ own_global.
        globals_ = np.empty_like(self.rotations)
        for i in range(self.n_joints):
            p = int(self.skeleton.parents[i])
            if p < 0:
                globals_[:, i] = self.rotations[:, i]
            else:
                globals_[:, i] = globals_[:, p] @ self.rotations[:, i]

        kept_globals = globals_[:, source]
        new_rotations = np.empty_like(kept_globals)
        for j in range(new_skeleton.n_joints):
            p = int(new_skeleton.parents[j])
            if p < 0:
                new_rotations[:, j] = kept_globals[:, j]
            else:
                parent_inv = np.swapaxes(kept_globals[:, p], -1, -2)
                new_rotations[:, j] = parent_inv @ kept_globals[:, j]

        return Motion(
            skeleton=new_skeleton,
            root_positions=self.root_positions,
            rotations=new_rotations,
            frame_time=self.frame_time,
            name=self.name,
            channel_order=self.channel_order,
            position_drift=self.position_drift,
        )

    def resample(self, target_fps: float) -> "Motion":
        """Resample to `target_fps`. Rotations are interpolated as quaternions,
        never as euler angles or matrices, both of which produce nonsense
        between keyframes."""
        if abs(self.fps - target_fps) < 1e-6:
            return self

        src_times = np.arange(self.n_frames) * self.frame_time
        duration = src_times[-1]
        n_target = int(np.floor(duration * target_fps)) + 1
        dst_times = np.arange(n_target) / target_fps

        idx = np.clip(np.searchsorted(src_times, dst_times, side="right") - 1, 0, self.n_frames - 2)
        span = src_times[idx + 1] - src_times[idx]
        alpha = ((dst_times - src_times[idx]) / np.maximum(span, 1e-12)).astype(np.float32)

        root = (1.0 - alpha)[:, None] * self.root_positions[idx] + alpha[:, None] * self.root_positions[idx + 1]

        q0 = rotations.matrix_to_quat(self.rotations[idx])
        q1 = rotations.matrix_to_quat(self.rotations[idx + 1])
        # Shortest arc, then normalized linear interpolation. nlerp rather than
        # slerp: between adjacent 60fps frames the angle is tiny and the two are
        # indistinguishable, while nlerp has no small-angle division.
        flip = np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0
        q1 = np.where(flip, -q1, q1)
        a = alpha[:, None, None]
        q = rotations.quat_normalize((1.0 - a) * q0 + a * q1)

        return Motion(
            skeleton=self.skeleton,
            root_positions=root.astype(np.float32),
            rotations=rotations.quat_to_matrix(q).astype(np.float32),
            frame_time=1.0 / target_fps,
            name=self.name,
            channel_order=self.channel_order,
            position_drift=self.position_drift,
        )

    def __repr__(self) -> str:
        return (
            f"Motion({self.name!r}, {self.n_frames} frames @ {self.fps:.1f}fps, "
            f"{self.n_joints} joints)"
        )


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

@dataclass
class _JointSpec:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str]


def _parse_hierarchy(lines, start: int):
    """Parse the HIERARCHY block. Returns (joints, index of the MOTION line).

    End Site blocks are read and discarded: they are leaf bone lengths, not
    joints, and carry no channels. Keeping them would change joint indexing
    for no gain.
    """
    joints: list[_JointSpec] = []
    stack: list[int] = []
    i = start
    pending_name: str | None = None
    pending_is_end_site = False

    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue

        if line == "MOTION":
            return joints, i - 1

        tokens = line.split()
        head = tokens[0]

        if head in ("ROOT", "JOINT"):
            pending_name = tokens[1]
            pending_is_end_site = False
        elif head == "End":
            pending_name = None
            pending_is_end_site = True
        elif head == "{":
            if pending_is_end_site:
                stack.append(-1)  # placeholder so the matching '}' pops cleanly
            else:
                parent = stack[-1] if stack else -1
                joints.append(
                    _JointSpec(name=pending_name, parent=parent, offset=np.zeros(3, np.float32), channels=[])
                )
                stack.append(len(joints) - 1)
        elif head == "}":
            if not stack:
                raise ValueError(f"unbalanced '}}' at line {i}")
            stack.pop()
        elif head == "OFFSET":
            if stack and stack[-1] >= 0:
                joints[stack[-1]].offset = np.array(tokens[1:4], dtype=np.float32)
        elif head == "CHANNELS":
            if stack and stack[-1] >= 0:
                count = int(tokens[1])
                channels = tokens[2 : 2 + count]
                if len(channels) != count:
                    raise ValueError(
                        f"CHANNELS declared {count} but listed {len(channels)} at line {i}"
                    )
                joints[stack[-1]].channels = channels
        elif head == "HIERARCHY":
            continue
        else:
            raise ValueError(f"unexpected token {head!r} in HIERARCHY at line {i}")

    raise ValueError("reached end of file without finding MOTION")


def _rotation_order(channels: list[str]) -> str:
    """The rotation channel order as a string such as 'ZYX'.

    In BVH the channel order is the multiplication order, so this string is
    passed straight to `rotations.euler_to_matrix`.
    """
    order = "".join(c[0].upper() for c in channels if c in _ROTATION_CHANNELS)
    if len(order) != 3:
        raise ValueError(f"expected 3 rotation channels, got {channels}")
    return order


def load(path: str, scale: float = 1.0, name: str | None = None) -> Motion:
    """Parse a BVH file.

    Args:
        path: file to read.
        scale: multiplies every length (offsets and root positions). Leave at
            1.0 for LaFAN1, which is already in the viewer's units.
        name: clip name; defaults to the filename stem.

    Returns:
        A `Motion` with local rotations as matrices.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    lines = text.splitlines()
    joints, motion_line = _parse_hierarchy(lines, 0)
    if not joints:
        raise ValueError(f"{path}: no joints found")

    # -- MOTION header --
    frames = frame_time = None
    cursor = motion_line + 1
    while cursor < len(lines):
        line = lines[cursor].strip()
        cursor += 1
        if not line:
            continue
        if line.lower().startswith("frames:"):
            frames = int(line.split(":")[1])
        elif line.lower().startswith("frame time:"):
            frame_time = float(line.split(":")[1])
        else:
            cursor -= 1
            break
    if frames is None or frame_time is None:
        raise ValueError(f"{path}: MOTION block missing 'Frames:' or 'Frame Time:'")

    # -- MOTION data. One flat parse of the whole block is far faster than
    # np.loadtxt, which is minutes rather than seconds on a 15k frame clip. --
    n_channels = sum(len(j.channels) for j in joints)
    block = "\n".join(lines[cursor:])
    values = np.array(block.split(), dtype=np.float32)

    expected = frames * n_channels
    if values.size < expected:
        raise ValueError(
            f"{path}: MOTION block has {values.size} numbers, expected "
            f"{expected} ({frames} frames x {n_channels} channels)"
        )
    if values.size > expected:
        # Some exporters append a trailing partial line; trust the header.
        values = values[:expected]
    channel_data = values.reshape(frames, n_channels)

    # -- split channels per joint --
    n_joints = len(joints)
    root_positions = np.zeros((frames, 3), dtype=np.float32)
    euler = np.zeros((frames, n_joints, 3), dtype=np.float32)
    bone_lengths = np.linalg.norm([j.offset for j in joints], axis=-1)
    position_drift = 0.0

    offset = 0
    for j, spec in enumerate(joints):
        for c, channel in enumerate(spec.channels):
            column = channel_data[:, offset + c]
            if channel in _POSITION_CHANNELS:
                axis = _POSITION_CHANNELS.index(channel)
                if j == 0:
                    # Quirk 2: the root's OFFSET duplicates frame 0, so the
                    # channels are the whole story and the offset is not added.
                    root_positions[:, axis] = column
                else:
                    # Quirk 1: these are constant and equal to OFFSET, up to
                    # the IK solver's residual noise. Measure it relative to
                    # bone length rather than asserting it is exactly zero.
                    bone = bone_lengths[j]
                    if bone > 1e-6:
                        deviation = float(np.abs(column - spec.offset[axis]).max())
                        position_drift = max(position_drift, deviation / bone)
            elif channel in _ROTATION_CHANNELS:
                axis = _ROTATION_CHANNELS.index(channel)
                euler[:, j, axis] = column
            else:
                raise ValueError(f"{path}: unknown channel {channel!r} on joint {spec.name}")
        offset += len(spec.channels)

    if position_drift > _POSITION_DRIFT_TOLERANCE:
        warnings.warn(
            f"{os.path.basename(path)}: non-root joint position channels differ "
            f"from their OFFSET by up to {position_drift:.1%} of bone length, "
            f"above the {_POSITION_DRIFT_TOLERANCE:.0%} tolerance, so this "
            "skeleton really is scaling over time. Forward kinematics uses the "
            "rest offsets and will ignore that.",
            stacklevel=2,
        )

    order = _rotation_order(joints[0].channels)
    for spec in joints[1:]:
        if _rotation_order(spec.channels) != order:
            raise ValueError(
                f"{path}: joint {spec.name!r} has rotation order "
                f"{_rotation_order(spec.channels)}, but the root has {order}. "
                "Mixed orders within one file are not supported."
            )

    # euler is stored as (X, Y, Z) per joint; reorder to channel order for the
    # conversion, since `euler_to_matrix` expects angle i to belong to order[i].
    axis_for = {"X": 0, "Y": 1, "Z": 2}
    ordered = np.stack([euler[..., axis_for[a]] for a in order], axis=-1)
    matrices = rotations.euler_to_matrix(ordered, order=order)

    skeleton = Skeleton(
        names=[j.name for j in joints],
        parents=np.array([j.parent for j in joints], dtype=np.int32),
        offsets=np.stack([j.offset for j in joints]) * scale,
    )

    return Motion(
        skeleton=skeleton,
        root_positions=root_positions * scale,
        rotations=matrices.astype(np.float32),
        frame_time=frame_time,
        name=name if name is not None else os.path.splitext(os.path.basename(path))[0],
        channel_order=order,
        position_drift=position_drift,
    )


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def save(path: str, motion: Motion) -> None:
    """Write a `Motion` back to BVH.

    Writes 6 channels on the root and 3 on every other joint, which is the
    common convention and is enough to reproduce the pose. A file written here
    and read back gives the same skeleton, frame time and FK joint positions as
    the original, which is what `tests/test_bvh_roundtrip.py` checks.
    """
    skeleton = motion.skeleton
    order = motion.channel_order
    children = skeleton.children()

    axis_for = {"X": 0, "Y": 1, "Z": 2}
    euler_ordered = rotations.matrix_to_euler(motion.rotations, order=order)
    # Back from channel order to (X, Y, Z) storage, then out in channel order.
    euler_xyz = np.zeros_like(euler_ordered)
    for i, a in enumerate(order):
        euler_xyz[..., axis_for[a]] = euler_ordered[..., i]

    lines: list[str] = ["HIERARCHY"]

    def write_joint(index: int, depth: int) -> None:
        pad = "\t" * depth
        keyword = "ROOT" if depth == 0 else "JOINT"
        lines.append(f"{pad}{keyword} {skeleton.names[index]}")
        lines.append(f"{pad}{{")
        ox, oy, oz = skeleton.offsets[index]
        lines.append(f"{pad}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        rotation_channels = " ".join(f"{a}rotation" for a in order)
        if depth == 0:
            lines.append(f"{pad}\tCHANNELS 6 Xposition Yposition Zposition {rotation_channels}")
        else:
            lines.append(f"{pad}\tCHANNELS 3 {rotation_channels}")
        if children[index]:
            for child in children[index]:
                write_joint(child, depth + 1)
        else:
            # A leaf needs an End Site to give the last bone a length.
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t{{")
            lines.append(f"{pad}\t\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{pad}\t}}")
        lines.append(f"{pad}}}")

    write_joint(0, 0)

    lines.append("MOTION")
    lines.append(f"Frames: {motion.n_frames}")
    lines.append(f"Frame Time: {motion.frame_time:.6f}")

    # Column order must match the CHANNELS declarations above, which follow the
    # same depth-first joint order as the hierarchy.
    dfs_order: list[int] = []

    def walk(index: int) -> None:
        dfs_order.append(index)
        for child in children[index]:
            walk(child)

    walk(0)

    for frame in range(motion.n_frames):
        row: list[str] = []
        for position_in_dfs, joint in enumerate(dfs_order):
            if position_in_dfs == 0:
                row.extend(f"{v:.6f}" for v in motion.root_positions[frame])
            angles = [euler_xyz[frame, joint, axis_for[a]] for a in order]
            row.extend(f"{v:.6f}" for v in angles)
        lines.append(" ".join(row))

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

def load_cached(path: str, cache_dir: str, scale: float = 1.0, rebuild: bool = False) -> Motion:
    """`load`, with the parsed result cached as .npz.

    Parsing 77 LaFAN1 clips takes minutes and is pure overhead after the first
    time. The cache key includes the source file's size and modification time,
    so editing a BVH invalidates its cache rather than silently reusing it.
    """
    os.makedirs(cache_dir, exist_ok=True)
    stat = os.stat(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    key = f"{stem}_{int(stat.st_mtime)}_{stat.st_size}_{scale:g}"
    cache_path = os.path.join(cache_dir, f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}.npz")

    if not rebuild and os.path.exists(cache_path):
        blob = np.load(cache_path, allow_pickle=False)
        skeleton = Skeleton(
            names=[str(n) for n in blob["joint_names"]],
            parents=blob["parents"],
            offsets=blob["offsets"],
        )
        return Motion(
            skeleton=skeleton,
            root_positions=blob["root_positions"],
            rotations=blob["rotations"],
            frame_time=float(blob["frame_time"]),
            name=str(blob["name"]),
            channel_order=str(blob["channel_order"]),
            position_drift=float(blob["position_drift"]),
        )

    motion = load(path, scale=scale)
    np.savez_compressed(
        cache_path,
        joint_names=np.array(motion.skeleton.names),
        parents=motion.skeleton.parents,
        offsets=motion.skeleton.offsets,
        root_positions=motion.root_positions,
        rotations=motion.rotations,
        frame_time=np.float32(motion.frame_time),
        name=motion.name,
        channel_order=motion.channel_order,
        position_drift=np.float32(motion.position_drift),
    )
    return motion
