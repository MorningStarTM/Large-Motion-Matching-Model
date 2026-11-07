"""
Foot contact labels that also work above the floor.

`mmm.data.contacts.detect_contacts` decides a foot is planted from its height
and speed, with `height_off = 7 cm`. That is correct for level ground and
**structurally unable** to label a foot standing on anything taller. Measured
on LaFAN1:

    walk1_subject1         0.0% of frames have a foot high and still
    obstacles1_subject1   44.5%
    obstacles5_subject2   30.4%

So across the seventeen obstacle clips, roughly 40 per cent of frames have a
foot genuinely planted on a prop and labelled "not in contact". The model is
being taught that standing on a box is not contact, and any loss term gated on
those labels inherits the mistake.

The fix, and why it needs no knowledge of the ground
----------------------------------------------------
A foot that is **still for long enough** is bearing weight, whatever it is
standing on. That test needs no floor height, no obstacle geometry and no
per-dataset tuning, so it works on LaFAN1's studio props, on a staircase, and
on a dataset we have never seen.

The thresholds are the ones already validated in `mmm.obstacles.recover`,
where this exact detector was used to recover the physical props from the
motion. Its false positive rate was measured there: over 34 non-obstacle clips
it produced **16 stray supports and 3 spurious surfaces in total**, against 94
to 178 supports per obstacle clip. Feet only -- including hands made it invent
surfaces at 85 to 96 cm, which is simply the height a hand hangs at mid-stride.

The two rules are OR'd. Level-ground contact keeps exactly the labels it had,
so nothing that depended on them changes; elevated contact is added on top.

This is a separate module rather than a change to `contacts.py` because that
detector is correct for what it was written to do, is used by the viewer and
the sliding metric, and has tests pinning its behaviour. Nothing here modifies
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmm.data.contacts import ContactThresholds, detect_contacts
from mmm.data.skeleton import LEFT_FOOT, RIGHT_FOOT, Skeleton

__all__ = ["ElevatedThresholds", "detect_contacts_any_height", "contact_breakdown"]

# The same joints `contacts.py` uses: toes, not ankles. The Foot joint sits
# inside the ankle, 6.7 to 8.8 cm above a surface it is flat on.
CONTACT_JOINTS = ("LeftToeBase", "RightToeBase")


@dataclass
class ElevatedThresholds:
    """What counts as weight-bearing when height cannot be used.

    Every default is the value validated in `mmm.obstacles.recover`; see that
    module for the measurements behind each.

    Attributes:
        speed: total joint speed below which a foot may be bearing weight,
            in units/second. A quarter of `ContactThresholds.speed_on`,
            because this wants genuinely planted rather than merely slow.
        vertical_speed: the vertical component must also stay below this.
            This is what rejects the apex of a jump, where the total speed
            dips for a few frames while the body is still arcing through.
        min_frames: how long it must hold. At 60 Hz, 10 frames is a sixth of
            a second -- longer than any apex, shorter than any real step onto
            a surface.
        max_height: above this, a stationary foot is not standing on
            anything. 190 cm clears the tallest surface recovered from
            LaFAN1 (181 cm, the top of a staircase).
    """

    speed: float = 20.0
    vertical_speed: float = 12.0
    min_frames: int = 10
    max_height: float = 190.0


def _sustained(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Keep only runs of True at least `min_frames` long.

    The minimum duration is the whole defence against false positives, so it
    is applied to runs rather than to individual frames.
    """
    out = np.zeros_like(mask, dtype=bool)
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.diff(padded.astype(np.int8))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= min_frames:
            out[start:stop] = True
    return out


def detect_contacts_any_height(
    joint_positions: np.ndarray,
    skeleton: Skeleton,
    frame_time: float = 1.0 / 60.0,
    thresholds: ContactThresholds | None = None,
    elevated: ElevatedThresholds | None = None,
    joints=None,
) -> np.ndarray:
    """Foot contacts, including feet planted on raised surfaces.

    The union of two tests:

    1. `mmm.data.contacts.detect_contacts` -- low and slow, with hysteresis.
       Unchanged, so every label it produced is still produced.
    2. still for a sustained stretch, at any height.

    Args:
        joint_positions: `(n_frames, n_joints, 3)`, ground near y = 0.
        skeleton: for the joint indices.
        frame_time: seconds per frame.
        thresholds: level-ground thresholds, passed straight through.
        elevated: the sustained-stillness thresholds.
        joints: which joints, left then right. Defaults to the toes.

    Returns:
        `(n_frames, 2)` booleans, left then right.
    """
    elevated = elevated or ElevatedThresholds()
    base = detect_contacts(
        joint_positions, skeleton, frame_time, thresholds, joints
    )

    names = joints if joints is not None else CONTACT_JOINTS
    indices = [
        skeleton.index(name) if skeleton.has(name) else skeleton.index(fallback)
        for name, fallback in zip(names, (LEFT_FOOT, RIGHT_FOOT))
    ]

    # The clip's own floor, from the lowest the toes ever get. A percentile
    # rather than the minimum: one frame of IK noise can put a toe a
    # centimetre through the floor and would shift every height in the clip.
    heights = np.concatenate([joint_positions[:, j, 1] for j in indices])
    floor = float(np.percentile(heights, 1.0))

    out = base.copy()
    for column, joint in enumerate(indices):
        track = joint_positions[:, joint]
        if len(track) < 2:
            continue

        step = np.diff(track, axis=0) / frame_time
        speed = np.empty(len(track), np.float32)
        speed[1:] = np.linalg.norm(step, axis=-1)
        speed[0] = speed[1]

        vertical = np.empty(len(track), np.float32)
        vertical[1:] = np.abs(step[:, 1])
        vertical[0] = vertical[1]

        still = (
            (speed < elevated.speed)
            & (vertical < elevated.vertical_speed)
            & ((track[:, 1] - floor) <= elevated.max_height)
        )
        out[:, column] |= _sustained(still, elevated.min_frames)

    return out


def contact_breakdown(
    joint_positions: np.ndarray,
    skeleton: Skeleton,
    frame_time: float = 1.0 / 60.0,
    elevation: float = 15.0,
) -> dict:
    """How many labels each rule contributed, for checking the change.

    A relabelling that silently doubles the positive rate would shift the
    contact loss and the no-skill baseline without anyone noticing, so the
    numbers are reported rather than assumed.
    """
    base = detect_contacts(joint_positions, skeleton, frame_time)
    both = detect_contacts_any_height(joint_positions, skeleton, frame_time)
    added = both & ~base

    indices = [skeleton.index(n) for n in CONTACT_JOINTS if skeleton.has(n)]
    heights = np.concatenate([joint_positions[:, j, 1] for j in indices])
    floor = float(np.percentile(heights, 1.0))
    toe = np.stack([joint_positions[:, j, 1] - floor for j in indices], axis=-1)

    high = added & (toe > elevation)
    return {
        "frames": int(len(base)),
        "base_fraction": float(base.any(axis=-1).mean()),
        "combined_fraction": float(both.any(axis=-1).mean()),
        "added_fraction": float(added.any(axis=-1).mean()),
        "added_above_elevation": float(high.any(axis=-1).mean()),
        "added_median_height": float(np.median(toe[added])) if added.any() else 0.0,
    }
