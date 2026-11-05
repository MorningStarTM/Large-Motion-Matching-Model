"""
Foot contact labelling.

A contact label says a foot is planted. Foot sliding, a planted foot moving
along the ground, is the most visible animation bug, so these labels drive both
the optional contact loss and the RL foot sliding penalty.

Detection is height plus speed with hysteresis. The hysteresis matters: a
single threshold makes a foot flicker in and out of contact from frame to frame
around the boundary, and a flickering label teaches the model that contact is
noise.

Detection uses the **toe** joints, not the ankles. Measured on LaFAN1 resolved,
the `Foot` joint sits 6.7 to 8.8 cm above the ground even when firmly planted,
because it is inside the ankle, and its height barely separates stance from
swing: thresholds loose enough to catch stance also catch half of swing. The
`ToeBase` joint sits at 0.4 to 2.4 cm when planted, which separates cleanly.
The reported sliding is still measured at whichever joint the caller asks
about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmm.data.skeleton import LEFT_FOOT, RIGHT_FOOT, Skeleton

__all__ = ["ContactThresholds", "detect_contacts", "contact_report", "CONTACT_JOINTS"]

# The joints contact is detected at, left then right. See the module docstring
# for why these are the toes rather than the ankles.
CONTACT_JOINTS = ("LeftToeBase", "RightToeBase")


@dataclass
class ContactThresholds:
    """Detection thresholds, in the data's own units.

    The defaults are for LaFAN1 resolved: centimetres, ground at y = 0, and
    detection at the toe joints, which sit at 0.4 to 2.4 cm when planted. Other
    datasets differ in scale and floor height, so these belong in the data
    config rather than hard coded.

    Attributes:
        height_on: a toe below this may enter contact.
        height_off: a toe must rise above this to leave contact. Higher than
            `height_on`, which is the hysteresis.
        speed_on: a toe slower than this may enter contact, in units/second.
        speed_off: a toe must exceed this to leave contact.
        min_duration: contacts and gaps shorter than this many frames are
            removed.
    """

    height_on: float = 4.0
    height_off: float = 7.0
    speed_on: float = 50.0
    speed_off: float = 100.0
    min_duration: int = 4


def _hysteresis(enter: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Walk a boolean state machine over time.

    `enter` is where contact may begin, `exit_` where it must end. Between the
    two the previous state is held, which is what stops the flicker.
    """
    state = np.zeros(len(enter), dtype=bool)
    on = False
    for i in range(len(enter)):
        if on:
            if exit_[i]:
                on = False
        else:
            if enter[i]:
                on = True
        state[i] = on
    return state


def _remove_short_runs(state: np.ndarray, min_duration: int) -> np.ndarray:
    """Drop runs shorter than `min_duration`, both of contact and of gap."""
    if min_duration <= 1 or len(state) == 0:
        return state

    state = state.copy()
    boundaries = np.flatnonzero(np.diff(state.astype(np.int8))) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(state)]])
    for start, stop in zip(starts, stops):
        if stop - start < min_duration:
            state[start:stop] = not state[start]
    return state


def detect_contacts(
    joint_positions: np.ndarray,
    skeleton: Skeleton,
    frame_time: float = 1.0 / 60.0,
    thresholds: ContactThresholds | None = None,
    joints=None,
) -> np.ndarray:
    """Label foot contacts.

    Args:
        joint_positions: `(n_frames, n_joints, 3)` in character space, with the
            ground at y = 0.
        skeleton: for the joint indices.
        frame_time: seconds per frame.
        thresholds: detection thresholds.
        joints: which joints to detect at, left then right. Defaults to the
            toes; see the module docstring.

    Returns:
        `(n_frames, 2)` booleans, left foot then right foot.
    """
    thresholds = thresholds or ContactThresholds()
    if joints is None:
        joints = CONTACT_JOINTS
    # Fall back to the ankles on a skeleton without toes, so this still works
    # on the SMPL 22 joint tree used by the AMASS route.
    feet = [
        skeleton.index(name) if skeleton.has(name) else skeleton.index(fallback)
        for name, fallback in zip(joints, (LEFT_FOOT, RIGHT_FOOT))
    ]

    contacts = np.zeros((len(joint_positions), len(feet)), dtype=bool)
    for f, joint in enumerate(feet):
        positions = joint_positions[:, joint]
        height = positions[:, 1]

        speed = np.zeros(len(positions), np.float32)
        if len(positions) > 1:
            speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / frame_time
            speed[0] = speed[1]

        enter = (height < thresholds.height_on) & (speed < thresholds.speed_on)
        exit_ = (height > thresholds.height_off) | (speed > thresholds.speed_off)
        state = _hysteresis(enter, exit_)
        contacts[:, f] = _remove_short_runs(state, thresholds.min_duration)

    return contacts


def contact_report(
    joint_positions: np.ndarray,
    contacts: np.ndarray,
    skeleton: Skeleton,
    frame_time: float = 1.0 / 60.0,
) -> dict:
    """Per clip diagnostics, used to drop broken clips during cleaning.

    Reports the fraction of frames in contact and the sliding distance while
    planted. A clip with almost no contact, or a lot of sliding while planted,
    is usually mis-scaled, floating, or captured on a treadmill.
    """
    feet = [skeleton.index(LEFT_FOOT), skeleton.index(RIGHT_FOOT)]
    report = {}
    for f, joint in enumerate(feet):
        planted = contacts[:, f]
        name = ("left", "right")[f]
        report[f"{name}_contact_fraction"] = float(planted.mean())

        if len(joint_positions) > 1:
            horizontal = joint_positions[:, joint][:, [0, 2]]
            step = np.linalg.norm(np.diff(horizontal, axis=0), axis=-1)
            while_planted = step[planted[1:]]
            report[f"{name}_slide_per_frame"] = (
                float(while_planted.mean()) if len(while_planted) else 0.0
            )
            report[f"{name}_slide_total"] = float(while_planted.sum()) if len(while_planted) else 0.0

    report["either_contact_fraction"] = float(contacts.any(axis=-1).mean())
    report["both_contact_fraction"] = float(contacts.all(axis=-1).mean())
    return report
