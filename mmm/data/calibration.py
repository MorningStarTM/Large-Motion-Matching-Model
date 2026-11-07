"""
Finding and removing the calibration T-pose at the start of a clip.

Every LaFAN1 take opens with the actor standing still, arms straight out, while
the capture system calibrates. It is recorded along with everything else, so
each clip begins with a few seconds of a pose no player will ever request.

Measured across the dataset it runs from 104 to 475 frames -- 1.7 to 7.9
seconds at 60 Hz -- and the length varies by more than 3x between clips, so a
fixed cut is wrong for almost all of them. It has to be detected.

Why it is worth removing
------------------------
Two separate costs:

- **Training.** Phase 1 learns to predict the next pose from the previous one.
  These frames teach "if you are standing still with your arms out, keep
  standing still with your arms out", which is about 2 to 4 per cent of the
  data spent on something useless, and it biases the model toward stillness.
- **The Matching Database, which is worse.** Every frame is a searchable row.
  If a query ever lands near one -- the player stops moving, say -- the search
  can legitimately return it and the character snaps into a T-pose mid-game.
  That is a visible, embarrassing failure.

How it is detected
------------------
Arm span: the distance between the two hands. It separates cleanly and needs
no thresholds tuned per clip.

    walking / running     45 to 55 cm
    calibration T-pose   125 to 130 cm

Measured on `walk1_subject1`, the span is 126.9 cm at frame 0 and 45.8 cm by
frame 200 -- the actor drops their arms over about a dozen frames. Thresholds
from 70 to 110 cm therefore give cuts of 156 to 168 frames: not identical, but
a spread of 0.2 seconds on a 2.7 second prefix, with no threshold in that
range landing anywhere else. That insensitivity is what makes this safe.

Only the **leading** run is removed. A T-pose in the middle of a clip is the
actor doing something, not the system calibrating, and cutting there would
split a clip in two.
"""

from __future__ import annotations

import numpy as np

from mmm.data.bvh import Motion
from mmm.data.fk import forward_kinematics

__all__ = [
    "ARM_SPAN_THRESHOLD",
    "arm_span",
    "calibration_length",
    "trim_calibration",
]

# Halfway between walking (about 50 cm) and the T-pose (about 127 cm). The gap
# is so wide that the exact value does not matter; 110 or 70 pick the same
# frame. 90 sits in the middle of the safe range.
ARM_SPAN_THRESHOLD = 90.0

# A clip whose opening pose is not a T-pose should be left alone entirely, and
# a clip that is *entirely* T-pose is broken rather than trimmable. Refusing to
# cut more than this fraction stops a detection bug from deleting a clip.
MAX_TRIM_FRACTION = 0.5

_HANDS = ("LeftHand", "RightHand")


def arm_span(motion: Motion) -> np.ndarray:
    """Distance between the hands, per frame, in the clip's units.

    Returns zeros for a skeleton without hands -- the SMPL tree from the AMASS
    route has them, but a reduced one might not, and a missing joint should
    mean "cannot detect" rather than a crash.
    """
    skeleton = motion.skeleton
    if not all(skeleton.has(name) for name in _HANDS):
        return np.zeros(motion.n_frames, np.float32)

    positions, _ = forward_kinematics(
        skeleton, motion.rotations, motion.root_positions
    )
    left = positions[:, skeleton.index("LeftHand")]
    right = positions[:, skeleton.index("RightHand")]
    return np.linalg.norm(left - right, axis=-1).astype(np.float32)


def calibration_length(
    motion: Motion,
    threshold: float = ARM_SPAN_THRESHOLD,
    max_fraction: float = MAX_TRIM_FRACTION,
) -> int:
    """How many leading frames are the calibration T-pose.

    Args:
        motion: the clip.
        threshold: arm span below which the actor is considered to have put
            their arms down.
        max_fraction: refuse to report more than this fraction of the clip.

    Returns:
        The number of frames to drop. 0 when the clip does not start in a
        T-pose, which is the right answer for a clip that was already trimmed.
    """
    span = arm_span(motion)
    if len(span) == 0 or span[0] <= threshold:
        return 0

    below = np.flatnonzero(span <= threshold)
    if len(below) == 0:
        # The arms never come down. That is not a calibration prefix, it is a
        # clip of someone standing in a T-pose, and silently deleting it would
        # be worse than keeping it.
        return 0

    length = int(below[0])
    limit = int(len(span) * max_fraction)
    return min(length, limit)


def trim_calibration(
    motion: Motion,
    threshold: float = ARM_SPAN_THRESHOLD,
    max_fraction: float = MAX_TRIM_FRACTION,
    verbose: bool = False,
) -> tuple[Motion, int]:
    """Drop the leading calibration T-pose.

    Returns:
        `(motion, dropped)`. The motion is returned unchanged, not copied,
        when there is nothing to drop.
    """
    dropped = calibration_length(motion, threshold, max_fraction)
    if dropped <= 0:
        return motion, 0

    name = motion.name
    trimmed = motion.slice(dropped, motion.n_frames)
    # `slice` renames to "clip[a:b]", which would then flow into the database's
    # clip table and every log line. The clip is the same clip.
    trimmed.name = name

    if verbose:
        print(
            f"  {name:24s} dropped {dropped:4d} calibration frames "
            f"({dropped / motion.fps:4.1f}s)"
        )
    return trimmed, dropped
