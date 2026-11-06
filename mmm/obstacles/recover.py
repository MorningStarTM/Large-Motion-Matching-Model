"""
Recovering obstacle geometry from motion alone.

LaFAN1's `obstacles` clips were captured with real boxes and platforms standing
in the studio. The BVH records only the skeleton, so the props are not in the
data -- but their *effect* is. A toe held still for a fifth of a second at a
height of 47 cm is a measurement of a 47 cm surface underneath it, and a hand
pressed still at 94 cm is a measurement of a vault box. Collect those
measurements and the props come back.

That this works is not an assumption. Pooling the elevated supports of the
eighteen obstacle clips recovers surfaces that nine clips across three separate
capture sessions agree on to within about a centimetre in height -- the studio
layout was persistent, so independent clips are independent measurements of the
same physical object. `scripts/recover_obstacles.py` prints that agreement.

Why not reuse `mmm.data.contacts`
---------------------------------
`detect_contacts` is tuned to answer a different question. It labels the foot
plants that the contact loss and the foot sliding metric need, on the ground,
at the toes, with thresholds chosen so that stance separates from swing on
level walking. Here the question is which plants are *load bearing surfaces*,
which needs hands as well as feet, no height ceiling, and a much longer minimum
duration -- a four frame plant is a footfall, and a fifteen frame one is someone
standing on something. Those thresholds pull in opposite directions, so this is
a separate detector rather than a reparameterisation of that one. `contacts.py`
is untouched.

The false positive that matters
-------------------------------
A foot at the apex of a jump is briefly slow, and naively that reads as a plant
two metres in the air. Three things rule it out: a minimum duration far longer
than an apex lasts, a requirement that the *vertical* speed be small
throughout rather than just the total speed being small on average, and a
ceiling on how high a supporting surface can plausibly be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mmm.data.bvh import Motion
from mmm.data.fk import forward_kinematics
from mmm.obstacles.geometry import Box, BoxField

__all__ = [
    "SupportThresholds",
    "Support",
    "detect_supports",
    "cluster_supports",
    "recover_field",
    "recover_studio_field",
    "SUPPORT_JOINTS",
]

# The joints a person bears weight through when negotiating an obstacle. Toes
# rather than ankles for the same reason `contacts.py` uses them: the Foot joint
# sits inside the ankle, 6.7 to 8.8 cm above a surface it is flat on, which
# biases every recovered height upward by that much.
SUPPORT_JOINTS = ("LeftToeBase", "RightToeBase")

# Hands are available but **off by default**, and the reason is a measurement.
# Including them was tried first and fails the negative control outright: over
# 34 non-obstacle clips it invents surfaces at 85 to 96 cm, because a hand
# swinging through the slow part of a walk cycle hangs at about 87 cm and sits
# still for long enough to pass every plant test. A height ceiling does not
# rescue it -- 87 cm is a perfectly plausible vault box. Feet-only over the
# same 34 clips yields 16 supports and 3 spurious surfaces in total, against 94
# to 178 supports per obstacle clip.
#
# Hands do carry real information about vault surfaces, and recovering them
# needs a test this detector does not implement: a hand bears weight only when
# it is below the hips with the torso loaded over it, which is a relation
# between joints rather than a property of one. Until that exists, passing
# hands in means accepting false surfaces.
HAND_JOINTS = ("LeftHand", "RightHand")


@dataclass
class SupportThresholds:
    """What counts as a load bearing contact, in the data's units.

    The defaults are for LaFAN1 resolved: centimetres, 60 Hz, ground near
    y = 0. Every one of them is a measurement rather than a guess; the reasons
    are given per field.

    Attributes:
        speed: a joint slower than this, in units/second, may be bearing
            weight. 20 cm/s is a quarter of `contacts.ContactThresholds.speed_on`,
            because this wants genuinely planted rather than merely slow.
        vertical_speed: the joint's vertical speed must also stay below this
            throughout. This is what rejects a jump apex, where the total speed
            dips but the trajectory is still arcing through.
        min_frames: how long the contact must hold. At 60 Hz, 10 frames is a
            sixth of a second -- longer than any apex, shorter than any real
            step onto a box.
        min_height: a support this far above the local floor is on something
            rather than on the ground. 15 cm rather than 12: at 12 the two walk
            clips `walk3_subject1` and `walk3_subject4` contribute surfaces at
            13.6, 14.4 and 19.4 cm, and 15 removes the first two while leaving
            every genuine step -- the lowest real one measured is 17.1 cm,
            backed by 11 clips.
        max_height: above this, a stationary foot is not standing on anything.
            190 cm clears the tallest recovered surface (181 cm, the top of a
            staircase) while excluding anything absurd.
        floor_percentile: the local floor is taken as this percentile of toe
            height over the clip, rather than assuming y = 0. Measured across
            LaFAN1 the per clip floor ranges from -1.86 to +0.77 cm, so
            assuming zero would shift every recovered height by up to 2 cm and
            destroy the cross clip agreement that makes this trustworthy.
    """

    speed: float = 20.0
    vertical_speed: float = 12.0
    min_frames: int = 10
    min_height: float = 15.0
    max_height: float = 190.0
    floor_percentile: float = 1.0


@dataclass
class Support:
    """One sustained load bearing contact.

    Attributes:
        x, z: where it happened, horizontally, in world space.
        height: how high above the clip's own floor.
        joint: which joint, for diagnostics.
        start, stop: the frame range, so a cluster can be traced back.
        clip: which clip it came from. Carried because the single most useful
            statistic about a recovered surface is how many *different* clips
            voted for it: one clip contributing forty contacts is one subject
            doing one thing, while eight clips contributing five each are eight
            independent measurements of a physical object. Without this the two
            are indistinguishable.
    """

    x: float
    z: float
    height: float
    joint: str
    start: int
    stop: int
    clip: str = ""

    @property
    def n_frames(self) -> int:
        return self.stop - self.start


def _sustained_runs(mask: np.ndarray, min_frames: int):
    """Frame ranges where `mask` holds for at least `min_frames`."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.diff(padded.astype(np.int8))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= min_frames:
            yield int(start), int(stop)


def _floor_height(positions: np.ndarray, skeleton, percentile: float) -> float:
    """The clip's own ground level, from the lowest the toes ever get.

    A percentile rather than the minimum, because a single frame of IK noise
    can put a toe a centimetre through the floor and that would shift every
    height in the clip.
    """
    toes = [
        skeleton.index(name)
        for name in ("LeftToeBase", "RightToeBase")
        if skeleton.has(name)
    ]
    if not toes:
        return 0.0
    heights = np.concatenate([positions[:, j, 1] for j in toes])
    return float(np.percentile(heights, percentile))


def detect_supports(
    motion: Motion,
    thresholds: SupportThresholds | None = None,
    joints=None,
) -> tuple[list[Support], float]:
    """Find every sustained elevated contact in one clip.

    Args:
        motion: a clip. Joint positions are computed here by forward
            kinematics, so this takes the motion rather than positions, and the
            caller need not know which joints matter.
        thresholds: detection thresholds.
        joints: which joints to look at. Defaults to `SUPPORT_JOINTS`; joints
            the skeleton lacks are skipped rather than raising, so this also
            runs on the SMPL joint tree from the AMASS route.

    Returns:
        `(supports, floor)` -- the contacts found, and the clip's floor height,
        which the caller needs to put the recovered boxes back into world space.
    """
    thresholds = thresholds or SupportThresholds()
    joints = joints if joints is not None else SUPPORT_JOINTS

    positions, _ = forward_kinematics(
        motion.skeleton, motion.rotations, motion.root_positions
    )
    floor = _floor_height(positions, motion.skeleton, thresholds.floor_percentile)

    supports: list[Support] = []
    for name in joints:
        if not motion.skeleton.has(name):
            continue
        track = positions[:, motion.skeleton.index(name)]
        if len(track) < 2:
            continue

        step = np.diff(track, axis=0) / motion.frame_time
        speed = np.zeros(len(track), np.float32)
        speed[1:] = np.linalg.norm(step, axis=-1)
        speed[0] = speed[1]

        vertical = np.zeros(len(track), np.float32)
        vertical[1:] = np.abs(step[:, 1])
        vertical[0] = vertical[1]

        planted = (speed < thresholds.speed) & (vertical < thresholds.vertical_speed)

        for start, stop in _sustained_runs(planted, thresholds.min_frames):
            segment = track[start:stop]
            # The median over the run, not the mean: a run that begins or ends
            # a frame early picks up one transitional frame, and the median
            # ignores it.
            height = float(np.median(segment[:, 1])) - floor
            if not (thresholds.min_height <= height <= thresholds.max_height):
                continue
            supports.append(
                Support(
                    x=float(np.median(segment[:, 0])),
                    z=float(np.median(segment[:, 2])),
                    height=height,
                    joint=name,
                    start=start,
                    stop=stop,
                    clip=motion.name,
                )
            )

    return supports, floor


@dataclass
class SupportCluster:
    """A group of supports believed to be the same physical surface."""

    supports: list[Support] = field(default_factory=list)

    @property
    def height(self) -> float:
        return float(np.median([s.height for s in self.supports]))

    @property
    def joints(self) -> set:
        return {s.joint for s in self.supports}

    @property
    def clips(self) -> set:
        """Which clips contributed. See `Support.clip` for why this matters."""
        return {s.clip for s in self.supports if s.clip}

    @property
    def height_spread(self) -> float:
        """Standard deviation of the contributing heights.

        The honest confidence measure for a recovered surface. A real rigid
        prop measured many times gives well under 2 cm; a cluster that has
        chained two different objects together gives much more.
        """
        return float(np.std([s.height for s in self.supports]))

    def bounds(self) -> tuple[float, float, float, float]:
        xs = np.array([s.x for s in self.supports])
        zs = np.array([s.z for s in self.supports])
        return float(xs.min()), float(xs.max()), float(zs.min()), float(zs.max())


def cluster_supports(
    supports,
    radius: float = 45.0,
    height_tolerance: float = 12.0,
    min_supports: int = 3,
) -> list[SupportCluster]:
    """Group supports into surfaces.

    Single link agglomeration on the horizontal plane, gated by height: two
    supports join the same cluster when they are within `radius` of *any*
    member and their heights agree to within `height_tolerance`. Single link
    rather than a centroid rule because a platform is long and a centroid rule
    would split it in half; the height gate is what stops single link chaining
    a 20 cm step into the 47 cm platform beside it, which is the failure a
    purely spatial clustering has here.

    Args:
        supports: as returned by `detect_supports`, possibly pooled over clips.
        radius: horizontal linking distance. 45 cm is a little over one stride,
            chosen so that consecutive footfalls along one platform link while
            two separate steps do not.
        height_tolerance: heights within this are the same surface. 12 cm is
            below the smallest gap between the distinct step heights measured in
            LaFAN1 (17, 21, 33, 39, 47 cm), so no two of them merge.
        min_supports: a cluster with fewer members than this is dropped. One or
            two stationary frames at an odd height is noise; three independent
            plants at the same place and height is a surface.

    Returns:
        Clusters, most heavily evidenced first.
    """
    clusters: list[SupportCluster] = []
    for support in supports:
        for cluster in clusters:
            if abs(support.height - cluster.height) >= height_tolerance:
                continue
            near = min(
                (support.x - other.x) ** 2 + (support.z - other.z) ** 2
                for other in cluster.supports
            )
            if near < radius * radius:
                cluster.supports.append(support)
                break
        else:
            clusters.append(SupportCluster([support]))

    clusters = [c for c in clusters if len(c.supports) >= min_supports]
    return sorted(clusters, key=lambda c: -len(c.supports))


def _cluster_to_box(cluster: SupportCluster, margin: float, floor: float) -> Box:
    """Fit a box to a cluster.

    The footprint is the bounding box of the contact points grown by `margin`.
    Growing it is not fudging: a foot samples the surface only where it was
    placed, which is strictly inside the real prop, so the bounding box of the
    samples is a *lower* bound on the footprint. The margin is half a foot
    length, the smallest correction that is certainly still an underestimate.

    The top is the cluster's median height, put back into world space by adding
    the clip's floor so the box lands where the character actually is.
    """
    x0, x1, z0, z1 = cluster.bounds()
    x0, x1 = x0 - margin, x1 + margin
    z0, z1 = z0 - margin, z1 + margin

    joints = cluster.joints
    if joints <= {"LeftHand", "RightHand"}:
        kind = "hand"
    elif joints <= {"LeftToeBase", "RightToeBase"}:
        kind = "foot"
    else:
        kind = "mixed"

    return Box(
        x0=x0,
        x1=x1,
        z0=z0,
        z1=z1,
        top=cluster.height + floor,
        base=0.0,
        label=f"{kind}:{len(cluster.supports)}",
    )


def recover_field(
    motion: Motion,
    thresholds: SupportThresholds | None = None,
    radius: float = 45.0,
    height_tolerance: float = 12.0,
    min_supports: int = 3,
    margin: float = 12.0,
) -> BoxField:
    """Recover the obstacles present during one clip.

    Convenience over `detect_supports` plus `cluster_supports`. Per clip
    recovery is noisier than pooling -- one clip only samples the props that
    subject happened to use -- so prefer `recover_studio_field` when several
    clips from the same capture volume are available.
    """
    supports, floor = detect_supports(motion, thresholds)
    clusters = cluster_supports(supports, radius, height_tolerance, min_supports)
    return BoxField([_cluster_to_box(c, margin, floor) for c in clusters])


def recover_studio_field(
    motions,
    thresholds: SupportThresholds | None = None,
    radius: float = 45.0,
    height_tolerance: float = 12.0,
    min_supports: int = 6,
    min_clips: int = 2,
    margin: float = 12.0,
) -> tuple[BoxField, list[SupportCluster]]:
    """Recover one obstacle layout from several clips of the same studio.

    This is the accurate route, and the reason it works is that the props did
    not move between captures. Each clip is then an independent measurement of
    the same geometry, so pooling both sharpens every height and fills in the
    parts of a platform that any one subject never stepped on.

    Heights are referenced to each clip's own floor before pooling, which is
    what makes clips with floors 2 cm apart agree rather than smear.

    `min_supports` defaults higher than for a single clip because there is far
    more evidence to demand: with eighteen clips pooled, a surface backed by
    three contacts is probably one subject doing something odd once.

    `min_clips` is the stronger of the two filters and the one to reach for.
    Forty contacts from a single clip is one subject standing somewhere
    repeatedly, which could as easily be a pose as a prop; two clips agreeing
    is already corroboration. Measured on the eighteen LaFAN1 obstacle clips,
    the surfaces that survive are backed by 2 to 13 clips each with height
    spreads of 0.5 to 3.7 cm, while every single-clip cluster has either a
    large spread or a tiny footprint.

    Args:
        motions: clips from one capture volume. Each needs `name` set for the
            per-surface clip counts to mean anything.

    Returns:
        The field, and the clusters behind it, so a caller can report how many
        clips voted for each box and how tightly they agreed.
    """
    pooled: list[Support] = []
    for motion in motions:
        supports, floor = detect_supports(motion, thresholds)
        # Heights are already floor relative, which is exactly why they pool.
        pooled.extend(supports)

    clusters = cluster_supports(pooled, radius, height_tolerance, min_supports)
    if min_clips > 1:
        clusters = [c for c in clusters if len(c.clips) >= min_clips]
    # Pooled heights are floor relative and the per clip floors differ by only
    # a couple of centimetres, so the pooled field is expressed against a
    # nominal floor of zero rather than any one clip's.
    return BoxField([_cluster_to_box(c, margin, 0.0) for c in clusters]), clusters
