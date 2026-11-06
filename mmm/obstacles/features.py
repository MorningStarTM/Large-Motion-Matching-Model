"""
The obstacle feature block, and the matching query to go with it.

The base 27 feature vector in `mmm/data/features.py` already describes where
the character is standing: foot positions are in character space with height
kept, so a foot planted on a 47 cm box reads as y = 47. What it cannot describe
is where the character is *about to be*. Nothing in those 27 numbers
distinguishes walking on flat ground from walking at a step two metres ahead,
and so the search cannot retrieve a step-up before the step arrives.

These five numbers supply exactly that missing anticipation, and nothing else:

    index  size  content
    0:3    3     support height at the future root positions (20, 40, 60
                 frames), relative to the support height under the character
                 now
    3:4    1     distance along the facing direction to the next obstacle edge,
                 clamped
    4:5    1     that obstacle's top, relative to the support height now

Why relative heights
--------------------
Everything is measured against the surface the character is standing on rather
than against the world floor. A character on flat ground facing a 47 cm box and
a character already on a 47 cm platform facing a 94 cm box are the same problem
-- step up 47 cm -- and should retrieve the same frames. Absolute heights would
make those two situations look completely different and halve the useful data.

Why an edge distance as well as the sampled profile
---------------------------------------------------
The three sampled heights say what the ground does at three instants, which is
a coarse picture: a 30 cm thick step between two samples is invisible. The edge
distance is continuous in the quantity that actually governs the motion -- how
many strides are left before the foot has to go somewhere else -- so it varies
smoothly as the character approaches and gives the search something to track
between samples.

Layout discipline
-----------------
Appended after the base 27, never interleaved, so a row built by
`mmm.data.features.compute_features` is a prefix of a row built here. That is
what lets the same Animation Database serve both an obstacle-aware and an
obstacle-blind search, and it mirrors the prefix rule the model's input layer
already follows in `mmm/models/io_layers.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np

from mmm.data import spaces
from mmm.data.bvh import Motion
from mmm.data.features import TRAJECTORY_HORIZONS
from mmm.data.fk import forward_kinematics
from mmm.obstacles.geometry import BoxField

__all__ = [
    "OBSTACLE_BLOCKS",
    "OBSTACLE_FEATURE_DIM",
    "MAX_EDGE_DISTANCE",
    "obstacle_block_slice",
    "ObstacleNormalization",
    "compute_obstacle_features",
    "build_obstacle_query",
]

OBSTACLE_BLOCKS = (
    ("future_support", len(TRAJECTORY_HORIZONS)),
    ("edge_distance", 1),
    ("edge_height", 1),
)

OBSTACLE_FEATURE_DIM = sum(size for _, size in OBSTACLE_BLOCKS)

# How far ahead the edge search looks, in centimetres. Four metres is about
# four strides at walking speed and two at a run, which is as far ahead as the
# anticipation in the data extends: the subjects did not begin adjusting for a
# box any earlier than that, so a longer horizon would add a feature that
# varies while the motion does not.
MAX_EDGE_DISTANCE = 400.0


def obstacle_block_slice(name: str) -> slice:
    """The slice of the obstacle vector a named block occupies.

    Indices are into the obstacle block alone, not into a concatenated row. The
    offset of the block within a full row belongs to `database.py`, which is
    the only place that knows how the two halves are joined.
    """
    start = 0
    for block_name, size in OBSTACLE_BLOCKS:
        if block_name == name:
            return slice(start, start + size)
        start += size
    raise KeyError(
        f"unknown obstacle block {name!r}. Known: {[b for b, _ in OBSTACLE_BLOCKS]}"
    )


@dataclass
class ObstacleNormalization:
    """Per dimension statistics and per block weights for the obstacle block.

    A separate class from `mmm.data.features.FeatureNormalization` rather than a
    generalisation of it, because that one validates its shape against
    `FEATURE_DIM` and is stored inside every existing database. Widening it
    would change the meaning of files already on disk; this leaves them alone.

    The weights deserve attention. Normalisation makes every dimension unit
    variance, which means the five obstacle numbers would carry the same total
    weight in the KNN distance as five of the twenty-seven base numbers. That is
    almost certainly too little: on flat ground these five are constant and
    contribute nothing, so their variance is concentrated in the minority of
    frames near an obstacle, and it is precisely in those frames that they
    should dominate. `default_weights` therefore starts them above 1, and
    `scripts/recover_obstacles.py` reports what the choice does to retrieval.
    """

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray = dataclass_field(default=None)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(self.std, dtype=np.float32).reshape(-1)
        if self.mean.shape != (OBSTACLE_FEATURE_DIM,):
            raise ValueError(
                f"mean must have shape ({OBSTACLE_FEATURE_DIM},); "
                f"got {self.mean.shape}"
            )
        if self.std.shape != (OBSTACLE_FEATURE_DIM,):
            raise ValueError(
                f"std must have shape ({OBSTACLE_FEATURE_DIM},); got {self.std.shape}"
            )
        # A database built entirely from flat clips has zero variance in every
        # obstacle dimension. Clamping leaves those dimensions at zero after
        # normalisation, which is the right answer: a constant feature cannot
        # discriminate, and the search degrades to the obstacle-blind one
        # rather than dividing by zero.
        self.std = np.maximum(self.std, 1e-6)
        if self.weights is None:
            self.weights = self.default_weights()
        else:
            self.weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)

    @staticmethod
    def default_weights() -> np.ndarray:
        """Starting weights, higher than 1 for the reason in the class docstring."""
        weights = np.ones(OBSTACLE_FEATURE_DIM, dtype=np.float32)
        weights[obstacle_block_slice("future_support")] = 3.0
        weights[obstacle_block_slice("edge_distance")] = 2.0
        weights[obstacle_block_slice("edge_height")] = 3.0
        return weights

    @classmethod
    def fit(cls, features: np.ndarray, block_weights: dict | None = None):
        """Measure statistics on a set of obstacle rows.

        Args:
            features: `(n_rows, OBSTACLE_FEATURE_DIM)`, unnormalized, training
                set only.
            block_weights: optional `{block_name: weight}` overriding
                `default_weights`, broadcast across the block.
        """
        weights = cls.default_weights()
        for name, weight in (block_weights or {}).items():
            weights[obstacle_block_slice(name)] = weight
        return cls(
            mean=np.asarray(features).mean(axis=0),
            std=np.asarray(features).std(axis=0),
            weights=weights,
        )

    def apply(self, features: np.ndarray) -> np.ndarray:
        return (((features - self.mean) / self.std) * self.weights).astype(np.float32)

    def invert(self, normalized: np.ndarray) -> np.ndarray:
        return (
            (normalized / np.maximum(self.weights, 1e-12)) * self.std + self.mean
        ).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, record: dict) -> "ObstacleNormalization":
        return cls(
            mean=np.array(record["mean"], np.float32),
            std=np.array(record["std"], np.float32),
            weights=np.array(record["weights"], np.float32),
        )


# ---------------------------------------------------------------------------
# computing rows
# ---------------------------------------------------------------------------

def _world_future_positions(frame: spaces.RootFrame, horizons) -> np.ndarray:
    """Future root positions in **world** XZ, `(n_frames, n_horizons, 2)`.

    The base features store the same path in character space, because that is
    what the search compares. Here world space is needed instead: the box field
    lives in the world, and rotating the field per frame would be both wrong
    and far slower than rotating three points.

    The lookahead is clamped at the last frame, matching
    `mmm.data.features._future_trajectory`, so the arrays stay in bounds. Those
    tail rows are marked invalid by the base features and never retrieved, so
    their value is unused.
    """
    n = frame.n_frames
    out = np.empty((n, len(horizons), 2), np.float32)
    index = np.arange(n)
    for h, horizon in enumerate(horizons):
        future = np.minimum(index + horizon, n - 1)
        out[:, h] = frame.origin[future][:, [0, 2]]
    return out


def compute_obstacle_features(
    motion: Motion,
    field: BoxField,
    horizons=TRAJECTORY_HORIZONS,
    max_distance: float = MAX_EDGE_DISTANCE,
) -> np.ndarray:
    """The obstacle block for every frame of one clip.

    Args:
        motion: a clip, on the same joint subset as the base features.
        field: the obstacles present during this clip, in the clip's own world
            coordinates. For LaFAN1 this comes from
            `mmm.obstacles.recover`; an empty field is legal and yields all
            zeros, which is the correct description of flat ground.
        horizons: lookahead in frames. Must match the base features, or the two
            halves of a row describe different instants.
        max_distance: how far ahead to look for an edge.

    Returns:
        `(n_frames, OBSTACLE_FEATURE_DIM)`, unnormalized.
    """
    _, global_rotations = forward_kinematics(
        motion.skeleton, motion.rotations, motion.root_positions
    )
    frame = spaces.extract_root_frames(global_rotations, motion.root_positions)
    n = motion.n_frames

    out = np.zeros((n, OBSTACLE_FEATURE_DIM), np.float32)
    if len(field) == 0:
        # Flat ground: every support height is 0, so every relative height is
        # 0, and there is no change within range. Returning early also skips a
        # couple of million pointless support queries, which is what a walk
        # clip in a mixed database would otherwise cost.
        out[:, obstacle_block_slice("edge_distance")] = max_distance
        return out

    here = frame.origin[:, [0, 2]]
    support_now = field.support_height(here[:, 0], here[:, 1])

    future = _world_future_positions(frame, horizons)
    support_future = np.stack(
        [
            field.support_height(future[:, h, 0], future[:, h, 1])
            for h in range(len(horizons))
        ],
        axis=-1,
    )
    out[:, obstacle_block_slice("future_support")] = support_future - support_now[:, None]

    # The edge search is done for every frame at once rather than by calling
    # `edge_ahead` in a loop. Same answer, but a clip of 15,000 frames at 80
    # samples each is 1.2 million support queries, which is fifteen vectorised
    # calls this way round and 15,000 Python iterations the other.
    sample_distances, profile = field.support_profile(
        here, frame.forward, max_distance=max_distance
    )
    changed = np.abs(profile - support_now[:, None]) > field.HEIGHT_TOLERANCE

    # `argmax` on a boolean row gives the first True, or 0 when there is none,
    # so the no-change rows are repaired explicitly afterwards rather than
    # trusting an index that means two different things.
    first = np.argmax(changed, axis=1)
    any_change = changed.any(axis=1)

    distances = np.where(any_change, sample_distances[first], max_distance)
    heights = np.where(any_change, profile[np.arange(n), first], support_now)

    out[:, obstacle_block_slice("edge_distance")] = distances[:, None]
    out[:, obstacle_block_slice("edge_height")] = (heights - support_now)[:, None]

    if not np.isfinite(out).all():
        raise ValueError(f"{motion.name}: obstacle features contain non-finite values")
    return out


# ---------------------------------------------------------------------------
# building a runtime query
# ---------------------------------------------------------------------------

def build_obstacle_query(
    field: BoxField,
    origin_xz,
    forward_xz,
    desired_positions,
    max_distance: float = MAX_EDGE_DISTANCE,
) -> np.ndarray:
    """Assemble one obstacle query, in the identical layout to
    `compute_obstacle_features`.

    The asymmetry with the database side is deliberate and is the whole trick:
    a database row describes the obstacles the actor *actually* negotiated,
    while a query describes the obstacles the user has *just placed* in front of
    the character. Because both are reduced to the same five numbers against
    whatever surface is underfoot, a cube dropped in the viewer retrieves the
    frames where the actor faced a comparable cube.

    Args:
        field: the obstacles in the live scene.
        origin_xz: the character's `(x, z)` on the floor, world space.
        forward_xz: the character's facing `(x, z)`, world space.
        desired_positions: `(n_horizons, 2)` the spring damper's predicted
            future root positions, in **character** space -- the same array
            that goes into `mmm.data.features.build_query`. Rotated into the
            world here, so the caller does not have to hold both forms.
        max_distance: must match the value used to build the rows.

    Returns:
        `(OBSTACLE_FEATURE_DIM,)`, unnormalized. Pass it through
        `ObstacleNormalization.apply` before searching.
    """
    origin_xz = np.asarray(origin_xz, np.float32).reshape(2)
    forward_xz = np.asarray(forward_xz, np.float32).reshape(2)
    desired = np.asarray(desired_positions, np.float32).reshape(-1, 2)

    query = np.zeros(OBSTACLE_FEATURE_DIM, np.float32)
    if len(field) == 0:
        query[obstacle_block_slice("edge_distance")] = max_distance
        return query

    support_now = float(field.support_height(origin_xz[0], origin_xz[1]))

    # Character space to world: the same rotation the base features invert, so
    # the query path and the database path are the identical curve expressed in
    # two frames rather than two different curves.
    world = spaces._rotate_out_of_frame(desired, forward_xz) + origin_xz
    support_future = field.support_height(world[:, 0], world[:, 1])
    query[obstacle_block_slice("future_support")] = support_future - support_now

    distance, top = field.edge_ahead(origin_xz, forward_xz, max_distance)
    query[obstacle_block_slice("edge_distance")] = distance
    query[obstacle_block_slice("edge_height")] = top - support_now
    return query
