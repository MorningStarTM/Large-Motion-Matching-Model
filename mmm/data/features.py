"""
Matching Database features: the short vector per frame used for searching.

Twenty seven numbers per frame for locomotion (plan section 3.1), all in
character space. The layout is declared once here, as `FEATURE_BLOCKS`, and
imported everywhere else:

    index   size  content
    0:6     6     future root positions at 20, 40, 60 frames (x, z each)
    6:12    6     future root facing directions at 20, 40, 60 frames (x, z each)
    12:18   6     left and right foot positions (x, y, z each)
    18:24   6     left and right foot velocities (x, y, z each)
    24:27   3     hip velocity (x, y, z)

A mismatch between the row layout and the query layout is the single most
likely cause of a teacher that retrieves nonsense while still appearing to run,
which is why `build_query` lives next to `compute_features` and why
`tests/test_features.py` checks that a frame's own state retrieves that frame.

Normalization is per dimension: subtract the mean, divide by the standard
deviation, both measured on the training set and stored with the database.
Optional per block weights then set how much each term counts in the KNN
distance; they default to 1 and are applied after normalization, so changing a
weight does not invalidate the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mmm.data import spaces
from mmm.data.bvh import Motion
from mmm.data.fk import forward_kinematics, joint_velocities
from mmm.data.skeleton import HIPS, LEFT_FOOT, RIGHT_FOOT

__all__ = [
    "FEATURE_DIM",
    "FEATURE_BLOCKS",
    "TRAJECTORY_HORIZONS",
    "JOYSTICK_DIM",
    "FeatureNormalization",
    "compute_features",
    "compute_joystick",
    "build_query",
    "feature_block_slice",
]

# How many frames ahead the trajectory is sampled. At 60Hz these are 1/3, 2/3
# and 1 second.
TRAJECTORY_HORIZONS = (20, 40, 60)

FEATURE_BLOCKS = (
    ("future_positions", 2 * len(TRAJECTORY_HORIZONS)),
    ("future_directions", 2 * len(TRAJECTORY_HORIZONS)),
    ("foot_positions", 6),
    ("foot_velocities", 6),
    ("hip_velocity", 3),
)

FEATURE_DIM = sum(size for _, size in FEATURE_BLOCKS)

# The joystick input the model reads is the first two blocks of the same
# vector: future root positions and facing. 12 numbers, plan section 4.3.
JOYSTICK_DIM = FEATURE_BLOCKS[0][1] + FEATURE_BLOCKS[1][1]


def feature_block_slice(name: str) -> slice:
    """The slice of the feature vector a named block occupies."""
    start = 0
    for block_name, size in FEATURE_BLOCKS:
        if block_name == name:
            return slice(start, start + size)
        start += size
    raise KeyError(f"unknown feature block {name!r}. Known: {[b for b, _ in FEATURE_BLOCKS]}")


@dataclass
class FeatureNormalization:
    """Per dimension statistics plus per block weights.

    Stored alongside the database so that search and query cannot disagree.
    """

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(self.std, dtype=np.float32).reshape(-1)
        if self.mean.shape != (FEATURE_DIM,) or self.std.shape != (FEATURE_DIM,):
            raise ValueError(
                f"mean and std must both have shape ({FEATURE_DIM},); "
                f"got {self.mean.shape} and {self.std.shape}"
            )
        # A dimension that never varies would divide by zero. Clamping to a
        # small floor leaves it at zero after normalization, which is correct:
        # a constant feature carries no information for the search.
        self.std = np.maximum(self.std, 1e-6)
        if self.weights is None:
            self.weights = np.ones(FEATURE_DIM, dtype=np.float32)
        else:
            self.weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)

    @classmethod
    def fit(cls, features: np.ndarray, block_weights: dict[str, float] | None = None):
        """Measure statistics on a set of feature rows.

        Args:
            features: `(n_rows, FEATURE_DIM)`, unnormalized, training set only.
            block_weights: optional `{block_name: weight}`, broadcast across
                every dimension of that block.
        """
        weights = np.ones(FEATURE_DIM, dtype=np.float32)
        for name, weight in (block_weights or {}).items():
            weights[feature_block_slice(name)] = weight
        return cls(mean=features.mean(axis=0), std=features.std(axis=0), weights=weights)

    def apply(self, features: np.ndarray) -> np.ndarray:
        """Normalize and weight, the form the KNN search actually compares."""
        return ((features - self.mean) / self.std) * self.weights

    def invert(self, normalized: np.ndarray) -> np.ndarray:
        return (normalized / np.maximum(self.weights, 1e-12)) * self.std + self.mean


# ---------------------------------------------------------------------------
# computing rows
# ---------------------------------------------------------------------------

def _future_trajectory(frame: spaces.RootFrame, horizons=TRAJECTORY_HORIZONS):
    """Future root positions and facing directions, in each frame's own
    character space.

    Returns `(positions, directions, valid)` where positions and directions are
    `(n_frames, n_horizons, 2)` and `valid` is `(n_frames,)`: False for the
    last `max(horizons)` frames, whose future does not exist.
    """
    n = frame.n_frames
    longest = max(horizons)

    positions = np.zeros((n, len(horizons), 2), dtype=np.float32)
    directions = np.zeros((n, len(horizons), 2), dtype=np.float32)

    for h, horizon in enumerate(horizons):
        # Clamp the lookahead so the array stays in bounds; those rows are
        # marked invalid below and never retrieved, so their value is unused.
        future = np.minimum(np.arange(n) + horizon, n - 1)

        offset_world = frame.origin[future] - frame.origin
        local = spaces._rotate_into_frame(offset_world[:, [0, 2]], frame.forward)
        positions[:, h] = local

        directions[:, h] = spaces._rotate_into_frame(frame.forward[future], frame.forward)

    valid = np.ones(n, dtype=bool)
    valid[max(0, n - longest):] = False
    return positions, directions, valid


def compute_features(motion: Motion, horizons=TRAJECTORY_HORIZONS):
    """The Matching Database rows for one clip.

    Args:
        motion: a clip, already reduced to the locomotion joint subset.
        horizons: lookahead in frames.

    Returns:
        `features` `(n_frames, FEATURE_DIM)` unnormalized, and `valid`
        `(n_frames,)` -- False where the future trajectory does not exist, so
        those rows must never be returned by a search.
    """
    skeleton = motion.skeleton
    positions, global_rotations = forward_kinematics(
        skeleton, motion.rotations, motion.root_positions
    )
    frame = spaces.extract_root_frames(global_rotations, motion.root_positions)

    local_positions = spaces.to_character_space(positions, frame)
    world_velocities = joint_velocities(positions, motion.frame_time)
    local_velocities = spaces.direction_to_character_space(world_velocities, frame)

    left = skeleton.index(LEFT_FOOT)
    right = skeleton.index(RIGHT_FOOT)
    hips = skeleton.index(HIPS)

    future_positions, future_directions, valid = _future_trajectory(frame, horizons)

    n = motion.n_frames
    features = np.empty((n, FEATURE_DIM), dtype=np.float32)
    features[:, feature_block_slice("future_positions")] = future_positions.reshape(n, -1)
    features[:, feature_block_slice("future_directions")] = future_directions.reshape(n, -1)
    features[:, feature_block_slice("foot_positions")] = np.concatenate(
        [local_positions[:, left], local_positions[:, right]], axis=-1
    )
    features[:, feature_block_slice("foot_velocities")] = np.concatenate(
        [local_velocities[:, left], local_velocities[:, right]], axis=-1
    )
    features[:, feature_block_slice("hip_velocity")] = local_velocities[:, hips]

    if not np.isfinite(features).all():
        raise ValueError(f"{motion.name}: features contain non-finite values")

    return features, valid


def compute_joystick(motion: Motion, horizons=TRAJECTORY_HORIZONS) -> np.ndarray:
    """The 12 number joystick signal for every frame of a clip.

    During Phase 1 pretraining the joystick is taken from the data's own real
    future root path, so this is simply the first two blocks of the feature
    vector. Sharing the code guarantees the model's joystick input and the
    teacher's query describe the same thing.
    """
    _, global_rotations = forward_kinematics(motion.skeleton, motion.rotations, motion.root_positions)
    frame = spaces.extract_root_frames(global_rotations, motion.root_positions)
    future_positions, future_directions, _ = _future_trajectory(frame, horizons)
    n = motion.n_frames
    return np.concatenate(
        [future_positions.reshape(n, -1), future_directions.reshape(n, -1)], axis=-1
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# building a runtime query
# ---------------------------------------------------------------------------

def build_query(
    foot_positions: np.ndarray,
    foot_velocities: np.ndarray,
    hip_velocity: np.ndarray,
    desired_positions: np.ndarray,
    desired_directions: np.ndarray,
) -> np.ndarray:
    """Assemble one query vector, in the identical layout to `compute_features`.

    The feet and hips come from the character's current state; the future path
    comes from the joystick via the spring damper. All arguments are already in
    character space.

    Args:
        foot_positions: `(2, 3)` left then right.
        foot_velocities: `(2, 3)` left then right.
        hip_velocity: `(3,)`.
        desired_positions: `(n_horizons, 2)` future root positions.
        desired_directions: `(n_horizons, 2)` future facing directions.

    Returns:
        `(FEATURE_DIM,)` unnormalized. Pass it through
        `FeatureNormalization.apply` before searching.
    """
    query = np.empty(FEATURE_DIM, dtype=np.float32)
    query[feature_block_slice("future_positions")] = np.asarray(desired_positions, np.float32).reshape(-1)
    query[feature_block_slice("future_directions")] = np.asarray(desired_directions, np.float32).reshape(-1)
    query[feature_block_slice("foot_positions")] = np.asarray(foot_positions, np.float32).reshape(-1)
    query[feature_block_slice("foot_velocities")] = np.asarray(foot_velocities, np.float32).reshape(-1)
    query[feature_block_slice("hip_velocity")] = np.asarray(hip_velocity, np.float32).reshape(-1)
    return query
