"""
Joining the obstacle block to an existing Matching Database.

The constraint this module exists to satisfy: `mmm/teacher/search.py` must not
change. It already does the right thing -- it reads
`database.matching.normalized()`, checks the query width against the row width,
and excludes clips and rows -- and it is covered by
`tests/test_knn_matches_reference.py`. Widening the rows is not a reason to
touch it.

So instead of modifying `MatchingDatabase`, this wraps one. `AugmentedMatching`
exposes the same three things `KnnIndex` reads -- `normalized()`, `clip_ids`,
and a row count -- and returns 32 wide rows instead of 27. `KnnIndex` cannot
tell, and `augment` hands back something it accepts unaltered.

Row layout
----------
    0:27    the base features, exactly as `mmm.data.features` produces them
    27:32   the obstacle block, as `mmm.obstacles.features` produces it

Appended, never interleaved, so a base row is a strict prefix of an augmented
row. Two things follow from that, both of which matter:

- The same Animation Database serves both searches. Row index 4,219 means the
  same pose either way, so an obstacle-aware teacher and an obstacle-blind one
  can be compared on identical data.
- Dropping the obstacle block is a slice, not a rebuild. That makes the
  ablation -- does obstacle awareness actually change what gets retrieved? --
  a one line change rather than a second database.

Normalisation stays split
-------------------------
The base statistics are the ones already stored with the database and are not
recomputed, because changing them would change what the obstacle-blind search
retrieves and destroy the comparison that justifies this work. The obstacle
statistics are fitted separately and carry their own weights.
"""

from __future__ import annotations

import json
import os

import numpy as np

from mmm.data import features as base_features
from mmm.data.database import MotionDatabase
from mmm.obstacles import features as obstacle_features
from mmm.obstacles.features import ObstacleNormalization
from mmm.obstacles.geometry import BoxField

__all__ = [
    "AugmentedNormalization",
    "AugmentedMatching",
    "AugmentedDatabase",
    "augment",
    "OBSTACLE_OFFSET",
    "AUGMENTED_FEATURE_DIM",
    "save_fields",
    "load_fields",
]

# Where the obstacle block starts in an augmented row, and how wide a row is.
# Derived rather than written down, so that a change to either half cannot
# leave a stale constant behind.
OBSTACLE_OFFSET = base_features.FEATURE_DIM
AUGMENTED_FEATURE_DIM = OBSTACLE_OFFSET + obstacle_features.OBSTACLE_FEATURE_DIM


class AugmentedNormalization:
    """Normalisation for a full 32 wide row, delegating to each half.

    This class is what makes the controller subclass almost empty. Both
    `KnnController._search` and `KnnController._choose` call
    `database.matching.normalization.apply(query)`; if that call handles the
    wide vector, neither method needs overriding and `mmm/teacher/controller.py`
    stays untouched. Exposing the two halves separately instead would have
    forced an override of every method that normalises anything.

    Normalising in halves is identical to normalising the concatenation, since
    normalisation is per dimension. Doing it in halves is what keeps the base
    statistics byte identical to the obstacle-blind database.
    """

    def __init__(self, base, obstacle):
        self.base = base
        self.obstacle = obstacle

    def apply(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, np.float32)
        if features.shape[-1] != AUGMENTED_FEATURE_DIM:
            raise ValueError(
                f"expected {AUGMENTED_FEATURE_DIM} wide features "
                f"(27 base + 5 obstacle); got {features.shape[-1]}. A 27 wide "
                "vector here means an obstacle-blind query reached an "
                "obstacle-aware search."
            )
        return np.concatenate(
            [
                self.base.apply(features[..., :OBSTACLE_OFFSET]),
                self.obstacle.apply(features[..., OBSTACLE_OFFSET:]),
            ],
            axis=-1,
        ).astype(np.float32)

    def invert(self, normalized: np.ndarray) -> np.ndarray:
        normalized = np.asarray(normalized, np.float32)
        return np.concatenate(
            [
                self.base.invert(normalized[..., :OBSTACLE_OFFSET]),
                self.obstacle.invert(normalized[..., OBSTACLE_OFFSET:]),
            ],
            axis=-1,
        ).astype(np.float32)


class AugmentedMatching:
    """A Matching Database with the obstacle block appended.

    Duck typed to what `KnnIndex` reads rather than subclassing
    `MatchingDatabase`: subclassing would inherit a `normalized()` that
    validates against the 27 wide normalisation and a dataclass shape that
    assumes one feature array, and overriding both leaves a class that is a
    `MatchingDatabase` in name only.

    Args:
        base: the existing database, with its normalization already fitted.
        obstacle: `(n_rows, OBSTACLE_FEATURE_DIM)` unnormalized obstacle rows,
            in the same row order.
        normalization: obstacle statistics. Fitted on the valid rows of
            `obstacle` when omitted.
    """

    def __init__(self, base, obstacle: np.ndarray, normalization=None):
        obstacle = np.asarray(obstacle, np.float32)
        if obstacle.shape[0] != base.n_rows:
            raise ValueError(
                f"obstacle rows ({obstacle.shape[0]}) must align with the base "
                f"database ({base.n_rows}). Build them from the same clips in "
                "the same order -- a shifted alignment retrieves poses from the "
                "wrong frames and still appears to work."
            )
        if obstacle.shape[1] != obstacle_features.OBSTACLE_FEATURE_DIM:
            raise ValueError(
                f"obstacle rows are {obstacle.shape[1]} wide, expected "
                f"{obstacle_features.OBSTACLE_FEATURE_DIM}"
            )

        self.base = base
        self.obstacle = obstacle
        if normalization is None:
            # Fitted on valid rows only. The invalid tail rows hold clamped
            # lookahead values, and letting them into the statistics would
            # shift the mean of every dimension by whatever the last frame of
            # each clip happened to be standing on.
            valid = base.valid if base.valid.any() else np.ones(base.n_rows, bool)
            normalization = ObstacleNormalization.fit(obstacle[valid])
        self.obstacle_normalization = normalization
        # The attribute the controller reaches for, covering the whole row.
        self.normalization = AugmentedNormalization(base.normalization, normalization)

    # -- what KnnIndex reads ---------------------------------------------

    @property
    def n_rows(self) -> int:
        return int(self.base.n_rows)

    @property
    def clip_ids(self) -> np.ndarray:
        return self.base.clip_ids

    @property
    def frame_in_clip(self) -> np.ndarray:
        return self.base.frame_in_clip

    @property
    def valid(self) -> np.ndarray:
        return self.base.valid

    @property
    def features(self) -> np.ndarray:
        """Unnormalized augmented rows, `(n_rows, AUGMENTED_FEATURE_DIM)`."""
        return np.concatenate([self.base.features, self.obstacle], axis=-1)

    def normalized(self) -> np.ndarray:
        """Rows in the space the search compares.

        Each half is normalised by its own statistics and then concatenated,
        which is the same thing as normalising the concatenation would be,
        since normalisation is per dimension. Doing it in halves is what keeps
        the base statistics byte identical to the obstacle-blind database.

        Invalid rows are pushed far away exactly as `MatchingDatabase` does,
        and for the same reason: row indices must stay aligned with the
        Animation Database, so an unusable row is made unreachable rather than
        removed.
        """
        rows = np.concatenate(
            [
                self.base.normalization.apply(self.base.features),
                self.obstacle_normalization.apply(self.obstacle),
            ],
            axis=-1,
        )
        rows[~self.base.valid] = 1e6
        return rows.astype(np.float32)

    def base_only(self) -> np.ndarray:
        """The obstacle-blind rows, for the ablation.

        A slice of the same array rather than a separate database, so the two
        searches are guaranteed to be over the same frames in the same order.
        """
        return self.base.normalized()


class AugmentedDatabase:
    """A `MotionDatabase` whose matching half knows about obstacles.

    Holds the animation half unchanged -- the poses are the poses, and an
        obstacle does not alter them -- and carries the per clip box fields so
    that a later append, or a diagnostic, can ask which obstacles a given clip
    was captured with.
    """

    def __init__(self, base: MotionDatabase, matching: AugmentedMatching, fields=None):
        self.base = base
        self.matching = matching
        self.fields: dict[str, BoxField] = dict(fields or {})

    @property
    def animation(self):
        return self.base.animation

    @property
    def clips(self):
        return self.base.clips

    @property
    def skeleton(self):
        return getattr(self.base, "skeleton", None)

    def __getattr__(self, name):
        """Delegate anything not defined here to the wrapped database.

        `KnnController` calls `next_frame_of`, reads `frame_time` and asks for
        `n_rows`; none of those are affected by obstacles, and forwarding them
        is what lets the controller take this object in place of a
        `MotionDatabase` without `mmm/teacher/controller.py` changing.

        Only reached for attributes normal lookup did not find, so `matching`
        and `animation` above still win -- which is the point, since `matching`
        is the one thing that must *not* be the base version.
        """
        # Guard against recursion while unpickling or before __init__ runs.
        if name.startswith("_") or "base" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.base, name)

    def __repr__(self) -> str:
        return (
            f"AugmentedDatabase({self.matching.n_rows} rows, "
            f"{AUGMENTED_FEATURE_DIM} features, {len(self.fields)} clip fields)"
        )


def augment(
    database: MotionDatabase,
    motions,
    fields,
    horizons=base_features.TRAJECTORY_HORIZONS,
    max_distance: float = obstacle_features.MAX_EDGE_DISTANCE,
    block_weights: dict | None = None,
) -> AugmentedDatabase:
    """Add the obstacle block to an existing database.

    Args:
        database: a built `MotionDatabase` with fitted normalization. Not
            modified.
        motions: the clips it was built from, in the same order, so that the
            obstacle rows line up with the feature rows. Checked against the
            database's own clip table rather than trusted.
        fields: `{clip_name: BoxField}`. A clip with no entry is treated as
            flat ground, which is the correct description of a walk clip and
            means a mixed database needs no special casing.
        horizons: must match whatever the base features used.
        max_distance: edge search range; must match at query time.
        block_weights: optional `{block_name: weight}` for the obstacle block.

    Returns:
        An `AugmentedDatabase` that `KnnIndex` accepts unchanged.
    """
    expected = [clip.name for clip in database.clips]
    given = [m.name for m in motions]
    if given != expected:
        raise ValueError(
            "the motions given do not match the database's clip table.\n"
            f"  database: {expected[:4]}{' ...' if len(expected) > 4 else ''}\n"
            f"  given:    {given[:4]}{' ...' if len(given) > 4 else ''}\n"
            "Obstacle rows are matched to feature rows by position, so a "
            "different order would silently describe the wrong frames."
        )

    blocks = []
    for motion, clip in zip(motions, database.clips):
        field = fields.get(motion.name, BoxField())
        rows = obstacle_features.compute_obstacle_features(
            motion, field, horizons=horizons, max_distance=max_distance
        )
        if len(rows) != clip.length:
            raise ValueError(
                f"{motion.name}: produced {len(rows)} obstacle rows but the "
                f"database holds {clip.length} for this clip"
            )
        blocks.append(rows)

    obstacle = np.concatenate(blocks, axis=0) if blocks else np.zeros(
        (0, obstacle_features.OBSTACLE_FEATURE_DIM), np.float32
    )

    valid = database.matching.valid
    normalization = ObstacleNormalization.fit(
        obstacle[valid] if valid.any() else obstacle, block_weights
    )
    matching = AugmentedMatching(database.matching, obstacle, normalization)
    return AugmentedDatabase(database, matching, fields)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_fields(path: str, fields: dict) -> None:
    """Write `{clip_name: BoxField}` as JSON.

    JSON rather than npz: a recovered studio layout is a few dozen boxes, it is
    the thing a human most wants to read and hand correct, and a recovery that
    cannot be inspected will not be trusted.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {name: field.to_dicts() for name, field in fields.items()}
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def load_fields(path: str) -> dict:
    """Read what `save_fields` wrote."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {name: BoxField.from_dicts(records) for name, records in payload.items()}
