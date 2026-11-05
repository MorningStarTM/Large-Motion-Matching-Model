"""
The two databases, and the clip append that makes the KB editable.

- `MatchingDatabase`: one feature row per frame, with normalization, clip
  boundaries and validity. Used for searching.
- `AnimationDatabase`: the full pose per frame. Used for drawing the character
  and as the retrieved context the model reads.
- `MotionDatabase`: both together, which is what everything else passes around,
  since a row index is only meaningful against the pair.

`append_clip` is the mechanism behind the main claim (plan section 4.6): compute
the clip's features, append its rows, extend the index. No encoder, no
retraining.

Pose layout
-----------
`pose_dim = n_joints * 6 + 7`, which for the 24 joint locomotion skeleton is
151, the "about 150" of plan section 4.3:

    0 : n_joints*6      joint rotations, 6D, local to each parent. The hips'
                        entry is yaw stripped -- expressed in the character
                        frame, so it keeps lean and twist but not facing.
    n_joints*6 + 0      hips height above the ground
    n_joints*6 + 1:4    root linear velocity, character space, units/second
    n_joints*6 + 4:7    root angular velocity, rotation vector, radians/second

Everything in a stored pose is therefore in character space, and the
character's world position and facing live only in the root frame the runtime
integrates. That split is what makes a retrieved row usable at any position and
any heading, and it is why the hips rotation is yaw stripped rather than stored
raw: keeping the facing in both places would count it twice, and a query built
from a raw world pose would land in a different space from the database rows it
is compared against.

The hips height is stored explicitly rather than integrated from a vertical
velocity. Integrating it would drift, and foot height relative to the ground is
what the matching features measure and what foot contact depends on, so it has
to be exact.

Memory is accounted for honestly: the full Animation Database stays resident,
about 50 MB for locomotion, the same as classic motion matching. The selling
point is editability, not memory. Compression (PCA or product quantization) is
future work and deliberately absent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from mmm.data import features as feat
from mmm.data import rotations, spaces
from mmm.data.bvh import Motion
from mmm.data.fk import forward_kinematics
from mmm.data.skeleton import Skeleton

__all__ = ["ClipInfo", "MatchingDatabase", "AnimationDatabase", "MotionDatabase", "pose_dim_for"]


def pose_dim_for(n_joints: int) -> int:
    """Pose width: 6D per joint, plus hips height, plus 6 root velocity numbers."""
    return n_joints * 6 + 7


@dataclass
class ClipInfo:
    """Provenance of one clip's block of rows."""

    name: str
    start: int
    length: int
    source: str = ""
    style: str = ""

    @property
    def stop(self) -> int:
        return self.start + self.length

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": self.start,
            "length": self.length,
            "source": self.source,
            "style": self.style,
        }


def motion_to_pose(motion: Motion) -> np.ndarray:
    """One clip's Animation Database rows, entirely in character space.

    Returns `(n_frames, pose_dim)` in the layout documented at the top of this
    module: yaw stripped 6D local rotations, hips height, then root linear and
    angular velocity.
    """
    _, global_rotations = forward_kinematics(motion.skeleton, motion.rotations, motion.root_positions)
    frame = spaces.extract_root_frames(global_rotations, motion.root_positions)
    linear, angular = spaces.root_velocities(frame, global_rotations[:, 0], motion.frame_time)

    # The hips' local rotation is also its global rotation, since it is the
    # root. Stripping the yaw is what puts the pose in character space; every
    # other joint is relative to its parent and so already is.
    local = motion.rotations.copy()
    local[:, 0] = spaces.strip_yaw(global_rotations[:, 0], frame.forward)

    hips_height = motion.root_positions[:, 1:2]

    sixd = rotations.matrix_to_sixd(local).reshape(motion.n_frames, -1)
    return np.concatenate([sixd, hips_height, linear, angular], axis=-1).astype(np.float32)


def pose_to_rotations(pose: np.ndarray, n_joints: int) -> np.ndarray:
    """Inverse of the rotation part of `motion_to_pose`.

    `pose` may be `(pose_dim,)` or `(..., pose_dim)`. Returns
    `(..., n_joints, 3, 3)`, re-orthonormalised, so a raw model output is
    always a valid rotation. The hips entry is in the character frame; pass it
    through `spaces.apply_yaw` to put it back into the world.
    """
    sixd = pose[..., : n_joints * 6].reshape(pose.shape[:-1] + (n_joints, 6))
    return rotations.sixd_to_matrix(sixd)


def pose_to_hips_height(pose: np.ndarray, n_joints: int) -> np.ndarray:
    """The hips height stored in a pose vector."""
    return pose[..., n_joints * 6]


def pose_to_root_velocity(pose: np.ndarray, n_joints: int) -> tuple[np.ndarray, np.ndarray]:
    """The linear and angular root velocity stored in a pose vector."""
    base = n_joints * 6 + 1
    return pose[..., base : base + 3], pose[..., base + 3 : base + 6]


def assemble_pose(
    local_rotations: np.ndarray,
    hips_height,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> np.ndarray:
    """Build a pose vector from its parts, the inverse of the `pose_to_*`
    readers. One place that knows the layout, so the controller and the model
    cannot disagree about it."""
    return np.concatenate(
        [
            rotations.matrix_to_sixd(local_rotations).reshape(-1),
            np.atleast_1d(np.asarray(hips_height, np.float32)).reshape(1),
            np.asarray(linear_velocity, np.float32).reshape(3),
            np.asarray(angular_velocity, np.float32).reshape(3),
        ]
    ).astype(np.float32)


@dataclass
class MatchingDatabase:
    """Feature rows, for searching."""

    features: np.ndarray
    valid: np.ndarray
    clip_ids: np.ndarray
    frame_in_clip: np.ndarray
    normalization: feat.FeatureNormalization | None = None

    @property
    def n_rows(self) -> int:
        return int(self.features.shape[0])

    def normalized(self) -> np.ndarray:
        """Rows in the space the search compares, i.e. normalized and weighted.

        Invalid rows are pushed far away rather than removed, so row indices
        stay aligned with the Animation Database. A search can therefore never
        return one, and no index remapping is needed anywhere.
        """
        if self.normalization is None:
            raise ValueError("normalization has not been fitted yet")
        rows = self.normalization.apply(self.features)
        rows[~self.valid] = 1e6
        return rows.astype(np.float32)


@dataclass
class AnimationDatabase:
    """Pose rows, for drawing."""

    poses: np.ndarray
    n_joints: int

    @property
    def n_rows(self) -> int:
        return int(self.poses.shape[0])

    @property
    def pose_dim(self) -> int:
        return int(self.poses.shape[1])


@dataclass
class MotionDatabase:
    """The Matching and Animation Databases, plus the clip index tying them.

    A row index is only meaningful against both at once, so they travel
    together and are built, saved and appended to as a unit.
    """

    skeleton: Skeleton
    frame_time: float
    matching: MatchingDatabase
    animation: AnimationDatabase
    clips: list[ClipInfo] = field(default_factory=list)

    # -- construction -----------------------------------------------------

    @classmethod
    def empty(cls, skeleton: Skeleton, frame_time: float) -> "MotionDatabase":
        n_joints = skeleton.n_joints
        return cls(
            skeleton=skeleton,
            frame_time=frame_time,
            matching=MatchingDatabase(
                features=np.zeros((0, feat.FEATURE_DIM), np.float32),
                valid=np.zeros((0,), bool),
                clip_ids=np.zeros((0,), np.int32),
                frame_in_clip=np.zeros((0,), np.int32),
            ),
            animation=AnimationDatabase(
                poses=np.zeros((0, pose_dim_for(n_joints)), np.float32),
                n_joints=n_joints,
            ),
            clips=[],
        )

    @classmethod
    def build(cls, motions, block_weights=None, source: str = "") -> "MotionDatabase":
        """Build from a list of clips, fitting normalization on all of them."""
        motions = list(motions)
        if not motions:
            raise ValueError("cannot build a database from zero clips")

        database = cls.empty(motions[0].skeleton, motions[0].frame_time)
        for motion in motions:
            database.append_clip(motion, source=source, refit=False)
        database.fit_normalization(block_weights=block_weights)
        return database

    # -- the editable part ------------------------------------------------

    def append_clip(
        self,
        motion: Motion,
        source: str = "",
        style: str = "",
        refit: bool = False,
    ) -> ClipInfo:
        """Append one clip's rows to both databases.

        This is plan section 4.6 in code: no encoder, no retraining. Existing
        rows are untouched, so indices already held elsewhere stay valid.

        Args:
            motion: the clip, on this database's skeleton.
            source: dataset name, for provenance.
            style: style label, used by the held out style experiment.
            refit: whether to re-measure the normalization statistics
                afterwards. Default False, because the point of the editability
                claim is that adding a clip changes nothing about how the model
                or the search behaves; re-fitting would shift every existing
                row's normalized position.

        Returns:
            The `ClipInfo` describing the appended block.
        """
        self.skeleton.assert_matches(motion.skeleton)
        if abs(motion.frame_time - self.frame_time) > 1e-6:
            raise ValueError(
                f"{motion.name}: frame time {motion.frame_time} does not match "
                f"the database's {self.frame_time}. Resample the clip first."
            )

        rows, valid = feat.compute_features(motion)
        poses = motion_to_pose(motion)
        if len(rows) != len(poses):
            raise AssertionError("feature and pose row counts disagree")

        start = self.matching.n_rows
        clip_id = len(self.clips)
        info = ClipInfo(
            name=motion.name, start=start, length=len(rows), source=source, style=style
        )

        self.matching.features = np.concatenate([self.matching.features, rows])
        self.matching.valid = np.concatenate([self.matching.valid, valid])
        self.matching.clip_ids = np.concatenate(
            [self.matching.clip_ids, np.full(len(rows), clip_id, np.int32)]
        )
        self.matching.frame_in_clip = np.concatenate(
            [self.matching.frame_in_clip, np.arange(len(rows), dtype=np.int32)]
        )
        self.animation.poses = np.concatenate([self.animation.poses, poses])
        self.clips.append(info)

        if refit:
            self.fit_normalization()

        return info

    def fit_normalization(self, block_weights=None) -> None:
        """Measure per dimension statistics on the valid rows only.

        Invalid rows hold clamped lookahead values that are not real motion, so
        including them would skew the statistics for every genuine row.
        """
        valid_rows = self.matching.features[self.matching.valid]
        if len(valid_rows) == 0:
            raise ValueError("no valid rows to fit normalization on")
        self.matching.normalization = feat.FeatureNormalization.fit(
            valid_rows, block_weights=block_weights
        )

    # -- lookups ----------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return self.matching.n_rows

    @property
    def pose_dim(self) -> int:
        return self.animation.pose_dim

    def next_frame_of(self, rows) -> np.ndarray:
        """The row after each given row, clamped at clip boundaries.

        The model may read the candidates' next frames rather than the
        candidates themselves, and a clip's last row has no successor; clamping
        rather than wrapping avoids stitching the end of one clip onto the
        start of another.
        """
        rows = np.asarray(rows, dtype=np.int64)
        clip_ids = self.matching.clip_ids[rows]
        frame = self.matching.frame_in_clip[rows]
        lengths = np.array([c.length for c in self.clips], dtype=np.int64)
        last = lengths[clip_ids] - 1
        return rows + (frame < last).astype(np.int64)

    def clip_of(self, row: int) -> ClipInfo:
        return self.clips[int(self.matching.clip_ids[row])]

    def rows_of_clip(self, name: str) -> np.ndarray:
        for clip_id, info in enumerate(self.clips):
            if info.name == name:
                return np.arange(info.start, info.stop)
        raise KeyError(f"no clip named {name!r} in this database")

    def memory_bytes(self) -> dict[str, int]:
        """Resident size of each array, for the honest memory reporting the
        thesis needs."""
        return {
            "matching_features": self.matching.features.nbytes,
            "animation_poses": self.animation.poses.nbytes,
            "index": (
                self.matching.clip_ids.nbytes
                + self.matching.frame_in_clip.nbytes
                + self.matching.valid.nbytes
            ),
        }

    # -- persistence ------------------------------------------------------

    def save(self, directory: str) -> None:
        """Write both databases plus a JSON manifest.

        The manifest is what makes a database traceable: it records the
        skeleton, the feature layout, the normalization and every clip's
        provenance. A stale database built against a different feature layout
        is then detected on load rather than silently reused.
        """
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "matching.npy"), self.matching.features)
        np.save(os.path.join(directory, "animation.npy"), self.animation.poses)
        np.savez_compressed(
            os.path.join(directory, "index.npz"),
            valid=self.matching.valid,
            clip_ids=self.matching.clip_ids,
            frame_in_clip=self.matching.frame_in_clip,
            parents=self.skeleton.parents,
            offsets=self.skeleton.offsets,
            mean=self.matching.normalization.mean if self.matching.normalization else np.zeros(0),
            std=self.matching.normalization.std if self.matching.normalization else np.zeros(0),
            weights=self.matching.normalization.weights if self.matching.normalization else np.zeros(0),
        )
        manifest = {
            "frame_time": self.frame_time,
            "n_rows": self.n_rows,
            "n_joints": self.skeleton.n_joints,
            "pose_dim": self.pose_dim,
            "joint_names": self.skeleton.names,
            "feature_dim": feat.FEATURE_DIM,
            "feature_blocks": [list(b) for b in feat.FEATURE_BLOCKS],
            "trajectory_horizons": list(feat.TRAJECTORY_HORIZONS),
            "normalized": self.matching.normalization is not None,
            "memory_bytes": self.memory_bytes(),
            "clips": [c.to_dict() for c in self.clips],
        }
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    @classmethod
    def load(cls, directory: str) -> "MotionDatabase":
        with open(os.path.join(directory, "manifest.json"), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest["feature_dim"] != feat.FEATURE_DIM:
            raise ValueError(
                f"{directory} was built with feature_dim={manifest['feature_dim']}, "
                f"but this code uses {feat.FEATURE_DIM}. Rebuild the database."
            )
        if list(manifest["trajectory_horizons"]) != list(feat.TRAJECTORY_HORIZONS):
            raise ValueError(
                f"{directory} was built with horizons {manifest['trajectory_horizons']}, "
                f"but this code uses {list(feat.TRAJECTORY_HORIZONS)}. Rebuild it."
            )

        index = np.load(os.path.join(directory, "index.npz"), allow_pickle=False)
        skeleton = Skeleton(
            names=list(manifest["joint_names"]),
            parents=index["parents"],
            offsets=index["offsets"],
        )
        normalization = None
        if manifest["normalized"]:
            normalization = feat.FeatureNormalization(
                mean=index["mean"], std=index["std"], weights=index["weights"]
            )

        return cls(
            skeleton=skeleton,
            frame_time=float(manifest["frame_time"]),
            matching=MatchingDatabase(
                features=np.load(os.path.join(directory, "matching.npy")),
                valid=index["valid"],
                clip_ids=index["clip_ids"],
                frame_in_clip=index["frame_in_clip"],
                normalization=normalization,
            ),
            animation=AnimationDatabase(
                poses=np.load(os.path.join(directory, "animation.npy")),
                n_joints=int(manifest["n_joints"]),
            ),
            clips=[ClipInfo(**c) for c in manifest["clips"]],
        )

    def __repr__(self) -> str:
        megabytes = sum(self.memory_bytes().values()) / 1e6
        return (
            f"MotionDatabase({self.n_rows} rows from {len(self.clips)} clips, "
            f"{self.skeleton.n_joints} joints, pose_dim={self.pose_dim}, "
            f"{megabytes:.1f} MB)"
        )
