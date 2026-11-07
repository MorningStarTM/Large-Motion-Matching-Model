"""
Windowed training dataset for Phase 1.

One sample is a fixed length window of consecutive frames from one clip,
carrying everything the model reads and everything the loss compares against:

    joystick       (seq, 12)             from the clip's own real future path
    previous_pose  (seq, pose_dim)       the frame before each target
    candidates     (seq, K, pose_dim)    retrieved from the database
    mask           (seq, K)              1 where a candidate is real
    target         (seq, pose_dim)       the next pose, i.e. the label
    contacts       (seq, n_feet)         foot contact labels

Retrieval is precomputed per window rather than at training time: a KNN query
per frame inside the data loader would dominate the step time, and the result
is deterministic given the database, so there is nothing to gain from redoing
it every epoch.

Leave one clip out is applied here, at retrieval time, because that is where
the restriction actually bites: candidates for a frame of clip X are drawn from
every clip except X, so the model cannot satisfy the loss by copying itself.
"""

from __future__ import annotations

import os

import numpy as np

from mmm.data import features as feat
from mmm.data.contacts import detect_contacts
from mmm.data.contacts_elevated import detect_contacts_any_height
from mmm.data.database import MotionDatabase
from mmm.teacher.search import KnnIndex

__all__ = ["WindowDataset", "collate"]


class WindowDataset:
    """Fixed length windows over a database, with retrieval precomputed.

    Implements the plain `__len__` / `__getitem__` protocol, so it works with
    `torch.utils.data.DataLoader` without importing torch here -- which keeps
    the data layer usable in Phase 0, where torch is not a dependency.

    Args:
        database: the built database.
        window: frames per sample.
        stride: step between window starts. A stride below the window length
            overlaps samples, which is usually what you want: it multiplies the
            number of training sequences without adding data.
        k: candidates to retrieve per frame.
        leave_one_clip_out: whether a frame's candidates may come from its own
            clip. False is the Phase 1 setting.
        retrieval_interval: retrieve every this many frames and hold the result
            in between, mirroring the runtime's 10 frame search. Retrieving
            fresh every frame would train the model on information it will not
            have.
        clips: restrict to these clip names. Used to hold out styles and to
            split train from validation.
        seed: for the window order when shuffling is done here.
        use_retrieval: whether to retrieve candidates at all. False is the
            Phase 1 setting: pre-training learns the motion prior from
            (joystick, previous pose) alone, so the candidate block is not
            built and the whole retrieval precompute -- seven minutes on
            264,000 rows -- is skipped. Samples then carry no `candidates` or
            `mask` key.
        cache_dir: where to keep the precomputed retrieval. The precompute
            costs about seven minutes on 264,000 rows and depends only on the
            database and on `k`, `leave_one_clip_out` and `retrieval_interval`,
            so caching it turns every run after the first into a second. The
            file name carries a fingerprint of all four, so a changed database
            or a changed setting recomputes rather than silently reusing a
            stale cache. Passing None recomputes every time.
    """

    def __init__(
        self,
        database: MotionDatabase,
        window: int = 64,
        stride: int = 32,
        k: int = 4,
        leave_one_clip_out: bool = True,
        retrieval_interval: int = 10,
        clips=None,
        index: KnnIndex | None = None,
        seed: int = 0,
        cache_dir: str | None = None,
        use_retrieval: bool = True,
        elevated_contacts: bool = False,
    ):
        if database.matching.normalization is None:
            raise ValueError("database normalization has not been fitted")

        self.database = database
        self.window = window
        self.stride = stride
        self.k = k
        self.leave_one_clip_out = leave_one_clip_out
        self.retrieval_interval = retrieval_interval
        self.rng = np.random.RandomState(seed)

        self.index = index if index is not None else KnnIndex(database, backend="kdtree")
        self.n_joints = database.skeleton.n_joints
        self.pose_dim = database.pose_dim

        allowed = set(clips) if clips is not None else None
        self.windows: list[tuple[int, int]] = []
        for clip_id, clip in enumerate(database.clips):
            if allowed is not None and clip.name not in allowed:
                continue
            # A window needs `window` inputs plus one more frame for the last
            # target, and every frame in it must have a valid future
            # trajectory, which the last max(horizon) frames of a clip do not.
            usable = clip.length - 1
            last_valid = usable
            valid_slice = database.matching.valid[clip.start : clip.start + usable]
            if not valid_slice.all():
                invalid = np.flatnonzero(~valid_slice)
                last_valid = int(invalid[0]) if len(invalid) else usable
            for start in range(0, max(0, last_valid - window + 1), stride):
                self.windows.append((clip_id, start))

        if not self.windows:
            raise ValueError(
                f"no windows: every clip is shorter than window={window} plus the "
                f"{max(feat.TRAJECTORY_HORIZONS)} frame trajectory lookahead"
            )

        self.cache_dir = cache_dir
        self.use_retrieval = use_retrieval
        self.elevated_contacts = elevated_contacts
        self._contacts = self._compute_contacts()
        if use_retrieval:
            self._candidate_next, self._candidate_mask = self._load_or_precompute()
        else:
            self._candidate_next = self._candidate_mask = None

    # -- contacts ---------------------------------------------------------

    def _compute_contacts(self) -> np.ndarray:
        """Foot contact labels for every row, `(n_rows, 2)`.

        Computed once here rather than stored in the database, because contacts
        are a training target rather than part of the pose, and the thresholds
        are a data config decision that may change without rebuilding the
        database.
        """
        from mmm.data.database import pose_to_hips_height, pose_to_rotations
        from mmm.data.fk import forward_kinematics

        skeleton = self.database.skeleton
        poses = self.database.animation.poses
        matrices = pose_to_rotations(poses, self.n_joints)
        heights = pose_to_hips_height(poses, self.n_joints)
        root = np.zeros((len(poses), 3), np.float32)
        root[:, 1] = heights
        positions, _ = forward_kinematics(skeleton, matrices, root)
        if self.elevated_contacts:
            # `detect_contacts` cannot label a foot planted above about 7 cm,
            # so on the obstacle clips it marks roughly 40% of frames "not in
            # contact" while the actor is standing on a prop. This adds those
            # back; the level-ground labels are untouched.
            return detect_contacts_any_height(
                positions, skeleton, self.database.frame_time
            )
        return detect_contacts(positions, skeleton, self.database.frame_time)

    # -- retrieval --------------------------------------------------------

    def _cache_path(self) -> str | None:
        if self.cache_dir is None:
            return None
        from mmm.teacher.logger import database_fingerprint

        name = (
            f"retrieval_{database_fingerprint(self.database)}"
            f"_k{self.k}_loo{int(self.leave_one_clip_out)}"
            f"_int{self.retrieval_interval}.npz"
        )
        return os.path.join(self.cache_dir, name)

    def _load_or_precompute(self) -> tuple[np.ndarray, np.ndarray]:
        path = self._cache_path()
        if path is not None and os.path.exists(path):
            with np.load(path) as handle:
                matched, mask = handle["next_rows"], handle["mask"]
            if matched.shape == (self.database.n_rows, self.k):
                return matched, mask
            # A shape mismatch means the fingerprint collided or the file was
            # written by a different version. Recompute rather than guess.

        matched, mask = self._precompute_candidates()

        if path is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Write beside the target and rename, so an interrupted run cannot
            # leave a half written cache that a later run would trust. The
            # suffix has to stay `.npz`: `np.savez` silently appends it to any
            # name that lacks it, and the rename would then miss its target.
            temporary = path + ".partial.npz"
            np.savez(temporary, next_rows=matched, mask=mask)
            os.replace(temporary, path)
        return matched, mask

    def _precompute_candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve for every row once, up front.

        Returns `(next_rows, mask)`, both `(n_rows, k)`. `next_rows` already
        holds the *next* frame of each match: a matched row describes a state,
        and what it implies for the following frame is the row after it, which
        is what the model is asked to blend.

        Doing this per `__getitem__` instead -- which is what this class did
        first, despite its docstring claiming otherwise -- cost 317 ms a
        sample, or 36 minutes an epoch of data loading alone, against a 45
        second epoch of actual compute. Two separate causes: one KNN query per
        search frame, around 57,000 queries an epoch, and
        `MatchingDatabase.normalized()` rebuilding its whole 28 MB array inside
        every single call.

        Leave one clip out is done by building one tree per excluded clip and
        querying that clip's own rows against it. The obvious alternative,
        over-fetching from a single tree and filtering, does not work here: a
        frame's nearest neighbours are overwhelmingly its own temporal
        neighbours, so surviving the removal of a whole 14,000 row clip would
        need an over-fetch of hundreds, and `(n_rows, hundreds)` does not fit
        in memory. One tree per clip is both exact and cheaper.
        """
        from scipy.spatial import cKDTree

        database = self.database
        rows = database.matching.normalized()
        n_rows = len(rows)
        k = self.k

        matched = np.zeros((n_rows, k), np.int64)
        mask = np.zeros((n_rows, k), np.float32)

        if not self.leave_one_clip_out:
            tree = cKDTree(rows)
            # k + 1, then drop the self match, so a frame is not handed itself
            # as its own target. Without leave one clip out that is the only
            # thing stopping the loss being satisfied by copying.
            _, found = tree.query(rows, k=k + 1, workers=-1)
            found = np.atleast_2d(found)[:, 1:]
            matched[:] = found
            mask[:] = 1.0
        else:
            index = np.arange(n_rows)
            for clip in database.clips:
                keep = np.ones(n_rows, bool)
                keep[clip.start : clip.stop] = False
                kept = index[keep]
                if len(kept) < k:
                    raise ValueError(
                        f"leave one clip out leaves only {len(kept)} rows for "
                        f"clip {clip.name!r}; a database of one clip cannot be "
                        "used with leave_one_clip_out=True"
                    )
                tree = cKDTree(rows[kept])
                _, found = tree.query(rows[clip.start : clip.stop], k=k, workers=-1)
                matched[clip.start : clip.stop] = kept[np.atleast_2d(found)]
                mask[clip.start : clip.stop] = 1.0

        return database.next_frame_of(matched.reshape(-1)).reshape(matched.shape), mask

    # -- the protocol -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        clip_id, start = self.windows[idx]
        clip = self.database.clips[clip_id]
        base = clip.start + start
        rows = np.arange(base, base + self.window)

        poses = self.database.animation.poses
        joystick = self.database.matching.features[
            rows, feat.feature_block_slice("future_positions").start : feat.feature_block_slice(
                "future_directions"
            ).stop
        ]

        sample = {
            "joystick": joystick.astype(np.float32),
            "previous_pose": poses[rows].astype(np.float32),
            "target": poses[rows + 1].astype(np.float32),
            "contacts": self._contacts[rows + 1].astype(np.float32),
            "clip_id": np.int64(clip_id),
        }

        if self.use_retrieval:
            # Hold each retrieval for `retrieval_interval` frames, mirroring
            # the runtime's 10 frame search. Retrieving fresh every frame would
            # train the model on information it will not have.
            anchors = base + (np.arange(self.window) // self.retrieval_interval) * (
                self.retrieval_interval
            )
            held = self._candidate_next[anchors]
            sample["candidates"] = poses[held].astype(np.float32)
            sample["mask"] = self._candidate_mask[anchors].astype(np.float32)

        return sample

    def _view(self, windows) -> "WindowDataset":
        """A dataset over a subset of the windows, sharing this one's arrays.

        Retrieval and contacts are properties of the database, not of the
        split, so recomputing them per half would double the setup cost and
        risk the two halves disagreeing.
        """
        other = object.__new__(WindowDataset)
        other.__dict__.update(self.__dict__)
        other.windows = list(windows)
        other.rng = np.random.RandomState(0)
        return other

    def split(self, validation_fraction: float = 0.1):
        """Split by clip, not by window.

        Splitting by window would put frames from the same clip on both sides,
        and consecutive frames of a 60Hz clip are nearly identical, so
        validation would simply measure memorisation.
        """
        names = [c.name for c in self.database.clips]
        order = self.rng.permutation(len(names))
        n_validation = max(1, int(round(len(names) * validation_fraction)))
        validation = {names[i] for i in order[:n_validation]}
        train = {names[i] for i in order[n_validation:]}

        by_name = {c.name: i for i, c in enumerate(self.database.clips)}
        train_ids = {by_name[n] for n in train}
        validation_ids = {by_name[n] for n in validation}

        # Reuse this dataset's own windows and precomputed arrays rather than
        # constructing two fresh datasets, which would retrieve the whole
        # database twice more.
        return (
            self._view([w for w in self.windows if w[0] in train_ids]),
            self._view([w for w in self.windows if w[0] in validation_ids]),
        )


def collate(batch):
    """Stack samples into tensors. Passed to `DataLoader(collate_fn=...)`."""
    import torch

    keys = [k for k in batch[0] if k != "clip_id"]
    out = {key: torch.from_numpy(np.stack([sample[key] for sample in batch])) for key in keys}
    out["clip_id"] = torch.tensor([int(sample["clip_id"]) for sample in batch])
    return out
