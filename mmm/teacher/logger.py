"""
Teacher rollout logging. Set up on day one, per plan section 5.1.

Every query, the top K candidates and the chosen frame are recorded from the
first time the teacher runs. Phase 2 is built entirely out of these logs, so
logging added later means collecting the data twice.

One rollout is one compressed `.npz` plus a JSON manifest. The manifest records
the database identity, the controller configuration and the joystick seed, so a
rollout can be reproduced and so a dataset built against an old database is
detected rather than silently reused.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from mmm.data.database import MotionDatabase

__all__ = ["RolloutWriter", "RolloutReader", "database_fingerprint"]


def database_fingerprint(database: MotionDatabase) -> str:
    """A short hash identifying a database's contents.

    Covers the row count, the clip list and the normalization, which is enough
    to catch the mistake that matters: training on logs recorded against a
    different database than the one now loaded.
    """
    digest = hashlib.sha256()
    digest.update(str(database.n_rows).encode())
    digest.update(str(database.skeleton.names).encode())
    for clip in database.clips:
        digest.update(f"{clip.name}:{clip.start}:{clip.length}".encode())
    if database.matching.normalization is not None:
        digest.update(database.matching.normalization.mean.tobytes())
        digest.update(database.matching.normalization.std.tobytes())
    return digest.hexdigest()[:16]


@dataclass
class RolloutWriter:
    """Accumulates per frame records and writes one rollout file.

    Fields are appended as Python lists and stacked once on close, which is
    faster and simpler than growing arrays, and the memory cost is bounded by
    the rollout length rather than the whole dataset.

    Args:
        path: output `.npz` path. The manifest goes beside it as `.json`.
        database: the database the teacher is running on, for the fingerprint.
        config: controller settings and anything else needed to reproduce.
        seed: the joystick generator seed.
    """

    path: str
    database: MotionDatabase | None = None
    config: dict = field(default_factory=dict)
    seed: int | None = None

    _joystick: list = field(default_factory=list, init=False, repr=False)
    _joystick_vector: list = field(default_factory=list, init=False, repr=False)
    _label_pose: list = field(default_factory=list, init=False, repr=False)
    _query: list = field(default_factory=list, init=False, repr=False)
    _candidates: list = field(default_factory=list, init=False, repr=False)
    _distances: list = field(default_factory=list, init=False, repr=False)
    _chosen: list = field(default_factory=list, init=False, repr=False)
    _searched: list = field(default_factory=list, init=False, repr=False)
    _pose: list = field(default_factory=list, init=False, repr=False)
    _origin: list = field(default_factory=list, init=False, repr=False)
    _forward: list = field(default_factory=list, init=False, repr=False)

    _k: int | None = field(default=None, init=False, repr=False)
    _feature_dim: int | None = field(default=None, init=False, repr=False)

    def append(
        self,
        stick: np.ndarray,
        record,
        joystick_vector: np.ndarray | None = None,
        label_pose: np.ndarray | None = None,
    ) -> None:
        """Record one frame.

        Args:
            stick: `(2,)` the raw joystick input that produced this frame. Kept
                for provenance and for the HUD, but it is not what the model
                reads: it is one input to the spring damper, not the damper's
                output.
            record: the `StepRecord` the controller returned.
            joystick_vector: `(mmm.data.features.JOYSTICK_DIM,)`, the future
                trajectory the model actually sees this frame --
                `TrajectoryPredictor.predict()`'s output, flattened. This is
                needed every frame, while `record.query` (which embeds the
                same 12 numbers as its first block) exists only on search
                frames. Optional and defaults to absent, so every existing
                caller that logs only for the viewer or for Phase 0 keeps
                working unchanged; Phase 2 collection is what passes it,
                since training on these logs needs the model's real input
                reconstructed exactly rather than re-derived from the raw
                stick after the fact.
            label_pose: `(pose_dim,)`, the teacher's correction for this
                exact frame -- only meaningful in a DAgger round, where
                `record.pose` is the *student's own* output (what actually
                drove the rollout forward, drift and all) and this is a
                second, different pose: what `KnnController.reference_pose`
                said the teacher would do from that same state. Logging both
                is the whole mechanism. `record.pose` stays the sequence
                `previous_pose` is read from when this dataset is trained on
                -- the student's real, possibly wrong state is the *question*
                -- while this, when present, is read as `target` -- the
                teacher's answer to that exact question, not to an idealised
                one. Optional and defaults to absent, which is Round 1's
                case: there the teacher drives itself, so its own next
                output already *is* the correct target and no second array
                is needed.
        """
        self._joystick.append(np.asarray(stick, np.float32).reshape(2))
        if joystick_vector is not None:
            self._joystick_vector.append(np.asarray(joystick_vector, np.float32).reshape(-1))
        if label_pose is not None:
            self._label_pose.append(np.asarray(label_pose, np.float32).reshape(-1))
        self._pose.append(record.pose)
        self._origin.append(record.origin)
        self._forward.append(record.forward)
        self._searched.append(bool(record.searched))
        self._chosen.append(int(record.chosen_row))

        # Non-search frames have no query or candidates. Padding them keeps one
        # rectangular array per field, which is what makes the file cheap to
        # memory map and stream; `searched` says which rows are real.
        if record.query is not None:
            if self._feature_dim is None:
                self._feature_dim = len(record.query)
            if self._k is None:
                self._k = len(record.candidates)
            self._query.append(np.asarray(record.query, np.float32))
            candidates = np.full(self._k, -1, np.int64)
            distances = np.full(self._k, np.inf, np.float32)
            candidates[: len(record.candidates)] = record.candidates
            distances[: len(record.distances)] = record.distances
            self._candidates.append(candidates)
            self._distances.append(distances)
        else:
            self._query.append(None)
            self._candidates.append(None)
            self._distances.append(None)

    def __len__(self) -> int:
        return len(self._pose)

    def close(self) -> dict:
        """Write the rollout and its manifest. Returns the manifest."""
        if not self._pose:
            raise ValueError("nothing to write: no frames were appended")

        feature_dim = self._feature_dim or 0
        k = self._k or 0

        query = np.zeros((len(self), feature_dim), np.float32)
        candidates = np.full((len(self), k), -1, np.int64)
        distances = np.full((len(self), k), np.inf, np.float32)
        for i, (q, c, d) in enumerate(zip(self._query, self._candidates, self._distances)):
            if q is not None:
                query[i] = q
                candidates[i] = c
                distances[i] = d

        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        arrays = dict(
            joystick=np.stack(self._joystick),
            pose=np.stack(self._pose),
            origin=np.stack(self._origin),
            forward=np.stack(self._forward),
            searched=np.array(self._searched, bool),
            chosen_row=np.array(self._chosen, np.int64),
            query=query,
            candidates=candidates,
            distances=distances,
        )

        if self._joystick_vector:
            # All or nothing: a partially populated array would silently
            # misalign with every other field, which is worse than not having
            # it, so a caller that only logs it sometimes gets a clear error
            # here rather than a dataset that is wrong in a way nothing checks.
            if len(self._joystick_vector) != len(self):
                raise ValueError(
                    f"joystick_vector was given on {len(self._joystick_vector)} "
                    f"of {len(self)} frames; it must be every frame or none"
                )
            arrays["joystick_vector"] = np.stack(self._joystick_vector)

        if self._label_pose:
            if len(self._label_pose) != len(self):
                raise ValueError(
                    f"label_pose was given on {len(self._label_pose)} of "
                    f"{len(self)} frames; it must be every frame or none"
                )
            arrays["label_pose"] = np.stack(self._label_pose)

        np.savez_compressed(self.path, **arrays)

        manifest = {
            "frames": len(self),
            "k": k,
            "feature_dim": feature_dim,
            "seed": self.seed,
            "config": self.config,
            "database_fingerprint": database_fingerprint(self.database) if self.database else None,
            "n_searches": int(np.sum(self._searched)),
        }
        manifest_path = os.path.splitext(self.path)[0] + ".json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest


class RolloutReader:
    """Reads logged rollouts back as training sequences.

    Hours of 60Hz rollouts are large, so files are opened lazily one at a time
    rather than concatenated in memory.

    Args:
        paths: a `.npz` file, a glob pattern, or a directory -- or a list of
            any mix of those, which is how Phase 2 points this at several
            rounds of collection at once: the original run 1 data plus every
            DAgger round's corrections, none of them thrown away between
            rounds.
        expect_fingerprint: if given, every rollout's manifest must carry this
            database fingerprint, and a mismatch raises. This is the guard
            against silently training on logs from a stale database.
    """

    def __init__(self, paths, expect_fingerprint: str | None = None):
        if isinstance(paths, str):
            paths = [paths]
        # Each entry is resolved on its own, not only when `paths` as a whole
        # is a bare string -- a caller passing a list of directories (exactly
        # what `argparse`'s `nargs="+"` hands back for a single directory
        # argument) must get the same expansion a lone string would, or a
        # directory silently becomes one nonexistent "file" whose made-up
        # `.json` sibling does not exist either, which used to produce an
        # empty manifest and a fingerprint of `None` rather than an error
        # naming the actual problem.
        resolved: list[str] = []
        for entry in paths:
            if os.path.isdir(entry):
                resolved.extend(sorted(glob.glob(os.path.join(entry, "*.npz"))))
            elif any(ch in entry for ch in "*?["):
                resolved.extend(sorted(glob.glob(entry)))
            else:
                resolved.append(entry)
        self.paths = resolved
        if not self.paths:
            raise FileNotFoundError("no rollout files found")
        self.expect_fingerprint = expect_fingerprint

    def manifests(self) -> list[dict]:
        out = []
        for path in self.paths:
            manifest_path = os.path.splitext(path)[0] + ".json"
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    out.append(json.load(handle))
            else:
                out.append({})
        return out

    def total_frames(self) -> int:
        return sum(m.get("frames", 0) for m in self.manifests())

    def __iter__(self):
        """Yield one rollout at a time, as a dict of arrays."""
        for path, manifest in zip(self.paths, self.manifests()):
            if self.expect_fingerprint is not None:
                found = manifest.get("database_fingerprint")
                if found != self.expect_fingerprint:
                    raise ValueError(
                        f"{path} was recorded against database {found!r}, but the "
                        f"loaded database is {self.expect_fingerprint!r}. Re-record "
                        "the rollouts or load the matching database."
                    )
            blob = np.load(path, allow_pickle=False)
            yield {key: blob[key] for key in blob.files}

    def windows(self, length: int, stride: int = 1):
        """Yield fixed length windows across every rollout.

        Windows never straddle two rollouts, since the hidden state and the
        character's continuity reset between them.
        """
        for rollout in self:
            n = len(rollout["pose"])
            for start in range(0, max(0, n - length + 1), stride):
                yield {key: value[start : start + length] for key, value in rollout.items()}
