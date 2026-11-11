"""
Phase 2 training windows, built from logged teacher rollouts.

`mmm.data.dataset.WindowDataset` turns a `MotionDatabase` into supervised
windows by looking one frame ahead inside a clip of real mocap. Phase 2's
windows come from somewhere else entirely: `scripts/collect_teacher_data.py`
(round 1) drives the KNN teacher with a synthetic joystick and logs everything
it does, and `scripts/dagger_round.py` (round 2 onward) does the same while
driving the student instead and asking the teacher to relabel wherever it
actually ends up. Both write through `mmm.teacher.logger.RolloutWriter`. This
class reads those logs back into the exact dict shape `WindowDataset`
produces -- `joystick`, `previous_pose`, `candidates`, `mask`, `target`,
`contacts` -- so `mmm.training.pretrainer.PreTrainer` trains on either without
caring which.

Three things a rollout log does not store directly, and how they are rebuilt
----------------------------------------------------------------------------
1. **The model's actual input.** `RolloutWriter` logs the raw 2D stick
   (`"joystick"`) for provenance, but that is one input to the spring damper,
   not what `MmmPolicy` reads. The 12 numbers the policy actually sees --
   `TrajectoryPredictor.predict()`'s output -- are logged separately as
   `"joystick_vector"`, every frame, by the collection scripts. A rollout
   without it cannot be used here and this raises rather than silently
   falling back to the raw stick, which is a different, wrong signal.

2. **Held candidates.** The teacher searches every `search_interval` frames
   and holds the result in between, exactly like the runtime controller, so
   the log's `candidates` array is real only on search frames and -1 padded
   elsewhere. This forward-fills from the last search, and frames before the
   first search -- which have no valid candidate set at all -- are excluded.
   Candidate *rows* are kept as indices here, not expanded into poses: a
   window's indices are looked up against `database.animation.poses`, which
   is already resident in memory, rather than duplicating a `(k, pose_dim)`
   block per frame across the whole dataset.

3. **Contacts.** Not logged at all. Computed here from the *target* poses by
   forward kinematics and the same detector `WindowDataset` uses, over each
   rollout's full length before windowing -- not per window -- so the
   detector's hysteresis and minimum-duration smoothing see real continuity
   at every window boundary instead of an artificial edge.

A fourth thing, present only from a DAgger round, changes what "target" means
----------------------------------------------------------------------------
Round 1's `pose` array is both the state the teacher was in and the thing it
did next -- the teacher drives itself, so its own next output already is the
correct label. A DAgger round is different: `pose` there is the *student's*
own output, which is what has to keep driving the rollout so the student
visits its own real error distribution, but it is not the correct label for
what should happen from that state -- that is a separate array,
`label_pose`, the teacher's one-step correction at each state the student
actually reached (see `RolloutWriter.append`'s `label_pose` argument).

So `previous_pose` always comes from `pose` -- the real, possibly wrong state
the question is about -- while `target` comes from `label_pose` when a
rollout has one, and from `pose` shifted by one otherwise, which is exactly
Round 1's relationship and is what a rollout with no `label_pose` falls back
to.

Memory
------
Every rollout's reconstructed arrays are kept in memory for the dataset's
lifetime, the same tradeoff `WindowDataset` makes by keeping a whole
`MotionDatabase` resident. That is the right call at the scale a single
researcher collects on one machine -- tens of minutes to a few hours of 60Hz
data -- and the wrong one well beyond it, where this would need to memory-map
instead.
"""

from __future__ import annotations

import numpy as np

from mmm.data.contacts import detect_contacts
from mmm.data.contacts_elevated import detect_contacts_any_height
from mmm.data.database import MotionDatabase, pose_to_hips_height, pose_to_rotations
from mmm.data.fk import forward_kinematics
from mmm.teacher.logger import RolloutReader, database_fingerprint

__all__ = ["RolloutWindowDataset"]


class RolloutWindowDataset:
    """Fixed length windows over logged teacher rollouts.

    Implements the plain `__len__` / `__getitem__` protocol, like
    `mmm.data.dataset.WindowDataset`, and its samples use `collate` from that
    same module -- the two are interchangeable as far as `PreTrainer` is
    concerned.

    Args:
        database: the database the rollouts were recorded against. Supplies
            the skeleton, the frame time, the candidate poses (see point 2
            above), and the fingerprint every rollout's manifest is checked
            against.
        paths: rollout `.npz` files, a glob, or a directory -- passed
            straight to `RolloutReader`.
        window: frames per sample.
        stride: step between window starts.
        elevated_contacts: label a foot planted at any height rather than
            only below 7cm. The same argument `WindowDataset` and
            `scripts/train_phase1.py` expose, for the same reason: clips with
            an actor standing on a recovered prop need it, and the teacher
            that produced these rollouts was free to retrieve exactly those
            frames.
        check_fingerprint: refuse a rollout recorded against a different
            database. The one reason to turn this off is a deliberate,
            understood mismatch, which should be asked for by name rather
            than silently allowed.
    """

    def __init__(
        self,
        database: MotionDatabase,
        paths,
        window: int = 64,
        stride: int = 32,
        elevated_contacts: bool = False,
        check_fingerprint: bool = True,
    ):
        self.database = database
        self.window = window
        self.stride = stride
        self.n_candidates: int | None = None

        fingerprint = database_fingerprint(database) if check_fingerprint else None
        reader = RolloutReader(paths, expect_fingerprint=fingerprint)
        detect = detect_contacts_any_height if elevated_contacts else detect_contacts
        n_rows = len(database.animation.poses)

        self._rollouts: list[dict] = []
        self.windows: list[tuple[int, int]] = []
        self.source_paths: list[str] = []
        too_short = 0

        for path, rollout in zip(reader.paths, reader):
            n = len(rollout["pose"])
            if n < 2:
                too_short += 1
                continue
            if "joystick_vector" not in rollout:
                raise ValueError(
                    f"{path} has no 'joystick_vector'. It was logged by a "
                    "writer that never passed one to RolloutWriter.append -- "
                    "most likely an interactive viewer session, logged for "
                    "inspection rather than for training. "
                    "scripts/collect_teacher_data.py and "
                    "scripts/dagger_round.py both pass it. The raw 2D "
                    "'joystick' field in this file is a different signal and "
                    "cannot stand in for it."
                )

            pose = rollout["pose"]
            searched = rollout["searched"]
            first_search = int(np.argmax(searched)) if searched.any() else n

            # Forward-fill the last real search across the frames it covers,
            # mirroring KnnController.search_interval at runtime. Frames
            # before the first search have no anchor at all.
            anchor = np.where(searched, np.arange(n), -1)
            anchor = np.maximum.accumulate(anchor)
            held = rollout["candidates"][np.clip(anchor, 0, n - 1)]
            mask = (held >= 0).astype(np.float32)
            rows = np.clip(held, 0, max(n_rows - 1, 0)).astype(np.int64)

            if self.n_candidates is None and searched.any():
                self.n_candidates = int(held.shape[-1])

            # `label_pose`, from a DAgger round, is the teacher's correction
            # at each state -- the right thing for `target` to come from.
            # Absent, this falls back to `pose` itself, Round 1's case, where
            # the teacher's own next output already is the correct label.
            # `previous_pose` always comes from `pose`: that is the real
            # state -- the student's own, drift included, in a DAgger round
            # -- that the question is about, never the idealised one.
            targets = rollout.get("label_pose", pose)

            # Contacts describe the target pose, so they are computed from
            # the same sequence `target` is, before any slicing.
            matrices = pose_to_rotations(targets, database.skeleton.n_joints)
            heights = pose_to_hips_height(targets, database.skeleton.n_joints)
            root = np.zeros((n, 3), np.float32)
            root[:, 1] = heights
            positions, _ = forward_kinematics(database.skeleton, matrices, root)
            contacts = detect(positions, database.skeleton, database.frame_time)

            # Shifted by one throughout: entry index i describes the step
            # that predicts original frame i + 1 from original frame i, so
            # the first usable entry is `first_search - 1` rather than
            # `first_search`.
            entry = dict(
                joystick=rollout["joystick_vector"][1:],
                previous_pose=pose[:-1],
                target=targets[1:],
                contacts=contacts[1:].astype(np.float32),
                candidate_rows=rows[1:],
                mask=mask[1:],
            )
            usable_from = max(first_search - 1, 0)
            usable_to = len(entry["target"])

            rollout_id = len(self._rollouts)
            self._rollouts.append(entry)
            self.source_paths.append(path)

            for start in range(usable_from, usable_to - window + 1, stride):
                self.windows.append((rollout_id, start))

        if not self.windows:
            raise ValueError(
                f"no usable windows from {len(reader.paths)} rollout file(s) "
                f"({too_short} too short). Either every rollout never "
                "searched the database at all, or none were long enough "
                f"past its first search to fill a {window} frame window."
            )

    # -- the protocol -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        rollout_id, start = self.windows[idx]
        rollout = self._rollouts[rollout_id]
        stop = start + self.window

        rows = rollout["candidate_rows"][start:stop]
        candidates = self.database.animation.poses[rows].astype(np.float32)
        mask = rollout["mask"][start:stop].astype(np.float32)
        # Masked here as well as at write time: write time zeroes what was
        # genuinely absent (no search yet), while a window can still clip a
        # candidate whose row happened to equal the clip index 0 by
        # coincidence of the padding value, and the mask is what disambiguates
        # that from a real candidate, exactly the convention
        # `MmmPolicy.build_input` already applies to this same block.
        candidates = candidates * mask[..., None]

        return {
            "joystick": rollout["joystick"][start:stop].astype(np.float32),
            "previous_pose": rollout["previous_pose"][start:stop].astype(np.float32),
            "target": rollout["target"][start:stop].astype(np.float32),
            "contacts": rollout["contacts"][start:stop],
            "candidates": candidates,
            "mask": mask,
            "clip_id": np.int64(rollout_id),
        }

    # -- splitting ----------------------------------------------------------

    def split(self, validation_fraction: float = 0.1, seed: int = 0):
        """Split by rollout file, not by window.

        The same reasoning as `WindowDataset.split`: consecutive frames of a
        60Hz rollout are nearly identical, so splitting by window would let
        validation measure memorisation rather than generalisation.
        """
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(self._rollouts))
        n_validation = max(1, int(round(len(self._rollouts) * validation_fraction)))
        validation_ids = set(order[:n_validation].tolist())

        train = [(r, s) for r, s in self.windows if r not in validation_ids]
        validation = [(r, s) for r, s in self.windows if r in validation_ids]
        if not train or not validation:
            raise ValueError(
                f"{len(self._rollouts)} rollout(s) is too few to split at "
                f"validation_fraction={validation_fraction}; collect more, or "
                "pass windows from this dataset to a loader without splitting"
            )
        return self._view(train), self._view(validation)

    def _view(self, windows) -> "RolloutWindowDataset":
        other = object.__new__(RolloutWindowDataset)
        other.__dict__.update(self.__dict__)
        other.windows = list(windows)
        return other
