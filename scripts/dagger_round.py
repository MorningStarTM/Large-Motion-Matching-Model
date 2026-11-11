"""
Phase 2, round 2 onward: let the student make its own mistakes, then ask the
teacher how to recover from exactly those.

Round 1 (`scripts/collect_teacher_data.py`) has the teacher drive itself and
writes down every situation and move. This script flips who drives. The
*student* -- the model being trained, loaded from a checkpoint -- plays an
entire session on its own: its own joystick response, its own drift, fed back
into itself exactly as it would be at runtime
(`mmm.models.runtime.ModelController`, including searching the database on
its own behalf now that it has retrieval in its input). Nobody corrects it
while it runs.

At every single frame, though, a snapshot is taken of exactly where the
student really is -- mistakes and all -- and handed to the teacher with one
question: "if you were standing like this, what would you do next?"
(`KnnController.reference_pose`, which answers that from any pose with no
side effects, so it can be asked at every step without disturbing anything).
The teacher's answer becomes a new label, paired with the student's own real
state rather than an idealised one. That pairing is the entire mechanism: it
is specifically the mistakes the student is making *right now*, with its
*current* skill level, that get corrected, which is why repeating this for a
few rounds keeps shrinking what is left to fix.

Nothing here retrains anything. This only produces a new directory of rollout
files -- the correction dataset for one round -- which
`scripts/train_phase2.py` then fine tunes on, pointed at every round's
rollouts at once:

    python scripts/dagger_round.py \\
        --checkpoint runs/phase2/best.safetensors \\
        --dataset data/bvh/bvh \\
        --frames 200000 \\
        --out runs/rollouts/dagger1

    python scripts/train_phase2.py \\
        --dataset data/bvh/bvh \\
        --rollouts runs/rollouts/round1 runs/rollouts/dagger1 \\
        --phase1-checkpoint runs/phase1/published.safetensors \\
        --resume runs/phase2/best.safetensors \\
        --epochs 10 \\
        --out runs/phase2_round1

`--phase1-checkpoint` is still required on that second command even though
`--resume` immediately overwrites every weight it produces: `train_phase2.py`
always performs the Phase 1 handoff first so that a *fresh* run never needs a
separate code path, and `--resume` is simply what wins when both are given.
The handoff's own few milliseconds of wasted work is the price of that one
code path, not a correctness issue.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mmm.data import features as feat
from mmm.data.database import MotionDatabase
from mmm.data.fk import forward_kinematics
from mmm.data.prepare import LOCOMOTION_PATTERN, ensure_database
from mmm.envs.joystick import JoystickConfig, SyntheticJoystick
from mmm.models.runtime import ModelController, load_policy_from_manifest
from mmm.teacher.controller import KnnController, StepRecord
from mmm.teacher.logger import RolloutWriter, database_fingerprint
from mmm.teacher.search import KnnIndex
from mmm.workspace import Workspace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    data = parser.add_argument_group("data (must match the checkpoint's own training database)")
    data.add_argument("--dataset", required=True)
    data.add_argument("--pattern", default=LOCOMOTION_PATTERN)
    data.add_argument("--fps", type=float, default=60.0)
    data.add_argument("--max-frames", type=int, default=0)
    data.add_argument("--full-skeleton", action="store_true")
    data.add_argument("--trim-calibration", action="store_true")

    model = parser.add_argument_group("student")
    model.add_argument("--checkpoint", required=True, help="the student to roll out")
    model.add_argument("--manifest", default="", help="defaults to manifest.json beside --checkpoint")
    model.add_argument("--device", default="cpu")
    model.add_argument("--root-source", default="hybrid", choices=["hybrid", "simulation", "animation"])

    teacher = parser.add_argument_group("teacher (relabels every frame)")
    teacher.add_argument("--search-interval", type=int, default=10, help="student's own search cadence")
    teacher.add_argument("--k", type=int, default=4)

    paths = parser.add_argument_group("paths")
    paths.add_argument("--out", default="")
    paths.add_argument("--work-dir", default="")
    paths.add_argument("--database-dir", default="")
    paths.add_argument("--cache-dir", default="")
    paths.add_argument("--run-name", default="phase2-dagger")

    episodes = parser.add_argument_group("episodes")
    episodes.add_argument(
        "--episode-frames",
        type=int,
        default=1800,
        help="reset the student to a fresh random starting pose this often. "
        "See scripts/collect_teacher_data.py's --episode-frames for why this "
        "is on by default rather than one long session.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--frames", type=int, required=True)
    output.add_argument("--shard-frames", type=int, default=36000)
    output.add_argument("--seed", type=int, default=0)

    return parser.parse_args(argv)


def existing_shards(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    highest = -1
    for name in os.listdir(directory):
        if name.startswith("shard_") and name.endswith(".npz"):
            try:
                highest = max(highest, int(name[len("shard_") : -len(".npz")]))
            except ValueError:
                continue
    return highest + 1


def random_seed_pose(database: MotionDatabase, rng: np.random.RandomState):
    """A real, valid pose to reset the student to."""
    valid = np.flatnonzero(database.matching.valid)
    row = int(valid[rng.randint(len(valid))]) if len(valid) else 0
    origin = np.zeros(3, np.float32)
    forward = np.array([0.0, 1.0], np.float32)
    return database.animation.poses[row].copy(), origin, forward


def main(argv=None) -> int:
    args = parse_args(argv)

    workspace = Workspace.resolve(
        out=args.out or None,
        database=args.database_dir or None,
        cache=args.cache_dir or None,
        root=args.work_dir or None,
        run_name=args.run_name,
    )
    print(workspace)

    database_path = ensure_database(
        source=args.dataset,
        out_dir=workspace.database,
        cache_dir=workspace.cache,
        pattern=args.pattern,
        fps=args.fps,
        max_frames=args.max_frames,
        full_skeleton=args.full_skeleton,
        trim_calibration=args.trim_calibration,
    )
    database = MotionDatabase.load(database_path)
    print(database)

    index = KnnIndex(database)
    teacher = KnnController(database, index=index, search_interval=10_000_000, k=args.k)
    # search_interval is effectively infinite: this controller is never
    # stepped, so nothing would ever trigger a search on its own anyway --
    # reference_pose is called directly, by hand, every frame below. The
    # large number is only so the intent reads correctly if anyone inspects
    # the object rather than relying on an implementation detail.

    rng = np.random.RandomState(args.seed)
    seed_pose, origin, forward = random_seed_pose(database, rng)

    policy, architecture, _ = load_policy_from_manifest(
        args.checkpoint, manifest=args.manifest or None, device=args.device
    )
    if not architecture.get("use_retrieval"):
        print(
            "note: this checkpoint has no retrieval in its input -- it looks "
            "like a Phase 1 checkpoint, not a Phase 2 one. DAgger relabels a "
            "model that already searches the database; "
            "scripts/collect_teacher_data.py is round 1, which is what a "
            "Phase 1 checkpoint needs before this.",
            file=sys.stderr,
        )
        return 1

    student = ModelController(
        policy, database.skeleton, seed_pose,
        frame_time=database.frame_time, device=args.device,
        root_source=args.root_source,
        database=database, index=index,
        search_interval=args.search_interval, k=args.k,
    )
    student.reset(origin=origin, forward=forward, seed_pose=seed_pose)

    joystick = SyntheticJoystick(
        frame_time=database.frame_time, config=JoystickConfig(), seed=args.seed
    )

    shard_index = existing_shards(workspace.out)
    if shard_index:
        print(f"resuming: {shard_index} shard(s) already in {workspace.out}, continuing after them")

    frames_done = 0
    since_episode_reset = 0
    started = time.time()

    while frames_done < args.frames:
        shard_path = os.path.join(workspace.out, f"shard_{shard_index:05d}")
        shard_frames = min(args.shard_frames, args.frames - frames_done)

        writer = RolloutWriter(
            path=shard_path,
            database=database,
            seed=args.seed,
            config={
                "source": "dagger_round",
                "checkpoint": args.checkpoint,
                "search_interval": args.search_interval,
                "k": args.k,
                "root_source": args.root_source,
                "episode_frames": args.episode_frames,
            },
        )

        for _ in range(shard_frames):
            if args.episode_frames and since_episode_reset >= args.episode_frames:
                seed_pose, origin, forward = random_seed_pose(database, rng)
                student.reset(origin=origin, forward=forward, seed_pose=seed_pose)
                since_episode_reset = 0

            # The question: exactly where the student really is right now,
            # before this frame moves it. Captured as plain numpy, in the
            # stored pose layout, which is what reference_pose expects.
            state_before = student._previous_pose[0].detach().cpu().numpy().astype(np.float32)
            forward_before_turn = student.state.forward.copy()

            stick, facing = joystick.step()
            record = student.step(stick, facing)

            # The desired path for THIS frame, in the SAME character-space
            # frame the student's own policy.step just used internally --
            # ModelController computes this before applying the turn this
            # frame produces, so it must be read off the forward captured
            # before step(), not the (already turned) one step() leaves
            # behind. See scripts/collect_teacher_data.py for the same
            # reasoning, and the measurement that found it: using the
            # post-step forward was off by as much as 31 of these numbers on
            # a clip that was simply turning.
            desired_positions, desired_directions = student.trajectory.predict(forward_before_turn)

            label_pose, result = teacher.reference_pose(
                pose=state_before,
                desired_positions=desired_positions,
                desired_directions=desired_directions,
            )

            # Rebuilt once more for logging only -- reference_pose does not
            # hand its own query back out, and RolloutWriter needs a non-None
            # query to know a frame was searched at all, which is what makes
            # it keep this frame's candidates rather than padding them away.
            # Mirrors exactly what reference_pose computes internally.
            local_rotations = feat_pose_to_rotations(state_before, database.skeleton.n_joints)
            hips_height = feat_pose_to_hips_height(state_before, database.skeleton.n_joints)
            root = np.array([[0.0, hips_height, 0.0]], np.float32)
            positions, _ = forward_kinematics(database.skeleton, local_rotations[None], root)
            foot_positions = np.stack(
                [positions[0, teacher.left_foot], positions[0, teacher.right_foot]]
            )
            query = feat.build_query(
                foot_positions=foot_positions,
                foot_velocities=np.zeros((2, 3), np.float32),
                hip_velocity=feat_pose_to_root_velocity(state_before, database.skeleton.n_joints)[0],
                desired_positions=desired_positions,
                desired_directions=desired_directions,
            )

            teacher_record = StepRecord(
                pose=record.pose,
                local_rotations=record.local_rotations,
                origin=record.origin,
                forward=record.forward,
                searched=True,
                jumped=False,
                query=query,
                candidates=result.rows,
                distances=result.distances,
                chosen_row=result.best,
            )
            joystick_vector = np.concatenate(
                [
                    np.asarray(desired_positions, np.float32).reshape(-1),
                    np.asarray(desired_directions, np.float32).reshape(-1),
                ]
            ).astype(np.float32)
            writer.append(
                stick, teacher_record, joystick_vector=joystick_vector, label_pose=label_pose
            )

            since_episode_reset += 1

        manifest = writer.close()
        frames_done += shard_frames
        shard_index += 1

        rate = frames_done / max(time.time() - started, 1e-9)
        eta = (args.frames - frames_done) / max(rate, 1e-9)
        print(
            f"shard {shard_index - 1:05d}: {manifest['frames']:6d} frames | "
            f"{frames_done:,}/{args.frames:,} total "
            f"({100 * frames_done / args.frames:5.1f}%) | "
            f"{rate:6.0f} fps | eta {eta / 60:5.1f} min"
        )

    mix = joystick.realised_mix()
    print("\nrealised primitive mix (fraction of frames):")
    for name, fraction in mix.items():
        print(f"  {name:12s} {100 * fraction:5.1f}%")

    with open(os.path.join(workspace.out, "round_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "frames": frames_done,
                "shards": shard_index,
                "seed": args.seed,
                "checkpoint": args.checkpoint,
                "database_fingerprint": database_fingerprint(database),
                "realised_mix": mix,
                "args": vars(args),
            },
            handle,
            indent=2,
        )

    print(f"\nwrote {shard_index} shard(s), {frames_done:,} frames, to {workspace.out}")
    return 0


# Imported with local names to keep the per-frame loop above readable; these
# are exactly mmm.data.database's pose_to_rotations / pose_to_hips_height /
# pose_to_root_velocity, used only to rebuild the logging-only query.
from mmm.data.database import (  # noqa: E402
    pose_to_hips_height as feat_pose_to_hips_height,
    pose_to_root_velocity as feat_pose_to_root_velocity,
    pose_to_rotations as feat_pose_to_rotations,
)


if __name__ == "__main__":
    raise SystemExit(main())
