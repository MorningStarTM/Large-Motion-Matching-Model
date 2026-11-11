"""
Phase 2 round 1: drive the KNN teacher with a synthetic joystick and log
everything it does.

This is the "expert does the whole job alone, we just write down the
situation and the move" half of Phase 2. The teacher (`KnnController`) plays
entirely on its own, driven by `SyntheticJoystick`'s mix of holds, turns,
stops, sprints, reversals and sustained idle spans. Every frame is logged
through `mmm.teacher.logger.RolloutWriter`: the joystick input, the committed
pose, and -- on search frames -- the query and the top K candidates. Nothing
here is learned; this script only produces the dataset
`scripts/train_phase2.py` trains the SFT pass on.

Local:
    python scripts/collect_teacher_data.py --dataset data/bvh/bvh --frames 1000000 --out runs/rollouts/round1

Kaggle, continuing a session that hit its time limit:
    !python scripts/collect_teacher_data.py \\
        --dataset /kaggle/input/lafan1-resolved \\
        --frames 3000000 \\
        --out /kaggle/working/rollouts/round1

A second run with the same `--out` adds to what is there rather than starting
over: existing shard files are left alone and numbering continues after the
highest one found, which is what a Kaggle session that hit its time limit
needs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mmm.data.database import MotionDatabase
from mmm.data.prepare import LOCOMOTION_PATTERN, ensure_database
from mmm.envs.joystick import JoystickConfig, SyntheticJoystick
from mmm.teacher.controller import KnnController
from mmm.teacher.logger import RolloutWriter, database_fingerprint
from mmm.teacher.search import KnnIndex
from mmm.workspace import Workspace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", required=True)
    data.add_argument("--pattern", default=LOCOMOTION_PATTERN)
    data.add_argument("--fps", type=float, default=60.0)
    data.add_argument("--max-frames", type=int, default=0)
    data.add_argument("--full-skeleton", action="store_true")
    data.add_argument("--rebuild-database", action="store_true")
    data.add_argument(
        "--trim-calibration",
        action="store_true",
        help="see scripts/train_phase1.py; the same flag, the same reason.",
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--out", default="", help="where rollout shards are written")
    paths.add_argument("--work-dir", default="")
    paths.add_argument("--database-dir", default="")
    paths.add_argument("--cache-dir", default="")
    paths.add_argument("--run-name", default="phase2-collect")

    teacher = parser.add_argument_group("teacher")
    teacher.add_argument("--search-interval", type=int, default=10)
    teacher.add_argument("--k", type=int, default=4)
    teacher.add_argument("--transition-cost", type=float, default=0.5)
    teacher.add_argument(
        "--root-source", default="hybrid", choices=["hybrid", "simulation", "animation"]
    )

    episodes = parser.add_argument_group("episodes")
    episodes.add_argument(
        "--episode-frames",
        type=int,
        default=1800,
        help="reset the teacher to a fresh random starting row this often, "
        "in frames. 1800 is 30 seconds at 60 Hz. 0 means one continuous "
        "session with no resets; thousands of short episodes beats one very "
        "long one for the same reason Phase 1's clip-disjoint split matters "
        "-- it is coverage of many starting situations, not of one long "
        "trajectory, that the mix of synthetic primitives is meant to give.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--frames", type=int, required=True, help="total frames to collect")
    output.add_argument(
        "--shard-frames",
        type=int,
        default=36000,
        help="frames per rollout file. 36000 is 10 minutes at 60 Hz -- a "
        "shard this size is a few MB compressed, small enough that one "
        "failed write loses minutes, not the whole run.",
    )
    output.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    return args


def existing_shards(directory: str) -> int:
    """The next free shard index, so a second run extends rather than
    overwrites. Shard files are named `shard_00000.npz`, so the highest
    existing index plus one is where to continue."""
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


def random_start(database: MotionDatabase, rng: np.random.RandomState) -> int:
    """A row the controller can validly start playing from."""
    valid = np.flatnonzero(database.matching.valid)
    if len(valid) == 0:
        return 0
    return int(valid[rng.randint(len(valid))])


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
        rebuild=args.rebuild_database,
        trim_calibration=args.trim_calibration,
    )
    database = MotionDatabase.load(database_path)
    print(database)

    index = KnnIndex(database)
    controller = KnnController(
        database,
        index=index,
        search_interval=args.search_interval,
        k=args.k,
        transition_cost=args.transition_cost,
        root_source=args.root_source,
    )
    joystick = SyntheticJoystick(
        frame_time=database.frame_time, config=JoystickConfig(), seed=args.seed
    )
    rng = np.random.RandomState(args.seed)
    controller.reset(row=random_start(database, rng))

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
                "search_interval": args.search_interval,
                "k": args.k,
                "transition_cost": args.transition_cost,
                "root_source": args.root_source,
                "episode_frames": args.episode_frames,
                "source": "collect_teacher_data",
            },
        )

        for _ in range(shard_frames):
            if args.episode_frames and since_episode_reset >= args.episode_frames:
                controller.reset(row=random_start(database, rng))
                since_episode_reset = 0

            # The forward BEFORE this frame's turn is applied, which is the
            # one the teacher's own search actually used: KnnController._search
            # (hence _current_query, hence trajectory.predict) runs before
            # state.forward is reassigned for root_source="hybrid". Calling
            # predict() with the forward captured after step() returns would
            # express the same prediction in next frame's facing instead of
            # this one's, which is wrong in exactly the cases that matter --
            # anything that turns.
            forward_before_turn = controller.state.forward.copy()
            stick, facing = joystick.step()
            record = controller.step(stick, facing)
            trajectory_vector = np.concatenate(
                [p.reshape(-1) for p in controller.trajectory.predict(forward_before_turn)]
            ).astype(np.float32)
            writer.append(stick, record, joystick_vector=trajectory_vector)

            since_episode_reset += 1

        manifest = writer.close()
        frames_done += shard_frames
        shard_index += 1

        rate = frames_done / max(time.time() - started, 1e-9)
        eta = (args.frames - frames_done) / max(rate, 1e-9)
        print(
            f"shard {shard_index - 1:05d}: {manifest['frames']:6d} frames, "
            f"{manifest['n_searches']:5d} searches | "
            f"{frames_done:,}/{args.frames:,} total "
            f"({100 * frames_done / args.frames:5.1f}%) | "
            f"{rate:6.0f} fps | eta {eta / 60:5.1f} min"
        )

    mix = joystick.realised_mix()
    print("\nrealised primitive mix (fraction of frames):")
    for name, fraction in mix.items():
        print(f"  {name:12s} {100 * fraction:5.1f}%")

    with open(os.path.join(workspace.out, "collection_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "frames": frames_done,
                "shards": shard_index,
                "seed": args.seed,
                "database_fingerprint": database_fingerprint(database),
                "realised_mix": mix,
                "args": vars(args),
            },
            handle,
            indent=2,
        )

    print(f"\nwrote {shard_index} shard(s), {frames_done:,} frames, to {workspace.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
