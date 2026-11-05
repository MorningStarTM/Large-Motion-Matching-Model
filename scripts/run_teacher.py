"""
Run the Phase 0 KNN motion matching controller.

The first milestone of the plan's getting started roadmap, and the thing to get
visibly right before any training starts.

Two modes:

- interactive, in the raylib viewer, driven by a gamepad;
- headless, driven by the synthetic joystick generator, which is how the hours
  of Phase 2 data are collected.

Either way every query, the top K candidates and the chosen frame are logged
from the first run (plan section 5.1), because Phase 2 is built out of these
logs and logging added later means collecting the data twice.

Usage:
    python scripts/run_teacher.py --database data/databases/locomotion
    python scripts/run_teacher.py --headless --frames 36000 --log runs/teacher/rollout_000.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mmm.data.database import MotionDatabase
from mmm.envs.joystick import JoystickConfig, SyntheticJoystick
from mmm.teacher.controller import KnnController
from mmm.teacher.logger import RolloutWriter
from mmm.teacher.search import KnnIndex
from mmm.teacher.trajectory import TrajectoryPredictor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", default=os.path.join("data", "databases", "locomotion"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=3600, help="headless only")
    parser.add_argument("--log", default="", help="rollout .npz to write")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-interval", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--transition-cost", type=float, default=0.5)
    parser.add_argument(
        "--max-speed",
        type=float,
        default=400.0,
        help="units/second at full stick. Data dependent: LaFAN1 locomotion "
        "tops out near 440 cm/s, and asking for more than the database "
        "contains makes every retrieval a poor match.",
    )
    parser.add_argument(
        "--facing-halflife",
        type=float,
        default=0.2,
        help="seconds for the body to cover half the remaining gap to a turn "
        "request. Lag is linear in it: 0.1 lags 8 degrees against a stick "
        "sweeping at 60 deg/s, 0.2 lags 16, 0.4 lags 33. Foot sliding is flat "
        "across that range, so this is a pure feel knob.",
    )
    parser.add_argument(
        "--root-source",
        default="hybrid",
        choices=["hybrid", "simulation", "animation"],
        help="what drives the character's position and facing. 'hybrid' takes "
        "facing from the spring damper and translation from the animation: "
        "steerable and no added foot sliding. 'simulation' takes both from the "
        "spring damper, which slides the feet. 'animation' integrates the "
        "retrieved clip's own root velocity, which is faithful but lets the "
        "heading random walk.",
    )
    parser.add_argument("--backend", default="kdtree", choices=["kdtree", "brute"])
    parser.add_argument("--start-row", type=int, default=0)
    return parser.parse_args(argv)


def build_controller(args, database: MotionDatabase) -> KnnController:
    index = KnnIndex(database, backend=args.backend)
    trajectory = TrajectoryPredictor(
        frame_time=database.frame_time,
        max_speed=args.max_speed,
        facing_halflife=args.facing_halflife,
    )
    controller = KnnController(
        database,
        index=index,
        search_interval=args.search_interval,
        k=args.k,
        transition_cost=args.transition_cost,
        trajectory=trajectory,
        root_source=args.root_source,
    )
    controller.reset(row=args.start_row)
    return controller


def run_headless(args, database: MotionDatabase) -> int:
    controller = build_controller(args, database)
    joystick = SyntheticJoystick(
        frame_time=database.frame_time, config=JoystickConfig(), seed=args.seed
    )

    writer = None
    if args.log:
        writer = RolloutWriter(
            path=args.log,
            database=database,
            config={
                "search_interval": args.search_interval,
                "k": args.k,
                "transition_cost": args.transition_cost,
                "max_speed": args.max_speed,
            },
            seed=args.seed,
        )

    started = time.time()
    searches = jumps = 0
    origins = []
    for frame in range(args.frames):
        stick, facing = joystick.step()
        record = controller.step(stick, facing)
        searches += record.searched
        jumps += record.jumped
        origins.append(record.origin.copy())
        if writer is not None:
            writer.append(stick, record)

        if (frame + 1) % 3600 == 0:
            print(f"  {frame + 1}/{args.frames} frames, {time.time() - started:.0f}s")

    elapsed = time.time() - started
    origins = np.array(origins)
    speeds = np.linalg.norm(np.diff(origins[:, [0, 2]], axis=0), axis=-1) / database.frame_time

    print(f"\n{args.frames} frames in {elapsed:.1f}s ({elapsed / args.frames * 1e6:.0f} us/frame, Python)")
    print(f"searches {searches}, jumps {jumps} ({jumps / max(searches, 1):.0%} of searches)")
    print(f"speed: mean {speeds.mean():.0f}, p95 {np.percentile(speeds, 95):.0f}, max {speeds.max():.0f} units/s")
    print("joystick mix:", {k: f"{v:.0%}" for k, v in joystick.realised_mix().items()})

    if writer is not None:
        manifest = writer.close()
        print(f"\nwrote {args.log}: {manifest['frames']} frames, {manifest['n_searches']} searches")
    return 0


def run_interactive(args, database: MotionDatabase) -> int:
    try:
        from mmm.viewer.app import ViewerApp
    except ImportError as error:
        print(
            f"cannot start the viewer: {error}\n"
            "Install raylib (pip install -r requirements.txt), or run with "
            "--headless to drive the teacher with the synthetic joystick.",
            file=sys.stderr,
        )
        return 1

    controller = build_controller(args, database)
    app = ViewerApp(database=database, controller=controller, log_path=args.log or None, seed=args.seed)
    app.run()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.exists(os.path.join(args.database, "manifest.json")):
        print(
            f"no database at {args.database}. Run scripts/build_databases.py first.",
            file=sys.stderr,
        )
        return 1

    database = MotionDatabase.load(args.database)
    print(database)

    if args.headless:
        return run_headless(args, database)
    return run_interactive(args, database)


if __name__ == "__main__":
    raise SystemExit(main())
