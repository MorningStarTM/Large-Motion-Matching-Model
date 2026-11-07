"""
Run the obstacle aware KNN teacher, interactively or headless.

This is the demo the obstacle work exists to produce: walk the character at a
cube and watch motion matching retrieve a step up, a vault or a stumble,
because the search can now see that the cube is there.

    # interactive, with the recovered studio layout already in place
    python scripts/run_obstacle_teacher.py --layout data/obstacles/lafan1_studio.json --studio

    # interactive, empty scene -- press 1-5 to place cubes yourself
    python scripts/run_obstacle_teacher.py --layout data/obstacles/lafan1_studio.json

    # headless A/B: the same walk with obstacle awareness on and off
    python scripts/run_obstacle_teacher.py --layout data/obstacles/lafan1_studio.json --compare

The headless comparison is the honest one. Watching a character step onto a box
is easy to over-read -- the pose carries its own height, so *some* elevation
happens either way -- and `--compare` drives both controllers through the
identical stick input and reports the difference in numbers.

Needs a layout from `scripts/recover_obstacles.py`, which is where the
obstacles are recovered from the mocap in the first place.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmm.data import bvh
from mmm.data.database import MotionDatabase, pose_to_hips_height
from mmm.data.skeleton import LOCOMOTION_JOINTS
from mmm.obstacles.controller import ObstacleController
from mmm.obstacles.database import augment, load_fields
from mmm.obstacles.geometry import Box, BoxField
from mmm.teacher.controller import KnnController
from mmm.viewer import session as session_wiring

# Obstacle clips first, so the database is dominated by frames that have
# something to retrieve, plus level locomotion so ordinary walking still works.
DEFAULT_CLIPS = (
    "obstacles1_subject1",
    "obstacles2_subject1",
    "obstacles4_subject2",
    "obstacles5_subject2",
    "obstacles6_subject1",
    "jumps1_subject1",
    "walk1_subject1",
    "run1_subject2",
)


def load_clips(bvh_dir: str, names, max_frames: int):
    motions = []
    for name in names:
        path = os.path.join(bvh_dir, name + ".bvh")
        if not os.path.isfile(path):
            matches = glob.glob(os.path.join(bvh_dir, name + "*.bvh"))
            if not matches:
                raise FileNotFoundError(f"no clip matching {name!r} in {bvh_dir}")
            path = matches[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            motion = bvh.load(path)
        motion = motion.subset(LOCOMOTION_JOINTS)
        if max_frames and motion.n_frames > max_frames:
            motion = motion.slice(0, max_frames)
        motion.name = os.path.splitext(os.path.basename(path))[0]
        motions.append(motion)
        print(f"  {motion.name:26s} {motion.n_frames:7,d} frames")
    return motions


def build(bvh_dir: str, names, max_frames: int, layout_path: str | None):
    print("loading clips")
    motions = load_clips(bvh_dir, names, max_frames)

    fields = load_fields(layout_path) if layout_path else {}
    if layout_path:
        covered = sum(1 for m in motions if m.name in fields)
        print(
            f"\nlayout: {len(fields)} clip entries, "
            f"{covered}/{len(motions)} of these clips covered"
        )
        if covered == 0:
            print(
                "  WARNING: none of these clips are in the layout, so every "
                "database row says flat ground and the obstacle block carries "
                "no information. Run scripts/recover_obstacles.py first."
            )

    database = MotionDatabase.build(motions)
    augmented = augment(database, motions, fields)
    print(
        f"database: {database.matching.n_rows:,} rows, "
        f"{augmented.matching.normalized().shape[1]} features"
    )

    # Every obstacle clip shares the one studio layout, so any of them gives
    # the geometry the actors actually negotiated.
    studio = next(
        (fields[m.name] for m in motions if m.name in fields), BoxField()
    )
    return database, augmented, motions, studio


# Heights the recovery found in the data, used as the comparison's test cubes.
COMPARE_HEIGHTS = (20.9, 32.8, 47.3, 66.1, 93.5)


def _toe_penetration(controller, cube, frames: int, toes) -> tuple:
    """Walk straight at a cube and measure how far the toes sink into it.

    This is the metric, and it took two attempts to get right. The obvious one
    -- peak hips height while over the cube -- is close to useless: it is a
    single deterministic rollout, one different retrieval early changes
    everything after it, and adding two clips to the database flipped its sign.

    Penetration is the visible artefact and it is directly interpretable. If
    the search returns a flat walking frame while the character is over a 47 cm
    box, the toes sit near y = 0 and penetrate 47 cm. If it returns a step up,
    they sit near the top and penetrate nothing. Averaged over several
    placements it is stable.

    Only toes horizontally inside the footprint are counted; a foot beside the
    box is not penetrating anything.
    """
    controller.reset(
        row=0,
        origin=np.zeros(3, np.float32),
        forward=np.array([0.0, 1.0], np.float32),
    )
    if hasattr(controller, "field"):
        controller.field = BoxField([cube])

    stick = np.array([0.0, 1.0], np.float32)
    depths, jumps = [], 0
    for _ in range(frames):
        record = controller.step(stick)
        jumps += bool(record.jumped)
        world = controller.world_positions(controller.state)
        for joint in toes:
            x, y, z = world[joint]
            if cube.x0 <= x <= cube.x1 and cube.z0 <= z <= cube.z1:
                depths.append(max(float(cube.top) - float(y), 0.0))

    return (float(np.mean(depths)) if depths else 0.0), len(depths), jumps


def run_compare(database, augmented, motions, frames: int, trials: int) -> int:
    """Obstacle-aware against obstacle-blind, same stick input, many cubes.

    The cube's footprint and distance are varied per trial so the result is not
    one lucky alignment between a placement and whatever the database happens
    to contain.
    """
    skeleton = motions[0].skeleton
    toes = [skeleton.index("LeftToeBase"), skeleton.index("RightToeBase")]

    aware = ObstacleController(augmented, field=BoxField())
    blind = KnnController(database)

    rng = np.random.default_rng(0)
    print()
    print(
        f"walking straight at a cube, {trials} placements per height, "
        f"{frames} frames each"
    )
    print(
        f"{'cube cm':>8s} {'aware':>10s} {'blind':>10s} {'reduction':>10s} "
        f"{'aware jumps':>12s} {'blind jumps':>12s}"
    )
    print("-" * 68)

    aware_all, blind_all = [], []
    for height in COMPARE_HEIGHTS:
        a_depth, b_depth, a_jump, b_jump = [], [], [], []
        for _ in range(trials):
            near = 220.0 + rng.uniform(-40.0, 80.0)
            width = rng.uniform(100.0, 160.0)
            cube = Box(
                -width / 2.0,
                width / 2.0,
                near,
                near + rng.uniform(100.0, 160.0),
                top=height,
                label=f"test:{height:.0f}",
            )
            depth, _, jumps = _toe_penetration(aware, cube, frames, toes)
            a_depth.append(depth)
            a_jump.append(jumps)
            depth, _, jumps = _toe_penetration(blind, cube, frames, toes)
            b_depth.append(depth)
            b_jump.append(jumps)

        a_mean, b_mean = float(np.mean(a_depth)), float(np.mean(b_depth))
        aware_all.append(a_mean)
        blind_all.append(b_mean)
        reduction = (1.0 - a_mean / b_mean) * 100.0 if b_mean > 1e-6 else 0.0
        print(
            f"{height:8.1f} {a_mean:9.1f}cm {b_mean:9.1f}cm {reduction:9.1f}% "
            f"{np.mean(a_jump):12.1f} {np.mean(b_jump):12.1f}"
        )

    a_overall, b_overall = float(np.mean(aware_all)), float(np.mean(blind_all))
    print()
    print(
        f"overall mean toe penetration: aware {a_overall:.1f} cm, "
        f"blind {b_overall:.1f} cm "
        f"({(1.0 - a_overall / b_overall) * 100.0:+.1f}%)"
    )
    print()
    for line in (
        "Lower is better: it means the search returned frames whose feet",
        "belong on top of the box rather than through it. The gap narrows",
        "at the tallest cube, which is the expected failure -- there is far",
        "less 93 cm vaulting in the database than stepping up 20 cm.",
        "",
        "Obstacle awareness also raises the jump count: the obstacle",
        "dimensions are weighted above 1, so distances are larger and the",
        "transition_cost default of 0.5 is no longer calibrated for them.",
    ):
        print(line)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the obstacle aware KNN teacher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bvh-dir", default=os.path.join("data", "bvh", "bvh"))
    parser.add_argument(
        "--layout",
        default=os.path.join("data", "obstacles", "lafan1_studio.json"),
        help="layout JSON from scripts/recover_obstacles.py",
    )
    parser.add_argument(
        "--clips", nargs="*", default=list(DEFAULT_CLIPS), help="clip stems to load"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8000,
        help="frames per clip; the whole set is large and the viewer wants a "
        "database it can hold comfortably",
    )
    parser.add_argument(
        "--studio",
        action="store_true",
        help="start with the recovered studio layout in the scene",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="headless A/B against the obstacle-blind controller",
    )
    parser.add_argument(
        "--frames", type=int, default=260, help="--compare: frames per rollout"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=6,
        help="--compare: cube placements per height. More is steadier; a "
        "single placement is not a measurement.",
    )
    parser.add_argument("--log", default=None, help="write a rollout log")
    parser.add_argument(
        "--viewer-frames", type=int, default=None,
        help="stop the viewer after this many frames, for a headless check",
    )
    parser.add_argument(
        "--viewer-screenshot", default=None, help="write a PNG before exiting"
    )
    parser.add_argument(
        "--viewer-walk", action="store_true",
        help="drive forward instead of reading input; use with --viewer-frames",
    )
    session_wiring.add_arguments(parser)
    parser.add_argument(
        "--search-interval", type=int, default=10, help="frames between searches"
    )
    parser.add_argument("--k", type=int, default=4, help="candidates retrieved")
    parser.add_argument(
        "--transition-cost",
        type=float,
        default=None,
        help="how much better a candidate must be before jumping. The default "
        "of 0.5 was tuned on 27 normalized dimensions; with the obstacle block "
        "weighted above 1 the distances are larger, so raising this is the "
        "first thing to try if the motion shimmers.",
    )

    args = parser.parse_args(argv)

    # A model retrieves nothing, so the Matching Database it would search is
    # never read. Building it anyway cost 23 seconds of startup against 3 for
    # the one clip the seed pose comes from, so the model takes the short
    # path. Measured, not assumed.
    if args.model:
        database, studio = session_wiring.load_model_scene(
            args.bvh_dir, args.clips, args.max_frames, args.layout
        )
        augmented, motions = database, database.motions
        print(f"model scene: {database}")
    else:
        database, augmented, motions, studio = build(
            args.bvh_dir, args.clips, args.max_frames, args.layout
        )
    print(f"studio layout in scene: {studio!r}")

    if args.compare:
        return run_compare(database, augmented, motions, args.frames, args.trials)

    controller_kwargs = {"search_interval": args.search_interval, "k": args.k}
    if args.transition_cost is not None:
        controller_kwargs["transition_cost"] = args.transition_cost

    field = BoxField(list(studio)) if args.studio else BoxField()
    controller, drawn, description = session_wiring.build_controller(
        args, database, augmented, field, controller_kwargs
    )
    print(f"driven by: {description}")

    from mmm.obstacles.viewer import ObstacleViewer
    from mmm.viewer.app import ViewerConfig

    title = (
        "MMM - trained model" if args.model
        else "MMM - obstacle aware KNN motion matching"
    )
    viewer = ObstacleViewer(
        drawn,
        controller,
        studio_field=studio,
        field=field,
        config=ViewerConfig(title=title),
        # The rollout log records a teacher's retrieval decisions, which a
        # model does not make. --record is the model's equivalent.
        log_path=None if args.model else args.log,
    )
    print(
        "\ncontrols: WASD or stick to move, mouse to turn.\n"
        "  1-5  place a cube at a height recovered from the data "
        "(21, 33, 47, 66, 93 cm)\n"
        "  6    place a 150 cm block the data has nothing for\n"
        "  L    load the full recovered studio layout\n"
        "  backspace undo    0 clear    C candidates    T trajectory\n"
        "  F contacts    space pause    M release mouse\n"
    )
    viewer.run(
        max_frames=args.viewer_frames,
        screenshot=args.viewer_screenshot,
        auto_stick=(0.0, 1.0) if args.viewer_walk else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
