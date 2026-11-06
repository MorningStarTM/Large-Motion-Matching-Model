"""
Recover the obstacle layout from mocap, report how trustworthy it is, and save
it.

LaFAN1's obstacle clips were captured with real boxes and platforms that the
BVH does not record. This gets them back out of the motion, then prints the
evidence for each recovered surface so the result can be judged rather than
taken on faith.

The three numbers that matter per surface:

  clips    how many different clips contributed contacts. The props did not
           move between capture sessions, so separate clips are independent
           measurements of the same object. One clip is a pose; eight clips is
           a prop.
  spread   standard deviation of the contributing heights, in centimetres. A
           rigid surface measured many times comes in under 2 cm. A large
           spread means the clustering has chained two different objects.
  n        how many contacts in total. The weakest of the three, since one
           subject standing in one place repeatedly inflates it.

Usage:

    # recover, report, and write the layout
    python scripts/recover_obstacles.py --bvh-dir data/bvh/bvh \\
        --out data/obstacles/lafan1_studio.json

    # check the detector against clips that should have no obstacles at all
    python scripts/recover_obstacles.py --bvh-dir data/bvh/bvh --control

Nothing in `mmm/data`, `mmm/teacher` or `mmm/models` is read or written by
this; it only produces a JSON layout that `mmm.obstacles.database.augment`
later consumes.
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
from mmm.data.skeleton import LOCOMOTION_JOINTS
from mmm.obstacles.database import save_fields
from mmm.obstacles.recover import (
    SupportThresholds,
    cluster_supports,
    detect_supports,
    recover_studio_field,
)

# Clips captured with props. Everything else is treated as flat ground, which
# for a walk or a dance clip is the truth.
OBSTACLE_PATTERN = "obstacles*.bvh"

# Clips that should recover nothing. The detector is only as good as its false
# positive rate, and level locomotion is where a bad detector shows up.
CONTROL_PATTERNS = ("walk*.bvh", "run*.bvh", "sprint*.bvh", "dance*.bvh", "aiming*.bvh")


def load_clips(paths, verbose: bool = True):
    """Load and reduce to the locomotion joint subset, naming each clip."""
    motions = []
    for path in paths:
        with warnings.catch_warnings():
            # Some LaFAN1 clips warn about joint position channels drifting
            # from their rest offsets. Forward kinematics uses the offsets and
            # the drift is about 1% of bone length; it is reported once by the
            # loader and is not actionable here.
            warnings.simplefilter("ignore")
            motion = bvh.load(path)
        motion = motion.subset(LOCOMOTION_JOINTS)
        motion.name = os.path.splitext(os.path.basename(path))[0]
        motions.append(motion)
        if verbose:
            print(f"  loaded {motion.name:26s} {motion.n_frames:7,d} frames")
    return motions


def report_layout(field, clusters) -> None:
    """Print each recovered surface with the evidence behind it."""
    print(
        f"\n{'top cm':>7s} {'dx':>5s} {'dz':>5s} "
        f"{'x0':>8s} {'x1':>8s} {'z0':>8s} {'z1':>8s} "
        f"{'n':>4s} {'clips':>5s} {'spread':>6s}"
    )
    print("-" * 78)
    for box, cluster in zip(field, clusters):
        print(
            f"{box.top:7.1f} {box.size[0]:5.0f} {box.size[2]:5.0f} "
            f"{box.x0:8.1f} {box.x1:8.1f} {box.z0:8.1f} {box.z1:8.1f} "
            f"{len(cluster.supports):4d} {len(cluster.clips):5d} "
            f"{cluster.height_spread:6.2f}"
        )

    if not clusters:
        print("  (nothing recovered)")
        return

    spreads = np.array([c.height_spread for c in clusters])
    votes = np.array([len(c.clips) for c in clusters])
    print(
        f"\n{len(clusters)} surfaces. "
        f"Height spread: median {np.median(spreads):.2f} cm, "
        f"worst {spreads.max():.2f} cm. "
        f"Clips per surface: median {int(np.median(votes))}, "
        f"max {int(votes.max())}."
    )


def run_control(bvh_dir: str, thresholds: SupportThresholds) -> int:
    """Recover from clips that have no obstacles, and report what leaks through.

    A detector that finds obstacles in a walk clip is worthless no matter how
    good it looks on the obstacle clips, so this is the test that decides
    whether the thresholds are honest.
    """
    paths = []
    for pattern in CONTROL_PATTERNS:
        paths.extend(sorted(glob.glob(os.path.join(bvh_dir, pattern))))
    if not paths:
        print(f"no control clips found in {bvh_dir}")
        return 1

    print(f"control: {len(paths)} clips that should contain no obstacles\n")
    total_supports, total_surfaces, offenders = 0, 0, []
    for motion in load_clips(paths, verbose=False):
        supports, _ = detect_supports(motion, thresholds)
        surfaces = cluster_supports(supports, min_supports=3)
        total_supports += len(supports)
        total_surfaces += len(surfaces)
        if surfaces:
            heights = ", ".join(f"{c.height:.1f}" for c in surfaces)
            offenders.append(f"  {motion.name:26s} {len(surfaces)} at {heights} cm")

    for line in offenders:
        print(line)
    print(
        f"\nfalse positives: {total_supports} supports and {total_surfaces} "
        f"surfaces across {len(paths)} clips "
        f"({total_supports / len(paths):.1f} supports per clip)."
    )
    print(
        "For comparison, an obstacle clip yields roughly 100 supports. A "
        "control rate near zero is what makes the recovery believable."
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover obstacle geometry from mocap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bvh-dir", required=True, help="directory holding the BVH clips"
    )
    parser.add_argument(
        "--pattern",
        default=OBSTACLE_PATTERN,
        help="which clips were captured with props",
    )
    parser.add_argument("--out", default=None, help="where to write the layout JSON")
    parser.add_argument(
        "--control",
        action="store_true",
        help="run the false positive check on level locomotion instead",
    )
    parser.add_argument(
        "--per-clip",
        action="store_true",
        help="also recover each clip on its own and report agreement with the "
        "pooled layout",
    )

    group = parser.add_argument_group("detection")
    group.add_argument("--speed", type=float, default=20.0)
    group.add_argument("--vertical-speed", type=float, default=12.0)
    group.add_argument("--min-frames", type=int, default=10)
    group.add_argument("--min-height", type=float, default=15.0)
    group.add_argument("--max-height", type=float, default=190.0)

    group = parser.add_argument_group("clustering")
    group.add_argument("--radius", type=float, default=45.0)
    group.add_argument("--height-tolerance", type=float, default=12.0)
    group.add_argument("--min-supports", type=int, default=6)
    group.add_argument("--min-clips", type=int, default=2)
    group.add_argument("--margin", type=float, default=12.0)

    args = parser.parse_args(argv)

    thresholds = SupportThresholds(
        speed=args.speed,
        vertical_speed=args.vertical_speed,
        min_frames=args.min_frames,
        min_height=args.min_height,
        max_height=args.max_height,
    )

    if args.control:
        return run_control(args.bvh_dir, thresholds)

    paths = sorted(glob.glob(os.path.join(args.bvh_dir, args.pattern)))
    if not paths:
        print(f"no clips matching {args.pattern!r} in {args.bvh_dir}")
        return 1

    print(f"recovering from {len(paths)} clips matching {args.pattern!r}")
    motions = load_clips(paths)

    field, clusters = recover_studio_field(
        motions,
        thresholds=thresholds,
        radius=args.radius,
        height_tolerance=args.height_tolerance,
        min_supports=args.min_supports,
        min_clips=args.min_clips,
        margin=args.margin,
    )
    report_layout(field, clusters)

    if args.per_clip:
        print("\nper clip recovery against the pooled layout:")
        errors = []
        for motion in motions:
            supports, _ = detect_supports(motion, thresholds)
            solo = cluster_supports(
                supports, args.radius, args.height_tolerance, min_supports=3
            )
            for cluster in solo:
                x0, x1, z0, z1 = cluster.bounds()
                centre = np.array([(x0 + x1) / 2, (z0 + z1) / 2])
                # Nearest pooled surface *of a comparable height*: matching on
                # footprint alone picks the wrong tread of a staircase, whose
                # steps sit almost on top of each other in plan view.
                best = None
                for box in field:
                    if abs(cluster.height - box.top) > args.height_tolerance:
                        continue
                    distance = float(np.linalg.norm(centre - box.centre))
                    if best is None or distance < best[0]:
                        best = (distance, box)
                if best is not None and best[0] < 150.0:
                    errors.append(abs(cluster.height - best[1].top))
            print(f"  {motion.name:26s} {len(solo):3d} surfaces")
        if errors:
            errors = np.array(errors)
            print(
                f"\nheight agreement over {len(errors)} matched surfaces: "
                f"median {np.median(errors):.2f} cm, "
                f"p90 {np.percentile(errors, 90):.2f} cm, "
                f"max {errors.max():.2f} cm"
            )

    if args.out:
        # Every obstacle clip shares the one studio layout, which is the whole
        # reason pooling is valid. Clips outside the pattern get no entry and
        # are read as flat ground by `augment`.
        fields = {motion.name: field for motion in motions}
        save_fields(args.out, fields)
        print(f"\nwrote {len(fields)} clip entries to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
