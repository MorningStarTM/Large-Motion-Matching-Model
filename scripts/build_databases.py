"""
Build the Matching and Animation Databases from BVH clips.

Loads LaFAN1 resolved, reduces each clip to the 24 joint locomotion skeleton,
computes features and poses, fits the normalization, and writes both databases
plus a manifest.

Usage:
    python scripts/build_databases.py --out data/databases/locomotion
    python scripts/build_databases.py --pattern "walk*,run*,sprint*" --max-frames 20000
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmm.data import bvh
from mmm.data.database import MotionDatabase
from mmm.data.skeleton import LOCOMOTION_JOINTS

# The locomotion takes in LaFAN1 resolved. Everything else in the dataset is
# aiming, dancing, fighting, falling and obstacle work, which is not what a
# locomotion controller should retrieve from.
LOCOMOTION_PATTERN = "walk*,run*,sprint*"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bvh-dir", default=os.path.join("data", "bvh", "bvh"))
    parser.add_argument("--out", default=os.path.join("data", "databases", "locomotion"))
    parser.add_argument("--cache-dir", default=os.path.join("data", "processed", "_cache"))
    parser.add_argument(
        "--pattern",
        default=LOCOMOTION_PATTERN,
        help="comma separated filename globs, without the .bvh suffix",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="truncate each clip to this many frames; 0 means keep all",
    )
    parser.add_argument(
        "--fps", type=float, default=60.0, help="resample every clip to this rate"
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--full-skeleton",
        action="store_true",
        help="keep all 75 joints instead of the 24 locomotion joints. Triples "
        "the pose width for fingers a locomotion controller cannot use.",
    )
    return parser.parse_args(argv)


def select_clips(bvh_dir: str, pattern: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(bvh_dir, "*.bvh")))
    if not paths:
        raise FileNotFoundError(
            f"no .bvh files in {bvh_dir}. Download LaFAN1 resolved into that directory."
        )

    globs = [p.strip() for p in pattern.split(",") if p.strip()]
    selected = [
        path
        for path in paths
        if any(fnmatch.fnmatch(os.path.splitext(os.path.basename(path))[0], g) for g in globs)
    ]
    if not selected:
        available = sorted({os.path.basename(p).split("_")[0] for p in paths})
        raise ValueError(f"pattern {pattern!r} matched nothing. Takes available: {available}")
    return selected


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = select_clips(args.bvh_dir, args.pattern)
    print(f"{len(paths)} clips matching {args.pattern!r}")

    motions = []
    started = time.time()
    for path in paths:
        clip_started = time.time()
        motion = bvh.load_cached(path, args.cache_dir, rebuild=args.rebuild_cache)
        if not args.full_skeleton:
            motion = motion.subset(LOCOMOTION_JOINTS)
        if abs(motion.fps - args.fps) > 1e-6:
            motion = motion.resample(args.fps)
        if args.max_frames and motion.n_frames > args.max_frames:
            name = motion.name
            motion = motion.slice(0, args.max_frames)
            motion.name = name
        motions.append(motion)
        print(
            f"  {motion.name:24} {motion.n_frames:7d} frames  "
            f"drift {motion.position_drift:.2%}  ({time.time() - clip_started:.1f}s)"
        )

    print(f"\nloaded {len(motions)} clips in {time.time() - started:.1f}s")

    started = time.time()
    database = MotionDatabase.build(motions, source="lafan1_resolved")
    print(f"{database}  built in {time.time() - started:.1f}s")
    print(f"valid rows: {int(database.matching.valid.sum())} / {database.n_rows}")
    for name, size in database.memory_bytes().items():
        print(f"  {name:20} {size / 1e6:8.1f} MB")

    database.save(args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
