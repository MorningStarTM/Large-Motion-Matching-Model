"""
Turn whatever `--dataset` points at into a usable database.

A training script should accept the thing a person actually has, which depends
on where they are running:

- a **built database directory** (holds `manifest.json`) -- reuse it
- a **directory of BVH files** -- build the database, then use it
- a Kaggle dataset mount, where the BVH files are usually some levels down
  inside a directory named after the dataset rather than sitting at the top

So the search is recursive and the build is cached. Without that, every Kaggle
session spends twenty seconds parsing BVH and seven minutes on retrieval before
the first gradient step, and a session that restarts pays it again.

The build output goes wherever the caller says, which on Kaggle must be under
`/kaggle/working` rather than beside the read only input. See `mmm.workspace`.
"""

from __future__ import annotations

import fnmatch
import json
import os
import time

from mmm.data.bvh import load_cached
from mmm.data.calibration import trim_calibration as trim_calibration_pose
from mmm.data.database import MotionDatabase
from mmm.data.skeleton import LOCOMOTION_JOINTS

__all__ = [
    "LOCOMOTION_PATTERN",
    "find_bvh",
    "looks_like_database",
    "ensure_database",
]

# Locomotion only. The full LaFAN1 set includes fighting, dancing and falls,
# which share no trajectory structure with walking and would be retrieved as
# neighbours for a walk query.
LOCOMOTION_PATTERN = "walk*,run*,sprint*"


def looks_like_database(path: str) -> bool:
    """Whether `path` is an already built database directory."""
    return os.path.isfile(os.path.join(path, "manifest.json"))


def find_bvh(root: str, pattern: str = LOCOMOTION_PATTERN) -> list[str]:
    """BVH files under `root` whose basename matches any glob in `pattern`.

    Recursive, because a Kaggle dataset mounts as
    `/kaggle/input/<name>/<whatever the uploader zipped>`, and the files are
    rarely at the top. Sorted, so a database built twice has its clips in the
    same order and the row indices mean the same thing.
    """
    if os.path.isfile(root):
        return [root] if root.lower().endswith(".bvh") else []

    globs = [p.strip() for p in pattern.split(",") if p.strip()] or ["*"]
    found = []
    for folder, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".bvh"):
                continue
            stem = os.path.splitext(name)[0]
            if any(fnmatch.fnmatch(stem, g) or fnmatch.fnmatch(name, g) for g in globs):
                found.append(os.path.join(folder, name))
    return sorted(found)


def _build_signature(paths, pattern, fps, max_frames, full_skeleton,
                     trim_calibration=False) -> dict:
    """What the built database depends on.

    Stored beside it so a cached build is only reused when it was made from the
    same inputs with the same settings. Reusing one made from a different clip
    selection is the kind of error that produces a trained model nobody can
    reproduce.
    """
    return {
        "clips": [os.path.basename(p) for p in paths],
        "pattern": pattern,
        "fps": fps,
        "max_frames": max_frames,
        "trim_calibration": bool(trim_calibration),
        "full_skeleton": full_skeleton,
    }


def ensure_database(
    source: str,
    out_dir: str,
    cache_dir: str | None = None,
    pattern: str = LOCOMOTION_PATTERN,
    fps: float = 60.0,
    max_frames: int = 0,
    full_skeleton: bool = False,
    rebuild: bool = False,
    verbose: bool = True,
    trim_calibration: bool = False,
) -> str:
    """Return a path to a built database, building it if necessary.

    Args:
        source: a built database directory, a directory to search for BVH, or a
            single BVH file.
        out_dir: where to build to. Must be writable; on Kaggle that rules out
            anywhere under `/kaggle/input`.
        cache_dir: where parsed BVH is cached. Defaults to `out_dir`.
        pattern: comma separated globs on the clip basename.
        fps: resample to this rate if the clips differ.
        max_frames: truncate each clip, for a quick run. 0 keeps everything.
        full_skeleton: keep all joints instead of the 24 locomotion ones.
        rebuild: build even if a matching database is already there.
        trim_calibration: drop the leading calibration T-pose from each clip.
            Every LaFAN1 take opens with the actor standing still with their
            arms out, 104 to 475 frames of it depending on the clip. Those
            frames are searchable rows, so a query that lands near one makes
            the character snap into a T-pose mid-game.

    Returns:
        The database directory.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"dataset path does not exist: {source}")

    # Already a database: nothing to do, and nothing to write, which matters
    # when it sits on a read only mount.
    if looks_like_database(source):
        if verbose:
            print(f"using existing database at {source}")
        return source

    paths = find_bvh(source, pattern)
    if not paths:
        listing = []
        for folder, _, files in os.walk(source):
            listing.extend(os.path.join(folder, f) for f in files[:5])
            if len(listing) > 10:
                break
        raise FileNotFoundError(
            f"no BVH files matching {pattern!r} under {source}. "
            f"Found instead: {listing[:10] or 'nothing'}. "
            "Pass --pattern '*' to take every clip."
        )

    signature = _build_signature(
        paths, pattern, fps, max_frames, full_skeleton, trim_calibration
    )
    signature_path = os.path.join(out_dir, "build_signature.json")

    if not rebuild and looks_like_database(out_dir) and os.path.isfile(signature_path):
        with open(signature_path, encoding="utf-8") as handle:
            if json.load(handle) == signature:
                if verbose:
                    print(
                        f"reusing database at {out_dir} "
                        f"({len(paths)} clips, signature matches)"
                    )
                return out_dir
            if verbose:
                print(f"database at {out_dir} was built from different inputs; rebuilding")

    os.makedirs(out_dir, exist_ok=True)
    cache_dir = cache_dir or out_dir
    os.makedirs(cache_dir, exist_ok=True)

    if verbose:
        print(f"building database from {len(paths)} clips matching {pattern!r}")

    motions = []
    trimmed_total = 0
    started = time.time()
    for path in paths:
        motion = load_cached(path, cache_dir, rebuild=rebuild)
        if not full_skeleton:
            motion = motion.subset(LOCOMOTION_JOINTS)
        if abs(motion.fps - fps) > 1e-6:
            motion = motion.resample(fps)
        if trim_calibration:
            motion, dropped = trim_calibration_pose(motion)
            trimmed_total += dropped
        if max_frames and motion.n_frames > max_frames:
            name = motion.name
            motion = motion.slice(0, max_frames)
            motion.name = name
        motions.append(motion)
        if verbose:
            print(f"  {motion.name:24} {motion.n_frames:7d} frames")

    if trim_calibration and verbose:
        print(
            f"  dropped {trimmed_total:,} leading calibration frames "
            f"({trimmed_total / fps:.0f}s) across {len(motions)} clips"
        )

    database = MotionDatabase.build(motions, source="lafan1_resolved")
    database.save(out_dir)

    with open(signature_path, "w", encoding="utf-8") as handle:
        json.dump(signature, handle, indent=2)

    if verbose:
        print(f"{database}  in {time.time() - started:.1f}s -> {out_dir}")
    return out_dir
