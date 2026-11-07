"""
Shared wiring for the two viewers: which controller drives, and whether the
session is recorded.

Both `scripts/run_obstacle_teacher.py` and `scripts/run_geno_viewer.py` need
exactly the same three decisions -- teacher or model, record or not, where to
start -- and duplicating them is how the two viewers quietly stop agreeing.
The argument names, the defaults and the seed-frame rule all live here once.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

__all__ = [
    "ModelScene",
    "load_model_scene",
    "add_arguments",
    "build_controller",
    "seed_pose_from_database",
    "DEFAULT_SEED_FRAME",
]

# Which frame of a clip to start a model from. Not 0: LaFAN1 clips open with
# roughly 150 frames of calibration T-pose, and seeding a model with one gives
# it an input unlike anything in its training distribution and a first second
# of nonsense. 1500 is about 25 seconds in, comfortably inside real motion.
DEFAULT_SEED_FRAME = 1500


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The model and recording flags, identical in both viewers."""
    group = parser.add_argument_group("trained model")
    group.add_argument(
        "--model",
        default=None,
        help="a published .safetensors checkpoint. The trained policy then "
        "drives the character instead of the KNN teacher, and nothing is "
        "retrieved or played back.",
    )
    group.add_argument(
        "--model-manifest",
        default=None,
        help="manifest.json describing that checkpoint. Defaults to the one "
        "beside it. Needed because a checkpoint alone does not record how "
        "wide the model is.",
    )
    group.add_argument(
        "--root-source",
        default="hybrid",
        choices=("hybrid", "simulation", "animation"),
        help="who owns the character's position and facing. 'hybrid' takes "
        "facing from the spring damper and translation from the motion; "
        "'animation' lets the model own both, which is the honest test of "
        "whether it learned to steer.",
    )
    group.add_argument(
        "--seed-frame",
        type=int,
        default=DEFAULT_SEED_FRAME,
        help="which frame of the first clip to seed the model from",
    )
    group.add_argument(
        "--device", default="cpu", help="torch device for the policy"
    )

class _EchoRow:
    """Returns the row index it is given. See `ModelScene.matching`."""

    def __getitem__(self, row):
        return int(row)


class _EchoZero:
    """Always clip 0: a model scene holds exactly one clip, the seed's."""

    def __getitem__(self, row):
        return 0


class ModelScene:
    """The little a model-driven viewer actually needs.

    A `MotionDatabase` carries the Matching Database -- 27 features a frame,
    normalisation statistics, a KD-tree -- so that a KNN teacher can search
    it. A model searches nothing. It needs a skeleton to pose, a frame rate,
    and one real pose to start from, and building the rest costs 23 seconds
    of startup for something that is never read.

    So this is the model's stand-in, exposing only the handful of attributes
    the viewers touch. It is not a `MotionDatabase` and deliberately does not
    pretend to be one: `next_frame_of` raises rather than returning something
    plausible, because anything calling it has mistaken this for a database
    and the useful outcome is an error that says so.
    """

    def __init__(self, skeleton, frame_time, poses, clips, motions=None):
        from types import SimpleNamespace

        self.skeleton = skeleton
        self.frame_time = frame_time
        self.clips = list(clips)
        self.motions = list(motions or [])
        self.animation = SimpleNamespace(
            poses=poses, n_rows=len(poses), pose_dim=int(poses.shape[1])
        )
        # The HUD reads `matching.frame_in_clip[row]` every frame. A generated
        # pose has no frame in any clip, and the controller says so by
        # carrying row -1, so this echoes the row straight back and the HUD
        # reads "frame -1". Returning a plausible-looking number instead would
        # claim the pose came from mocap, which is the opposite of true.
        self.matching = SimpleNamespace(
            frame_in_clip=_EchoRow(),
            clip_ids=_EchoZero(),
            n_rows=len(poses),
        )

    @property
    def n_rows(self) -> int:
        return self.animation.n_rows

    def clip_of(self, row: int):
        """Which clip a row came from.

        The HUD asks this every frame. A model-driven state carries row -1 --
        nothing is being played back -- so the seed clip is returned and the
        HUD shows where the character started rather than crashing. Unlike
        `next_frame_of`, answering here is honest: the question has a
        sensible answer and the caller is not confused about what it holds.
        """
        return self.clips[0]

    def next_frame_of(self, rows):
        raise RuntimeError(
            "ModelScene holds no Matching Database -- a model retrieves "
            "nothing, so there is no next row to advance to. Whatever called "
            "this expected a MotionDatabase."
        )

    def __repr__(self) -> str:
        return (
            f"ModelScene({self.skeleton.n_joints} joints, "
            f"{self.animation.n_rows:,} poses, {len(self.clips)} clips)"
        )


def load_model_scene(bvh_dir, clips, max_frames=4000, layout_path=None,
                     joint_names=None):
    """Load the minimum a model needs, and the obstacle layout to draw.

    One clip, not the whole set: the only things taken from it are the joint
    tree, the frame rate and a seed pose. Loading eight clips to read one
    frame out of the first is what made startup slow.

    The obstacle layout comes from its own JSON and never needed a database
    at all, so the boxes still appear.
    """
    import warnings

    from mmm.data import bvh as bvh_loader
    from mmm.data.database import ClipInfo, motion_to_pose
    from mmm.data.skeleton import LOCOMOTION_JOINTS
    from mmm.obstacles.geometry import BoxField

    joint_names = list(joint_names or LOCOMOTION_JOINTS)
    names = list(clips) if isinstance(clips, (list, tuple)) else [clips]
    if not names:
        raise ValueError("no clip given to seed the model from")

    name = names[0]
    path = os.path.join(bvh_dir, name + ".bvh")
    if not os.path.isfile(path):
        import glob

        matches = sorted(glob.glob(os.path.join(bvh_dir, name + "*.bvh")))
        if not matches:
            raise FileNotFoundError(f"no clip matching {name!r} in {bvh_dir}")
        path = matches[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        motion = bvh_loader.load(path)
    motion = motion.subset(joint_names)
    if max_frames and motion.n_frames > max_frames:
        motion = motion.slice(0, max_frames)
    motion.name = os.path.splitext(os.path.basename(path))[0]

    poses = motion_to_pose(motion)
    scene = ModelScene(
        skeleton=motion.skeleton,
        frame_time=motion.frame_time,
        poses=poses,
        clips=[ClipInfo(name=motion.name, start=0, length=len(poses))],
        motions=[motion],
    )

    studio = BoxField()
    if layout_path and os.path.isfile(layout_path):
        from mmm.obstacles.database import load_fields

        fields = load_fields(layout_path)
        if fields:
            studio = next(iter(fields.values()))

    return scene, studio


def seed_pose_from_database(database, frame: int = DEFAULT_SEED_FRAME):
    """A real pose to start a model from, and the row it came from.

    The model predicts the *next* pose, so frame zero has to be given one.
    It has to be a real pose: zeros are not a valid rotation and the model has
    never seen anything resembling them.

    The frame is taken within the first clip and clamped to it, so a short
    clip still yields something sensible rather than a row from whichever clip
    happens to follow.
    """
    clips = getattr(database, "clips", None)
    if clips:
        clip = clips[0]
        row = clip.start + min(max(frame, 0), max(clip.length - 1, 0))
    else:
        row = min(max(frame, 0), database.animation.n_rows - 1)
    return np.asarray(database.animation.poses[row], np.float32), int(row)


def build_controller(args, database, augmented, field, teacher_kwargs: dict):
    """A teacher or a model controller, by whether `--model` was given.

    Returns `(controller, drawn_database, description)`. `drawn_database` is
    what the viewer should draw from -- the augmented one when the teacher
    needs obstacle-aware retrieval, the plain one otherwise.
    """
    if not args.model:
        from mmm.obstacles.controller import ObstacleController

        controller = ObstacleController(augmented, field=field, **teacher_kwargs)
        return controller, augmented, "KNN teacher"

    from mmm.models.runtime import ModelController

    seed, row = seed_pose_from_database(database, args.seed_frame)
    controller = ModelController.from_checkpoint(
        args.model,
        skeleton=database.skeleton,
        seed_pose=seed,
        manifest=args.model_manifest,
        device=args.device,
        root_source=args.root_source,
        frame_time=database.frame_time,
    )
    # The model has no obstacle input in Phase 1, so a field placed in the
    # scene is drawn but not reacted to. Saying so is better than letting
    # someone conclude the model ignores geometry it was never shown.
    if field is not None and len(field):
        print(
            f"note: {len(field)} obstacles are in the scene, but this model "
            "has no obstacle input -- they are drawn, not avoided."
        )
    return (
        controller,
        database,
        f"model {os.path.basename(args.model)} (seed row {row}, "
        f"root_source={args.root_source})",
    )
