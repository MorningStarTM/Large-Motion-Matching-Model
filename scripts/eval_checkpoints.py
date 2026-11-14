"""
Free-running rollout evaluation for one checkpoint, with and without retrieval.

For a Phase 2 (retrieval) checkpoint, runs the exact same joystick session
twice -- once with the model searching the database every `--search-interval`
frames as it was trained to, once with the real search still running (so the
numbers are comparable) but zeroed candidates handed to the policy instead.
This is the same ablation `PreTrainer._validate` computes during training
(`val_rollout_total_without_retrieval`), run here on an actual rollout rather
than teacher-forced validation windows -- the question this answers is
whether retrieval is pulling its weight in the regime that actually matters,
free running, not just lowering a validation number.

A Phase 1 checkpoint has nothing to ablate, so it runs once.

Two kinds of number are reported, for every rollout:

1. **Visible symptoms, not losses:** foot sliding (how fast a planted foot
   moves when it shouldn't), hips height (the "floating" failure this
   project chased down once already), and jitter (how rough the root's own
   acceleration is).
2. **Agreement with the KNN teacher, numerically.** At every frame, before
   advancing, the model's own exact current state is handed to
   `KnnController.reference_pose` -- the same "what would the real teacher
   do from here" primitive `scripts/dagger_round.py` uses to build DAgger
   labels. That answer is never fed back into the model; it is only kept as
   a label to score the model's own next output against, with the project's
   own `PoseLoss` (the identical rotation/position/velocity/root terms every
   training log already reports), using the checkpoint's own calibrated
   weights when its manifest has them. This is what makes different
   checkpoints -- Phase 1, SFT, a DAgger round, with or without retrieval --
   comparable on one scale: how far each one drifts from what classic motion
   matching would have done, from the states each one actually visits.

Usage:
    python scripts/eval_checkpoints.py --dataset data/bvh/bvh --pattern "*" --trim-calibration --checkpoint runs/phase2_sft_v1/best.safetensors --frames 1800
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from mmm.data.contacts import CONTACT_JOINTS, detect_contacts
from mmm.data.database import MotionDatabase, pose_to_hips_height, pose_to_rotations
from mmm.data.fk import forward_kinematics
from mmm.data.prepare import LOCOMOTION_PATTERN, ensure_database
from mmm.data.skeleton import HIPS
from mmm.envs.joystick import JoystickConfig, SyntheticJoystick
from mmm.models.runtime import ModelController, load_policy_from_manifest
from mmm.teacher.controller import KnnController
from mmm.teacher.search import KnnIndex
from mmm.training.checkpoints import read_metadata
from mmm.training.losses import LossWeights, PoseLoss
from mmm.viewer.session import DEFAULT_SEED_FRAME, seed_pose_from_database
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
    data.add_argument("--trim-calibration", action="store_true")
    data.add_argument("--database-dir", default="")
    data.add_argument("--cache-dir", default="")

    model = parser.add_argument_group("model")
    model.add_argument("--checkpoint", required=True)
    model.add_argument("--manifest", default="")
    model.add_argument("--device", default="cpu")
    model.add_argument("--root-source", default="hybrid", choices=["hybrid", "simulation", "animation"])
    model.add_argument("--search-interval", type=int, default=10)
    model.add_argument("--k", type=int, default=4)
    model.add_argument("--seed-frame", type=int, default=DEFAULT_SEED_FRAME)

    run = parser.add_argument_group("rollout")
    run.add_argument("--frames", type=int, default=1800, help="1800 is 30s at 60Hz")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", default="", help="write the metrics as JSON here")

    return parser.parse_args(argv)


def compute_metrics(poses: np.ndarray, skeleton, frame_time: float) -> dict:
    """Foot sliding, hips height and jitter over one rollout's poses."""
    n_joints = skeleton.n_joints
    rotations = pose_to_rotations(poses, n_joints)
    heights = pose_to_hips_height(poses, n_joints)
    root = np.zeros((len(poses), 3), np.float32)
    root[:, 1] = heights
    positions, _ = forward_kinematics(skeleton, rotations, root)
    contacts = detect_contacts(positions, skeleton, frame_time)

    sliding_speeds = []
    for i, name in enumerate(CONTACT_JOINTS):
        joint = skeleton.index(name)
        horizontal = positions[:, joint, [0, 2]]
        velocity = np.linalg.norm(np.diff(horizontal, axis=0), axis=-1) / frame_time
        planted = contacts[1:, i] > 0.5
        if planted.any():
            sliding_speeds.append(float(velocity[planted].mean()))
    # Positions (and hips height below) are already in centimetres -- this
    # codebase's whole convention, same reason PoseLoss divides by a
    # position_scale of 100 rather than the other way around. No further
    # scaling here; an earlier version of this function wrongly multiplied
    # by 100 again, inflating a real ~20 cm/s into ~2000 cm/s.
    sliding_cm_s = float(np.mean(sliding_speeds)) if sliding_speeds else float("nan")

    hips = positions[:, skeleton.index(HIPS)]
    jitter = np.linalg.norm(np.diff(hips, n=2, axis=0), axis=-1) / (frame_time**2)

    return {
        "frames": int(len(poses)),
        "finite": bool(np.isfinite(poses).all()),
        "foot_sliding_cm_s": sliding_cm_s,
        "hips_height_mean_cm": float(heights.mean()),
        "hips_height_std_cm": float(heights.std()),
        "hips_height_min_cm": float(heights.min()),
        "jitter_mean": float(jitter.mean()) if len(jitter) else float("nan"),
    }


def rollout(policy, database, index, teacher, pose_loss, seed_pose, args, ablate_retrieval: bool) -> dict:
    """One free-running session, driven by a fresh, seeded joystick.

    At every frame, before the model advances, the teacher is asked what it
    would do from the model's own exact current state (`teacher.reference_pose`)
    -- never fed back in, only kept to score this frame's real output against
    afterwards.
    """
    joystick = SyntheticJoystick(
        frame_time=database.frame_time, config=JoystickConfig(), seed=args.seed
    )
    use_retrieval = bool(getattr(policy.spec, "use_retrieval", False))
    controller = ModelController(
        policy, database.skeleton, seed_pose,
        frame_time=database.frame_time, device=args.device,
        root_source=args.root_source,
        database=database if use_retrieval else None,
        index=index if use_retrieval else None,
        search_interval=args.search_interval, k=args.k,
        ablate_retrieval=ablate_retrieval,
    )

    poses = np.empty((args.frames, database.animation.pose_dim), np.float32)
    teacher_labels = np.empty_like(poses)
    for i in range(args.frames):
        # Captured before step(): the question teacher.reference_pose answers
        # is about the state the model is REALLY in right now, and the forward
        # has to be the one this frame's own search/trajectory used internally
        # -- see scripts/dagger_round.py for the same ordering and the
        # measurement that found it matters (up to 31 of 12 numbers off
        # otherwise).
        state_before = controller._previous_pose[0].detach().cpu().numpy().astype(np.float32)
        forward_before_turn = controller.state.forward.copy()

        stick, facing = joystick.step()
        record = controller.step(stick, facing)
        poses[i] = record.pose

        desired_positions, desired_directions = controller.trajectory.predict(forward_before_turn)
        teacher_labels[i], _ = teacher.reference_pose(
            pose=state_before, desired_positions=desired_positions, desired_directions=desired_directions,
        )

    metrics = compute_metrics(poses, database.skeleton, database.frame_time)
    metrics["ablate_retrieval"] = ablate_retrieval
    metrics["used_retrieval_input"] = use_retrieval and not ablate_retrieval

    total, terms = pose_loss(
        torch.from_numpy(poses)[None], torch.from_numpy(teacher_labels)[None]
    )
    metrics["vs_teacher"] = {"total": float(total), **terms}
    return metrics


def _print_row(label: str, metrics: dict) -> None:
    vs = metrics["vs_teacher"]
    print(
        f"  {label:18s} sliding {metrics['foot_sliding_cm_s']:7.1f} cm/s | "
        f"hips {metrics['hips_height_mean_cm']:6.1f}"
        f" (+/-{metrics['hips_height_std_cm']:4.1f}, min {metrics['hips_height_min_cm']:6.1f}) cm | "
        f"jitter {metrics['jitter_mean']:8.1f} | finite={metrics['finite']}"
    )
    print(
        f"  {'':18s} vs teacher: total {vs['total']:7.4f}  "
        f"(rot {vs.get('rotation', float('nan')):.4f} "
        f"pos {vs.get('position', float('nan')):.4f} "
        f"vel {vs.get('velocity', float('nan')):.4f} "
        f"root {vs.get('root', float('nan')):.4f})"
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    workspace = Workspace.resolve(database=args.database_dir or None, cache=args.cache_dir or None)
    database_path = ensure_database(
        source=args.dataset, out_dir=workspace.database, cache_dir=workspace.cache,
        pattern=args.pattern, fps=args.fps, max_frames=args.max_frames,
        full_skeleton=args.full_skeleton, trim_calibration=args.trim_calibration,
    )
    database = MotionDatabase.load(database_path)
    print(database)

    policy, architecture, _ = load_policy_from_manifest(
        args.checkpoint, manifest=args.manifest or None, device=args.device
    )
    # Always built, not only for a retrieval checkpoint: this index backs the
    # teacher used to score EVERY checkpoint against classic motion matching,
    # independent of whether the checkpoint itself retrieves anything.
    index = KnnIndex(database)
    teacher = KnnController(database, index=index, k=args.k)

    weights = LossWeights()
    calibrated = (read_metadata(args.checkpoint) or {}).get("loss_weights_calibrated")
    if calibrated:
        weights = LossWeights(**calibrated)
    pose_loss = PoseLoss(database.skeleton, frame_time=database.frame_time, weights=weights)

    seed_pose, row = seed_pose_from_database(database, args.seed_frame)

    print(f"\n{os.path.basename(args.checkpoint)} (seed row {row}, {args.frames} frames):")
    with_retrieval = rollout(
        policy, database, index, teacher, pose_loss, seed_pose, args, ablate_retrieval=False
    )
    _print_row("with retrieval", with_retrieval)

    without_retrieval = None
    if architecture.get("use_retrieval"):
        without_retrieval = rollout(
            policy, database, index, teacher, pose_loss, seed_pose, args, ablate_retrieval=True
        )
        _print_row("without retrieval", without_retrieval)
        gain = without_retrieval["foot_sliding_cm_s"] - with_retrieval["foot_sliding_cm_s"]
        teacher_gain = without_retrieval["vs_teacher"]["total"] - with_retrieval["vs_teacher"]["total"]
        print(f"  retrieval reduces sliding by {gain:+.1f} cm/s (positive = retrieval helps)")
        print(f"  retrieval reduces vs-teacher loss by {teacher_gain:+.4f} (positive = retrieval helps)")
    else:
        print("  (Phase 1 checkpoint -- nothing to ablate)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "seed_row": row,
                    "with_retrieval": with_retrieval,
                    "without_retrieval": without_retrieval,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
