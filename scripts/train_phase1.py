"""
Phase 1 entry point: pretrain the motion prior on raw mocap.

Learns how bodies move -- next pose prediction from the joystick request and
the model's own previous pose. Retrieval is **not** in the input: measured on
LaFAN1, the true previous pose scores 2.29 against 16.24 for the nearest
retrieved candidate, so under teacher forcing the candidate block carries no
signal the objective rewards, and training with it risks teaching the model
that its knowledge base is noise. Retrieval enters in Phase 2, where the target
is the KNN teacher's own output. `--use-retrieval` restores it for that
ablation.

Runs unchanged on a workstation, in a Kaggle notebook and in Colab: paths are
resolved by `mmm.workspace` rather than assumed, because Kaggle mounts the
dataset read only and anything written beside it fails minutes into a session.

Local:
    python scripts/train_phase1.py --dataset data/bvh --epochs 50

Kaggle:
    !python scripts/train_phase1.py \\
        --dataset /kaggle/input/lafan1-resolved \\
        --epochs 50 \\
        --hf-repo ernestbeckham/mmm

`--dataset` takes a built database directory, a directory to search for BVH
files, or a single BVH file; the database is built and cached on first use, so
a restarted session does not pay for it twice.

Publishing needs a token in the environment, never on the command line. On
Kaggle, add it under Add-ons -> Secrets as `HF_TOKEN` and attach it to the
notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from mmm.data.database import MotionDatabase
from mmm.data.dataset import WindowDataset, collate
from mmm.data.prepare import LOCOMOTION_PATTERN, ensure_database
from mmm.models.baselines.gru import GruBackbone
from mmm.models.io_layers import IoSpec
from mmm.models.mamba_backbone import MambaBackbone, MambaConfig
from mmm.models.policy import MmmPolicy
from mmm.training.losses import LossWeights
from mmm.training.pretrainer import PreTrainer, PreTrainerConfig
from mmm.workspace import Workspace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset",
        default="",
        help="a built database directory, a directory of BVH files, or one BVH "
        "file. On Kaggle this is the read only mount, e.g. "
        "/kaggle/input/lafan1-resolved.",
    )
    # Accepted because earlier revisions and scripts used these names. Argparse
    # would otherwise fail the whole run on an unrecognised flag, several
    # minutes of setup in.
    data.add_argument("--database", dest="dataset_legacy", default="", help=argparse.SUPPRESS)
    data.add_argument("--bvh-dir", dest="dataset_legacy2", default="", help=argparse.SUPPRESS)
    data.add_argument(
        "--pattern",
        default=LOCOMOTION_PATTERN,
        help="comma separated globs on the clip name. '*' takes every clip.",
    )
    data.add_argument("--fps", type=float, default=60.0)
    data.add_argument("--max-frames", type=int, default=0, help="truncate clips; 0 keeps all")
    data.add_argument("--full-skeleton", action="store_true")
    data.add_argument("--rebuild-database", action="store_true")

    paths = parser.add_argument_group("paths (all optional; resolved per platform)")
    paths.add_argument("--out", default="", help="checkpoints, history, TensorBoard")
    paths.add_argument("--work-dir", default="", help="writable base for everything produced")
    paths.add_argument("--database-dir", default="", help="where to build the database")
    paths.add_argument("--cache-dir", default="", help="BVH and retrieval caches")
    paths.add_argument("--run-name", default="phase1")

    windows = parser.add_argument_group("windows")
    windows.add_argument("--window", type=int, default=64)
    windows.add_argument("--stride", type=int, default=32)
    windows.add_argument("--batch-size", type=int, default=16)
    windows.add_argument("--validation-fraction", type=float, default=0.15)
    windows.add_argument("--workers", type=int, default=0)

    model = parser.add_argument_group("model")
    model.add_argument("--backbone", default="mamba", choices=["mamba", "gru"])
    model.add_argument("--d-model", type=int, default=256)
    model.add_argument("--n-layers", type=int, default=4)
    model.add_argument("--d-state", type=int, default=16)
    model.add_argument("--head", default="full", choices=["full", "residual"])
    model.add_argument("--no-contacts", action="store_true")
    model.add_argument("--input-dropout", type=float, default=0.0)
    model.add_argument("--hidden-dropout", type=float, default=0.0)

    retrieval = parser.add_argument_group("retrieval (Phase 2 ablation)")
    retrieval.add_argument(
        "--use-retrieval",
        action="store_true",
        help="put the retrieved candidates in the input, widening it from 163 "
        "to 771. Off by default; see this script's docstring.",
    )
    retrieval.add_argument("--k", type=int, default=4)
    retrieval.add_argument("--retrieval-dropout", type=float, default=0.1)
    retrieval.add_argument("--no-leave-one-clip-out", action="store_true")

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=50)
    optimisation.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    optimisation.add_argument("--weight-decay", type=float, default=1e-4)
    optimisation.add_argument("--grad-clip", type=float, default=1.0)
    optimisation.add_argument(
        "--lr-schedule", default="cosine", choices=["cosine", "linear", "constant", "step"]
    )
    optimisation.add_argument("--warmup-epochs", type=float, default=1.0)
    optimisation.add_argument("--min-lr-factor", type=float, default=0.05)

    drift = parser.add_argument_group("drift")
    drift.add_argument(
        "--ss-start",
        type=int,
        default=5,
        help="epoch at which the model starts being fed its own output. "
        "Measured: three epochs of pure teacher forcing made the free running "
        "loss worse (2.688 -> 3.643) while the teacher forced loss halved, so "
        "starting late costs drift resistance. Starting at 1 is also bad: a "
        "model fed its own untrained output never learns the one step mapping.",
    )
    drift.add_argument("--ss-end", type=int, default=25)
    drift.add_argument("--ss-max", type=float, default=0.5)

    stopping = parser.add_argument_group("selection and stopping")
    stopping.add_argument("--patience", type=int, default=10)
    stopping.add_argument(
        "--monitor",
        default="val_rollout_total",
        help="metric deciding best.safetensors and early stopping. The free "
        "running loss by default, since the teacher forced loss cannot see "
        "drift and Phase 1's deliverable is a drift resistant prior.",
    )
    stopping.add_argument("--monitor-start-epoch", type=int, default=0, help="0 means --ss-start")
    stopping.add_argument("--rollout-eval-batches", type=int, default=8)
    stopping.add_argument("--residual-weight", type=float, default=0.0)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default=None)
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--no-tensorboard", action="store_true")
    runtime.add_argument("--progress-every", type=int, default=25)
    runtime.add_argument(
        "--resume",
        default="",
        help="checkpoint to continue from, or 'auto' to pick up "
        "last.safetensors in --out if it is there. 'auto' is what a Kaggle "
        "session that hit its time limit needs.",
    )

    publish = parser.add_argument_group("publishing")
    publish.add_argument(
        "--hf-repo",
        "--push-to-hub",
        dest="hf_repo",
        default="",
        metavar="REPO_ID",
        help="after the final epoch, publish the best checkpoint to this "
        "Hugging Face repository. Private unless --hf-public. Needs HF_TOKEN "
        "in the environment; on Kaggle attach it as a notebook secret.",
    )
    publish.add_argument("--hf-public", action="store_true")
    publish.add_argument("--hf-license", default="")

    args = parser.parse_args(argv)
    args.dataset = args.dataset or args.dataset_legacy or args.dataset_legacy2
    if not args.dataset:
        parser.error("--dataset is required (a database directory, a BVH directory, or a BVH file)")
    return args


def build_policy(args, database: MotionDatabase) -> MmmPolicy:
    spec = IoSpec(
        pose_dim=database.pose_dim,
        n_candidates=args.k,
        use_retrieval=args.use_retrieval,
        predict_contacts=not args.no_contacts,
    )
    if args.backbone == "mamba":
        backbone = MambaBackbone(
            MambaConfig(d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state)
        )
    else:
        backbone = GruBackbone(d_model=args.d_model, n_layers=args.n_layers)

    return MmmPolicy(
        spec,
        backbone=backbone,
        head=args.head,
        d_model=args.d_model,
        retrieval_dropout=args.retrieval_dropout,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

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
    )
    database = MotionDatabase.load(database_path)
    print(database)

    dataset = WindowDataset(
        database,
        window=args.window,
        stride=args.stride,
        k=args.k,
        leave_one_clip_out=not args.no_leave_one_clip_out,
        seed=args.seed,
        use_retrieval=args.use_retrieval,
        # Never beside the database: on Kaggle that can be a read only mount.
        cache_dir=workspace.cache,
    )
    train_set, val_set = dataset.split(validation_fraction=args.validation_fraction)
    print(f"{len(train_set)} training windows, {len(val_set)} validation windows")

    train_clips = sorted({database.clips[c].name for c, _ in train_set.windows})
    val_clips = sorted({database.clips[c].name for c, _ in val_set.windows})
    print(f"  train clips: {len(train_clips)}  {train_clips}")
    print(f"  val clips:   {len(val_clips)}  {val_clips}")
    # Split by clip, not by window: consecutive frames of a 60Hz clip are
    # nearly identical, so sharing a clip across the split would make
    # validation measure memorisation.
    overlap = set(train_clips) & set(val_clips)
    assert not overlap, f"clips in both splits: {sorted(overlap)}"
    if not val_clips:
        print("no validation clips; raise --validation-fraction or add clips", file=sys.stderr)
        return 1

    policy = build_policy(args, database)
    print(f"policy: {policy.n_parameters():,} parameters")
    for key, value in policy.describe().items():
        print(f"  {key:18} {value}")

    config = PreTrainerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        scheduled_sampling_start=args.ss_start,
        scheduled_sampling_end=args.ss_end,
        scheduled_sampling_max=args.ss_max,
        patience=args.patience,
        device=args.device,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        min_lr_factor=args.min_lr_factor,
        monitor=args.monitor,
        monitor_start_epoch=args.monitor_start_epoch or None,
        rollout_eval_batches=args.rollout_eval_batches,
        tensorboard=not args.no_tensorboard,
        progress_every=args.progress_every,
        loss_weights=LossWeights(residual=args.residual_weight),
    )
    trainer = PreTrainer(policy, database, config, checkpoint_dir=workspace.out)
    print(f"device: {trainer.device}")

    resume = args.resume
    if resume == "auto":
        candidate = os.path.join(workspace.out, "last.safetensors")
        resume = candidate if os.path.exists(candidate) else ""
        print(f"--resume auto: {'resuming ' + resume if resume else 'nothing to resume from'}")
    if resume:
        trainer.resume(resume)

    with open(os.path.join(workspace.out, "run_args.json"), "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "workspace": workspace.describe()}, handle, indent=2)

    loader_kwargs = dict(collate_fn=collate, num_workers=args.workers)
    if args.workers:
        # Each worker would otherwise re-parse nothing but still copy the
        # database; keeping them alive avoids paying that every epoch.
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    trainer.fit(train_loader, val_loader)
    print(f"\nwrote checkpoints and history to {workspace.out}")

    if args.hf_repo:
        best = os.path.join(workspace.out, "best.safetensors")
        if not os.path.exists(best):
            print("no best checkpoint to publish", file=sys.stderr)
            return 1
        trainer.load(best)
        # Without the optimizer: two thirds of the file is Adam state that is
        # useless to anyone running the model.
        published = trainer.save(
            os.path.join(workspace.out, "published"), epoch=None, optimizer=False
        )

        from publish_checkpoint import main as publish

        argv = ["--repo-id", args.hf_repo, "--checkpoint", published]
        if args.hf_public:
            argv.append("--public")
        if args.hf_license:
            argv += ["--license", args.hf_license]
        print(f"\npublishing {published} to {args.hf_repo}")
        return publish(argv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
