"""
Phase 1 entry point: pretrain on raw mocap.

Thin wrapper over `mmm.training.pretrainer.PreTrainer`. Loads the database,
builds the windowed dataset with leave one clip out retrieval, builds the
policy, asserts the split is clean, and trains.

Usage:
    python scripts/train_phase1.py --database data/databases/locomotion
    python scripts/train_phase1.py --epochs 100 --window 64 --batch-size 8 --head residual
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from mmm.data.database import MotionDatabase
from mmm.data.dataset import WindowDataset, collate
from mmm.models.baselines.gru import GruBackbone
from mmm.models.io_layers import IoSpec
from mmm.models.mamba_backbone import MambaBackbone, MambaConfig
from mmm.models.policy import MmmPolicy
from mmm.training.losses import LossWeights
from mmm.training.pretrainer import PreTrainer, PreTrainerConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", default=os.path.join("data", "databases", "locomotion"))
    parser.add_argument("--out", default=os.path.join("runs", "phase1"))

    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--no-leave-one-clip-out",
        action="store_true",
        help="allow a frame's candidates to come from its own clip. Only for "
        "an ablation: it lets the model satisfy the loss by copying itself.",
    )

    parser.add_argument("--backbone", default="mamba", choices=["mamba", "gru"])
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--head", default="full", choices=["full", "residual"])
    parser.add_argument("--retrieval-dropout", type=float, default=0.1)
    parser.add_argument(
        "--use-retrieval",
        action="store_true",
        help="include the retrieved candidates in the input (771 wide). OFF by "
        "default: pre-training learns the motion prior from joystick plus "
        "previous pose (163 wide). Measured, retrieval contributes -0.0013 "
        "here, because the true previous pose scores 2.29 against 16.24 for "
        "the nearest candidate. Retrieval belongs in Phase 2, where the target "
        "is the teacher's own output. Use this flag only for that ablation.",
    )
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dropout", type=float, default=0.0)
    parser.add_argument("--no-contacts", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--lr-schedule", default="cosine", choices=["cosine", "linear", "constant", "step"]
    )
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-lr-factor", type=float, default=0.05)
    parser.add_argument(
        "--monitor",
        default="val_rollout_total",
        help="metric deciding best.safetensors and early stopping. Defaults to "
        "the free running loss, since a drift resistant prior is what Phase 1 "
        "is for and the teacher forced loss cannot see drift.",
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--resume", default="", help="checkpoint to continue from")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--residual-weight", type=float, default=0.0)
    parser.add_argument(
        "--push-to-hub",
        default="",
        metavar="REPO_ID",
        help="after the final epoch, publish the best checkpoint to this "
        "Hugging Face repository (private). Off by default: pushing is "
        "publishing, so it should be a deliberate act. Pushes ONCE at the "
        "end, never per epoch. Equivalent to running "
        "scripts/publish_checkpoint.py afterwards.",
    )
    parser.add_argument(
        "--retrieval-cache",
        default="",
        help="where to keep the precomputed retrieval. Defaults to the "
        "database directory. Pass 'none' to recompute every run.",
    )
    return parser.parse_args(argv)


def build_policy(args, database: MotionDatabase) -> MmmPolicy:
    spec = IoSpec(
        pose_dim=database.pose_dim,
        n_candidates=args.k,
        use_retrieval=args.use_retrieval,
        predict_contacts=not args.no_contacts,
    )
    if args.backbone == "mamba":
        backbone = MambaBackbone(MambaConfig(d_model=args.d_model, n_layers=args.n_layers))
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

    if not os.path.exists(os.path.join(args.database, "manifest.json")):
        print(f"no database at {args.database}. Run scripts/build_databases.py first.", file=sys.stderr)
        return 1

    database = MotionDatabase.load(args.database)
    print(database)

    # Cache the retrieval beside the database. It takes about seven minutes on
    # 264,000 rows and depends only on the database and the retrieval settings,
    # so every run after the first starts in seconds.
    dataset = WindowDataset(
        database,
        window=args.window,
        stride=args.stride,
        k=args.k,
        leave_one_clip_out=not args.no_leave_one_clip_out,
        seed=args.seed,
        use_retrieval=args.use_retrieval,
        cache_dir=None
        if args.retrieval_cache == "none"
        else (args.retrieval_cache or args.database),
    )
    train_set, val_set = dataset.split(validation_fraction=args.validation_fraction)
    print(f"{len(train_set)} training windows, {len(val_set)} validation windows")

    train_clips = sorted({database.clips[c].name for c, _ in train_set.windows})
    val_clips = sorted({database.clips[c].name for c, _ in val_set.windows})
    print(f"  train clips: {train_clips}")
    print(f"  val clips:   {val_clips}")
    # Split by clip, not by window: consecutive frames of a 60Hz clip are
    # nearly identical, so sharing a clip across the split would make
    # validation measure memorisation.
    overlap = set(train_clips) & set(val_clips)
    assert not overlap, f"clips in both splits: {sorted(overlap)}"

    policy = build_policy(args, database)
    print(f"policy: {policy.n_parameters():,} parameters")
    for key, value in policy.describe().items():
        print(f"  {key:18} {value}")

    config = PreTrainerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=args.device,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        min_lr_factor=args.min_lr_factor,
        monitor=args.monitor,
        tensorboard=not args.no_tensorboard,
        loss_weights=LossWeights(residual=args.residual_weight),
    )
    trainer = PreTrainer(policy, database, config, checkpoint_dir=args.out)
    print(f"device: {trainer.device}")
    if args.resume:
        trainer.resume(args.resume)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.workers,
    )

    trainer.fit(train_loader, val_loader)
    print(f"\nwrote checkpoints and history to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
