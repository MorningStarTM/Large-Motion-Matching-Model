"""
Phase 2 entry point: SFT on round 1 teacher rollouts.

Starts from a finished Phase 1 checkpoint, widens its input layer for
retrieval (`mmm.training.phase2_distill.build_sft_policy`), builds training
windows out of the logs `scripts/collect_teacher_data.py` produced
(`mmm.data.rollout_dataset.RolloutWindowDataset`), and trains with
`mmm.training.pretrainer.PreTrainer` -- the same trainer Phase 1 uses; see
`mmm/training/phase2_distill.py`'s docstring for why that is correct rather
than a shortcut. DAgger rounds are run by `scripts/dagger_round.py`, which
calls back into this module rather than duplicating the training loop, so
that each round is a separate, inspectable step.

`--dataset` here must describe the *same* database `collect_teacher_data.py`
was pointed at -- same source, same `--pattern`, same `--max-frames`, same
`--trim-calibration`. `ensure_database` caches by exactly those inputs, so
matching them reuses the cached build and reproduces its fingerprint, which is
what every rollout's manifest is checked against; the database itself is
never read while collecting, so the two commands can be run on different
machines as long as those flags agree.

Local:
    python scripts/train_phase2.py --dataset data/bvh/bvh --rollouts runs/rollouts/round1 --phase1-checkpoint runs/phase1/published.safetensors --out runs/phase2

Kaggle:
    !python scripts/train_phase2.py \\
        --dataset /kaggle/input/lafan1-resolved \\
        --rollouts /kaggle/input/round1-rollouts \\
        --phase1-checkpoint /kaggle/input/phase1-checkpoint/published.safetensors \\
        --epochs 30 \\
        --hf-repo ernestbeckham/mmm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from mmm.data.database import MotionDatabase
from mmm.data.dataset import collate
from mmm.data.prepare import LOCOMOTION_PATTERN, ensure_database
from mmm.data.rollout_dataset import RolloutWindowDataset
from mmm.training.distributed import DistributedContext
from mmm.training.losses import LossWeights
from mmm.training.phase2_distill import build_sft_policy
from mmm.training.pretrainer import PreTrainer, PreTrainerConfig
from mmm.workspace import Workspace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    data = parser.add_argument_group("data (must match scripts/collect_teacher_data.py)")
    data.add_argument("--dataset", required=True)
    data.add_argument("--pattern", default=LOCOMOTION_PATTERN)
    data.add_argument("--fps", type=float, default=60.0)
    data.add_argument("--max-frames", type=int, default=0)
    data.add_argument("--full-skeleton", action="store_true")
    data.add_argument("--trim-calibration", action="store_true")

    rollouts = parser.add_argument_group("rollouts")
    rollouts.add_argument(
        "--rollouts",
        required=True,
        nargs="+",
        help="one or more rollout directories, globs, or .npz files. DAgger "
        "rounds pass the original collection plus every round's corrections "
        "here, all at once -- nothing is thrown away between rounds.",
    )
    rollouts.add_argument("--elevated-contacts", action="store_true")
    rollouts.add_argument(
        "--allow-fingerprint-mismatch",
        action="store_true",
        help="skip the check that a rollout was recorded against this exact "
        "database. Only for rollouts collected before "
        "mmm.teacher.logger.database_fingerprint started rounding its "
        "normalization statistics -- those carry a fingerprint computed by "
        "the old, stricter hash, which a new session's otherwise-identical "
        "database can never reproduce. Before passing this, confirm by hand "
        "that collection's --dataset/--pattern/--max-frames/--trim-calibration "
        "really do match this run's (collection's collection_summary.json "
        "has them); this flag does not check that for you.",
    )
    rollouts.add_argument("--k", type=int, default=4, help="used only if no rollout ever searched")

    paths = parser.add_argument_group("paths")
    paths.add_argument("--out", default="")
    paths.add_argument("--work-dir", default="")
    paths.add_argument("--database-dir", default="")
    paths.add_argument("--cache-dir", default="")
    paths.add_argument("--run-name", default="phase2")

    model = parser.add_argument_group("model")
    model.add_argument("--phase1-checkpoint", required=True)

    windows = parser.add_argument_group("windows")
    windows.add_argument("--window", type=int, default=64)
    windows.add_argument("--stride", type=int, default=32)
    windows.add_argument("--batch-size", type=int, default=16)
    windows.add_argument("--validation-fraction", type=float, default=0.15)
    windows.add_argument("--workers", type=int, default=0)

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=30)
    optimisation.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=3e-4)
    optimisation.add_argument("--weight-decay", type=float, default=1e-4)
    optimisation.add_argument("--grad-clip", type=float, default=1.0)
    optimisation.add_argument(
        "--lr-schedule", default="cosine", choices=["cosine", "linear", "constant", "step"]
    )
    optimisation.add_argument("--warmup-epochs", type=float, default=0.5)
    optimisation.add_argument("--min-lr-factor", type=float, default=0.05)

    drift = parser.add_argument_group("drift")
    drift.add_argument(
        "--ss-start",
        type=int,
        default=1,
        help="unlike Phase 1, Phase 2 starts from a model that already "
        "produces plausible poses, so there is no reason to wait before "
        "exposing it to its own output.",
    )
    drift.add_argument("--ss-end", type=int, default=10)
    drift.add_argument("--ss-max", type=float, default=0.5)

    stopping = parser.add_argument_group("selection and stopping")
    stopping.add_argument("--patience", type=int, default=8)
    stopping.add_argument("--monitor", default="val_rollout_total")
    stopping.add_argument("--monitor-start-epoch", type=int, default=0, help="0 means --ss-start")
    stopping.add_argument("--rollout-eval-batches", type=int, default=8)

    speed = parser.add_argument_group("speed")
    speed.add_argument("--compile", dest="compile_policy", action="store_true")
    speed.add_argument("--lr-scale", default="none", choices=["none", "linear", "sqrt"])

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default=None)
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--no-tensorboard", action="store_true")
    runtime.add_argument("--progress-every", type=int, default=25)
    runtime.add_argument("--resume", default="")

    publish = parser.add_argument_group("publishing")
    publish.add_argument("--hf-repo", "--push-to-hub", dest="hf_repo", default="", metavar="REPO_ID")
    publish.add_argument("--hf-public", action="store_true")
    publish.add_argument("--hf-license", default="")

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    distributed = DistributedContext.from_env(device=args.device)
    main_rank = distributed.is_main

    def say(*parts, **kwargs):
        if main_rank:
            print(*parts, **kwargs)

    say(distributed)

    workspace = Workspace.resolve(
        out=args.out or None,
        database=args.database_dir or None,
        cache=args.cache_dir or None,
        root=args.work_dir or None,
        run_name=args.run_name,
    )
    say(workspace)

    if not main_rank:
        distributed.barrier()
    database_path = ensure_database(
        source=args.dataset,
        out_dir=workspace.database,
        cache_dir=workspace.cache,
        pattern=args.pattern,
        fps=args.fps,
        max_frames=args.max_frames,
        full_skeleton=args.full_skeleton,
        trim_calibration=args.trim_calibration,
        verbose=main_rank,
    )
    if main_rank:
        distributed.barrier()

    database = MotionDatabase.load(database_path)
    say(database)

    if args.allow_fingerprint_mismatch:
        say(
            "note: --allow-fingerprint-mismatch is set -- training against "
            "rollouts without checking they match this database. Make sure "
            "that's actually true before trusting this run."
        )
    dataset = RolloutWindowDataset(
        database,
        args.rollouts,
        window=args.window,
        stride=args.stride,
        elevated_contacts=args.elevated_contacts,
        check_fingerprint=not args.allow_fingerprint_mismatch,
    )
    n_candidates = dataset.n_candidates or args.k
    if dataset.n_candidates and dataset.n_candidates != args.k:
        say(
            f"note: the logged rollouts used k={dataset.n_candidates}; "
            f"building the policy for that, not --k={args.k}"
        )
    train_set, val_set = dataset.split(validation_fraction=args.validation_fraction, seed=args.seed)
    say(
        f"{len(train_set)} training windows, {len(val_set)} validation windows, "
        f"from {len(dataset.source_paths)} rollout file(s), k={n_candidates}"
    )

    policy, phase1_architecture = build_sft_policy(
        args.phase1_checkpoint, n_candidates=n_candidates, device="cpu"
    )
    say(f"policy: {policy.n_parameters():,} parameters (extended from {args.phase1_checkpoint})")
    for key, value in policy.describe().items():
        say(f"  {key:18} {value}")

    learning_rate = args.learning_rate
    effective_batch = args.batch_size * distributed.world_size
    if distributed.world_size > 1:
        if args.lr_scale == "linear":
            learning_rate *= distributed.world_size
        elif args.lr_scale == "sqrt":
            learning_rate *= distributed.world_size**0.5
        say(
            f"effective batch {effective_batch} "
            f"({args.batch_size} x {distributed.world_size} processes); "
            f"lr {args.learning_rate:.2e} -> {learning_rate:.2e} [--lr-scale {args.lr_scale}]"
        )

    config = PreTrainerConfig(
        epochs=args.epochs,
        learning_rate=learning_rate,
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
        compile_policy=args.compile_policy,
        loss_weights=LossWeights(),
    )
    trainer = PreTrainer(
        policy, database, config, checkpoint_dir=workspace.out, distributed=distributed, phase=2
    )
    say(f"device: {trainer.device}")

    resume = args.resume
    if resume == "auto":
        candidate = os.path.join(workspace.out, "last.safetensors")
        resume = candidate if os.path.exists(candidate) else ""
        say(f"--resume auto: {'resuming ' + resume if resume else 'nothing to resume from'}")
    if resume:
        trainer.resume(resume)

    if main_rank:
        with open(os.path.join(workspace.out, "run_args.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "args": vars(args),
                    "workspace": workspace.describe(),
                    "world_size": distributed.world_size,
                    "effective_batch": effective_batch,
                    "learning_rate": learning_rate,
                    "n_candidates": n_candidates,
                    "phase1_architecture": phase1_architecture,
                },
                handle,
                indent=2,
            )

    loader_kwargs = dict(collate_fn=collate, num_workers=args.workers)
    if args.workers:
        loader_kwargs["persistent_workers"] = True
    if distributed.enabled:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            train_set, num_replicas=distributed.world_size, rank=distributed.rank,
            shuffle=True, seed=args.seed,
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    try:
        trainer.fit(train_loader, val_loader)
    finally:
        distributed.shutdown()
    say(f"\nwrote checkpoints and history to {workspace.out}")

    if args.hf_repo and main_rank:
        best = os.path.join(workspace.out, "best.safetensors")
        if not os.path.exists(best):
            print("no best checkpoint to publish", file=sys.stderr)
            return 1

        # The whole run folder -- best/last checkpoints, manifest, history,
        # metrics, run_args -- goes up as one subfolder named after this run,
        # the same shape `publish_checkpoint.py --subfolder <name>` has been
        # pushed by hand so far. One subfolder per run keeps every version
        # (v1's SFT, v1's first DAgger round, v2's SFT, ...) separately
        # browsable in the same repo instead of needing a repo each.
        from huggingface_hub import HfApi

        from publish_checkpoint import build_card
        from mmm.training.checkpoints import read_metadata

        subfolder = os.path.basename(os.path.normpath(workspace.out)) or args.run_name
        metadata = read_metadata(best)
        history_path = os.path.join(workspace.out, "history.json")
        history = None
        if os.path.exists(history_path):
            with open(history_path, encoding="utf-8") as handle:
                history = json.load(handle)

        card_args = argparse.Namespace(repo_id=subfolder, license=args.hf_license)
        card = build_card(card_args, metadata, history, "best.safetensors")
        with open(os.path.join(workspace.out, "README.md"), "w", encoding="utf-8") as handle:
            handle.write(card)

        api = HfApi()
        try:
            whoami = api.whoami()
        except Exception as error:  # noqa: BLE001 - the hub raises several types
            print(
                f"not authenticated ({error}). Set HF_TOKEN or run "
                "`huggingface-cli login`.",
                file=sys.stderr,
            )
            return 1
        say(f"authenticated as {whoami.get('name', '?')}")

        api.create_repo(
            repo_id=args.hf_repo, repo_type="model",
            private=not args.hf_public, exist_ok=True,
        )
        say(f"\npublishing {workspace.out} to {args.hf_repo}/{subfolder}")
        api.upload_folder(
            folder_path=workspace.out,
            path_in_repo=subfolder,
            repo_id=args.hf_repo,
            repo_type="model",
            commit_message=f"Phase 2 run {subfolder}, epoch {metadata.get('epoch')}",
        )
        say(f"\nhttps://huggingface.co/{args.hf_repo}/tree/main/{subfolder}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
