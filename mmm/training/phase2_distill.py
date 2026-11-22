"""
Phase 2: distil from the KNN teacher. Learns game style joystick response.

Plan section 5.3, in three steps:

1. **Round 1 -- imitation.** Drive the Phase 0 teacher with a synthetic
   joystick, for a long time, recording everything it does:
   `scripts/collect_teacher_data.py`.
2. **SFT.** Train the model to reproduce what was recorded. Done by reusing
   `mmm.training.pretrainer.PreTrainer` wholesale: see "Why this reuses
   PreTrainer" below.
3. **DAgger rounds.** Run the *student* instead of the teacher, ask the
   teacher what it would do at every state the student actually reaches, add
   those corrections to the dataset, and fine tune again:
   `scripts/dagger_round.py`.

Why this reuses `PreTrainer`
-----------------------------
`PreTrainer` already is next pose prediction with scheduled sampling, over
windows carrying `joystick, previous_pose, target, contacts`. None of that is
Phase 1 specific. Point it at a dataset of logged teacher rollouts instead of
raw mocap windows, and every mechanism -- calibration, scheduled sampling,
the rollout evaluation, early stopping, checkpointing -- is correct for
Phase 2 without being rewritten. The one thing that was Phase 1 specific, the
manifest's `"phase": 1`, is now a constructor argument
(`PreTrainer(..., phase=2)`).

This module is therefore thin on purpose: a policy builder that performs the
Phase 1 to Phase 2 handoff, and the glue a script needs to go from a
checkpoint and a directory of rollouts to a trained `PreTrainer`.
"""

from __future__ import annotations

from mmm.data.database import MotionDatabase
from mmm.data.rollout_dataset import RolloutWindowDataset
from mmm.models.policy import MmmPolicy
from mmm.models.runtime import build_policy
from mmm.training.checkpoints import load_checkpoint, read_metadata
from mmm.training.pretrainer import PreTrainer, PreTrainerConfig

__all__ = ["build_sft_policy", "build_policy_for_resume", "build_sft_dataset", "build_sft_trainer"]


def _read_architecture(checkpoint: str) -> dict:
    import json
    import os

    manifest_path = os.path.join(os.path.dirname(checkpoint), "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            architecture = (json.load(handle) or {}).get("architecture")
    else:
        architecture = (read_metadata(checkpoint) or {}).get("architecture")
    if not architecture:
        raise ValueError(
            f"no architecture found for {checkpoint}. Looked in "
            f"{manifest_path} and the checkpoint's own metadata."
        )
    return architecture


def build_policy_for_resume(resume_checkpoint: str, device: str = "cpu") -> tuple[MmmPolicy, dict]:
    """A policy shaped to match a Phase 2 checkpoint that will be resumed.

    Used instead of `build_sft_policy` when `--resume` is given: resuming
    overwrites every weight anyway (`PreTrainer.resume` restores weights,
    optimizer moments and calibration), so loading the Phase 1 checkpoint's
    weights first is pure waste -- only its *architecture* ever mattered, and
    the resume checkpoint carries the same architecture itself (Phase 2 never
    changes shape; see `build_sft_policy`'s docstring). This lets a resumed
    SFT or DAgger run skip needing the original Phase 1 checkpoint at all.

    Args:
        resume_checkpoint: the Phase 2 `.safetensors` file about to be resumed.
        device: where to put the returned policy. Weights are left at
            initialisation -- `trainer.resume()` overwrites them immediately
            after, so loading real weights here would be redundant.

    Returns:
        `(policy, architecture)`.
    """
    architecture = _read_architecture(resume_checkpoint)
    policy = build_policy(architecture)
    return policy.to(device), architecture


def build_sft_policy(phase1_checkpoint: str, device: str = "cpu") -> tuple[MmmPolicy, dict]:
    """The Phase 1 model, loaded as-is for Phase 2 SFT.

    Phase 2 trains on the KNN teacher's labeled rollouts instead of raw
    mocap, but reads the exact same input -- joystick plus the model's own
    previous pose, no retrieved candidates -- so this is a plain checkpoint
    load, not an architecture change. (An earlier version of this function
    widened the input layer to add a retrieval block; that mechanism has
    been removed. Retrieval is coming back later as an encoder feeding the
    backbone through cross-attention, which does not need this handoff to
    change shape at all.)

    Args:
        phase1_checkpoint: a Phase 1 `.safetensors` file. Its `manifest.json`
            (same folder, or `<checkpoint>.json` if that is absent) supplies
            the architecture.
        device: where to put the returned policy.

    Returns:
        `(policy, architecture)`.
    """
    architecture = _read_architecture(phase1_checkpoint)
    policy = build_policy(architecture)
    load_checkpoint(phase1_checkpoint, policy=policy, device="cpu", strict=True)
    return policy.to(device), architecture


def build_sft_dataset(
    database: MotionDatabase,
    rollout_paths,
    window: int = 64,
    stride: int = 32,
    elevated_contacts: bool = False,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[RolloutWindowDataset, RolloutWindowDataset]:
    """The train/validation split over a directory of logged rollouts.

    A thin wrapper -- `RolloutWindowDataset(...).split(...)` -- kept as its
    own function so a script has one call to make and one place to change the
    defaults in step with `build_sft_trainer`.
    """
    dataset = RolloutWindowDataset(
        database,
        rollout_paths,
        window=window,
        stride=stride,
        elevated_contacts=elevated_contacts,
    )
    return dataset.split(validation_fraction=validation_fraction, seed=seed)


def build_sft_trainer(
    policy: MmmPolicy,
    database: MotionDatabase,
    config: PreTrainerConfig | None = None,
    checkpoint_dir: str = "runs/phase2",
    distributed=None,
) -> PreTrainer:
    """`PreTrainer`, configured for Phase 2 rather than for Phase 1.

    Identical machine, different defaults: Phase 2 starts from a working
    motion prior, not from nothing, so there is no reason to warm up into
    teacher forcing the way Phase 1 does. `config` is accepted as-is so a
    caller can override any of this; these are only the sane starting point
    for *this* phase.
    """
    config = config or PreTrainerConfig(
        epochs=30,
        learning_rate=3e-4,
        scheduled_sampling_start=1,
        scheduled_sampling_end=10,
        scheduled_sampling_max=0.5,
        monitor="val_rollout_total",
    )
    return PreTrainer(
        policy,
        database,
        config=config,
        checkpoint_dir=checkpoint_dir,
        distributed=distributed,
        phase=2,
    )
