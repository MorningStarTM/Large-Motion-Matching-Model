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
windows carrying `joystick, previous_pose, candidates, mask, target, contacts`
-- and it is already conditional on `use_retrieval` throughout: `_rollout`
passes `candidates`/`mask` through only when they exist, `build_input` reads
them only when the spec asks for them, and `validate` already computes
`retrieval_gain` by ablating them. None of that is Phase 1 specific; Phase 1
simply never exercises the retrieval path because its dataset never populates
it. Point `PreTrainer` at a policy built with `use_retrieval=True` and a
dataset of logged teacher rollouts instead of raw mocap windows, and every
mechanism -- calibration, scheduled sampling, the rollout evaluation, early
stopping, checkpointing -- is correct for Phase 2 without being rewritten. The
one thing that was Phase 1 specific, the manifest's `"phase": 1`, is now a
constructor argument (`PreTrainer(..., phase=2)`).

This module is therefore thin on purpose: a policy builder that performs the
Phase 1 to Phase 2 handoff, and the glue a script needs to go from a
checkpoint and a directory of rollouts to a trained `PreTrainer`.
"""

from __future__ import annotations

import torch

from mmm.data.database import MotionDatabase
from mmm.data.rollout_dataset import RolloutWindowDataset
from mmm.models.policy import MmmPolicy
from mmm.models.runtime import build_policy
from mmm.training.checkpoints import extend_input_layer, load_checkpoint, read_metadata
from mmm.training.pretrainer import PreTrainer, PreTrainerConfig

__all__ = ["build_sft_policy", "build_sft_dataset", "build_sft_trainer"]


def build_sft_policy(
    phase1_checkpoint: str,
    n_candidates: int = 4,
    device: str = "cpu",
) -> tuple[MmmPolicy, dict]:
    """The Phase 1 model, with retrieval added to its input.

    The backbone (88% of the parameters) and the output layer transfer
    *wholesale* and unchanged: neither depends on `use_retrieval` at all --
    `OutputLayer` is a function of `pose_dim` and `n_feet` only, and the
    backbone never sees the input layout, only `d_model`. Only the input
    layer changes shape, because only it reads the assembled input vector
    that retrieval widens.

    That layer is extended, not reinitialised: `extend_input_layer`
    zero-fills the new candidate columns, so on its first step the Phase 2
    model reproduces the finished Phase 1 model bit for bit and only then
    learns to use what retrieval adds. A randomly initialised extension would
    inject noise through 600-plus new inputs into a recurrence tuned without
    them, and the first thing training would have to do is recover the prior
    it was handed for free.

    Args:
        phase1_checkpoint: a Phase 1 `.safetensors` file. Its `manifest.json`
            (same folder, or `<checkpoint>.json` if that is absent) supplies
            the architecture.
        n_candidates: K, how many retrieved poses the Phase 2 input carries.
        device: where to put the returned policy.

    Returns:
        `(policy, phase1_architecture)`.
    """
    import json
    import os

    manifest_path = os.path.join(os.path.dirname(phase1_checkpoint), "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            architecture = (json.load(handle) or {}).get("architecture")
    else:
        architecture = (read_metadata(phase1_checkpoint) or {}).get("architecture")
    if not architecture:
        raise ValueError(
            f"no architecture found for {phase1_checkpoint}. Looked in "
            f"{manifest_path} and the checkpoint's own metadata."
        )
    if architecture.get("use_retrieval"):
        raise ValueError(
            f"{phase1_checkpoint} already has retrieval in its input -- this "
            "is not a Phase 1 checkpoint to extend, it already is a Phase 2 "
            "one. Resume it directly rather than extending it again."
        )

    phase1_policy = build_policy(architecture)
    load_checkpoint(phase1_checkpoint, policy=phase1_policy, device="cpu", strict=True)

    phase2_policy = build_policy({**architecture, "use_retrieval": True, "n_candidates": n_candidates})

    phase2_policy.backbone.load_state_dict(phase1_policy.backbone.state_dict())
    phase2_policy.output_layer.load_state_dict(phase1_policy.output_layer.state_dict())

    weight, bias = extend_input_layer(
        phase1_policy.input_layer.proj.weight,
        phase1_policy.input_layer.proj.bias,
        phase2_policy.spec.input_dim,
    )
    with torch.no_grad():
        phase2_policy.input_layer.proj.weight.copy_(weight)
        phase2_policy.input_layer.proj.bias.copy_(bias)

    return phase2_policy.to(device), architecture


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
    teacher forcing the way Phase 1 does, and no `use_retrieval=False` data to
    hold retrieval back from during an early epoch. `config` is accepted
    as-is so a caller can override any of this; these are only the sane
    starting point for *this* phase.
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
