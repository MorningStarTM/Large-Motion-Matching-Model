"""
Learning rate schedules.

A constant learning rate for fifty epochs is not a neutral choice, it is a bad
one at both ends. Early on, the output layer starts near zero and the model is
being fed its own untrained poses through the pose feedback, so a full rate
step can move the weights somewhere the recurrence does not recover from.
Late on, a rate that was right for the first epoch is far too large to settle
into a minimum, and the validation curve flattens into noise rather than
converging -- which also makes early stopping fire on noise instead of on
genuine plateau.

Schedules here are computed per *step*, not per epoch, because an epoch is 424
steps at the default batch size and a per-epoch step function would jump the
rate by a factor at an arbitrary point mid-run.

Written as a plain function of the step rather than a `torch.optim.lr_scheduler`
subclass. The schedule then has no hidden state, so resuming a run is a matter
of knowing which step it reached, and the same code can be asked what the rate
*will* be at any future step -- which `lr_scheduler` cannot do without actually
stepping it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Schedule", "build_schedule", "SCHEDULES"]


@dataclass
class Schedule:
    """A learning rate as a function of the global step.

    Args:
        kind: "cosine", "linear", "constant" or "step".
        base_lr: the rate after warmup.
        total_steps: steps in the whole run, for the decay to span.
        warmup_steps: steps spent ramping linearly from `warmup_start_factor`
            to `base_lr`.
        min_lr_factor: the floor, as a fraction of `base_lr`. Not zero by
            default: a rate that reaches exactly zero stops training a few
            epochs before the run nominally ends, which looks like convergence
            and is not.
        warmup_start_factor: where warmup begins, as a fraction of `base_lr`.
        step_gamma, step_every: for "step" only.
    """

    kind: str = "cosine"
    base_lr: float = 1e-3
    total_steps: int = 1
    warmup_steps: int = 0
    min_lr_factor: float = 0.05
    warmup_start_factor: float = 0.1
    step_gamma: float = 0.5
    step_every: int = 1

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULES:
            raise ValueError(
                f"unknown schedule {self.kind!r}; known: {sorted(SCHEDULES)}"
            )
        if self.total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if self.warmup_steps >= self.total_steps:
            # Otherwise the run is entirely warmup and never decays, which is
            # a silent misconfiguration rather than an error worth raising.
            self.warmup_steps = max(0, self.total_steps - 1)

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear ramp. Starting above zero rather than at it, because a
            # first step at exactly zero wastes a step and makes the first
            # logged rate meaningless.
            progress = (step + 1) / max(self.warmup_steps, 1)
            start = self.warmup_start_factor
            return self.base_lr * (start + (1.0 - start) * progress)

        return SCHEDULES[self.kind](self, step)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_factor": self.min_lr_factor,
            "warmup_start_factor": self.warmup_start_factor,
            "step_gamma": self.step_gamma,
            "step_every": self.step_every,
        }


def _decay_progress(schedule: Schedule, step: int) -> float:
    """How far through the post-warmup part of the run, in `[0, 1]`."""
    span = max(schedule.total_steps - schedule.warmup_steps, 1)
    return min(max((step - schedule.warmup_steps) / span, 0.0), 1.0)


def _cosine(schedule: Schedule, step: int) -> float:
    progress = _decay_progress(schedule, step)
    floor = schedule.min_lr_factor
    factor = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule.base_lr * factor


def _linear(schedule: Schedule, step: int) -> float:
    progress = _decay_progress(schedule, step)
    floor = schedule.min_lr_factor
    return schedule.base_lr * (1.0 - (1.0 - floor) * progress)


def _constant(schedule: Schedule, step: int) -> float:
    return schedule.base_lr


def _step(schedule: Schedule, step: int) -> float:
    taken = (step - schedule.warmup_steps) // max(schedule.step_every, 1)
    factor = schedule.step_gamma**max(taken, 0)
    return schedule.base_lr * max(factor, schedule.min_lr_factor)


SCHEDULES = {
    "cosine": _cosine,
    "linear": _linear,
    "constant": _constant,
    "step": _step,
}


def build_schedule(
    kind: str,
    base_lr: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float = 1.0,
    **extra,
) -> Schedule:
    """Build a schedule from epoch counts, which is how runs are configured.

    `steps_per_epoch` comes from `len(train_loader)`, so a schedule always
    spans the actual run rather than an assumed one.
    """
    steps_per_epoch = max(int(steps_per_epoch), 1)
    return Schedule(
        kind=kind,
        base_lr=base_lr,
        total_steps=max(epochs, 1) * steps_per_epoch,
        warmup_steps=int(round(warmup_epochs * steps_per_epoch)),
        **extra,
    )
