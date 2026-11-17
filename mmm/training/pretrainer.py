"""
`PreTrainer`: the Phase 1 trainer. Learns how bodies move.

Task (plan section 5.2): next pose prediction over long sequences, from the
joystick (taken from the data's own real future root path) and the model's
own previous pose. There is no retrieval in this input -- a mechanism that
concatenated raw candidate poses onto it, and the retrieval-dropout/ablation
machinery built around that, used to live here and has been removed.
Retrieval is coming back as an encoder feeding the backbone through
cross-attention instead, which is a different mechanism than this trainer
needs to know about.

Drift is handled by scheduled sampling: teacher forcing first, then feeding the
model's own predictions, on a schedule that ramps over training.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import torch

from mmm.data.database import MotionDatabase
from mmm.models.policy import MmmPolicy
from mmm.training.callbacks import (
    Callback,
    CallbackList,
    CSVLogger,
    EarlyStopping,
    LearningRateLogger,
    ProgressLogger,
    StallWatchdog,
    TensorBoard,
)
from mmm.training.checkpoints import load_checkpoint, save_checkpoint
from mmm.training.distributed import DistributedContext
from mmm.training.schedules import build_schedule
from mmm.training.losses import LossWeights, PoseLoss

__all__ = ["PreTrainerConfig", "PreTrainer"]


@dataclass
class PreTrainerConfig:
    """Phase 1 settings.

    Attributes:
        epochs: passes over the training windows.
        learning_rate: initial Adam learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: max gradient norm; 0 disables clipping. Autoregressive
            rollouts through an SSM can produce occasional large gradients, so
            this is on by default.
        scheduled_sampling_start: epoch at which the model begins to be fed its
            own predictions.
        scheduled_sampling_end: epoch by which the probability reaches
            `scheduled_sampling_max`.
        scheduled_sampling_max: final probability of using the model's own
            previous output instead of ground truth.
        patience: epochs without validation improvement before stopping.
        device: "cuda", "cpu", or None to pick automatically.
        amp: mixed precision. Off by default: the SSM recurrence accumulates
            over the whole window and is sensitive to fp16 underflow in
            `exp(dt * A)`.
        log_every: epochs between log lines.
    """

    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduled_sampling_start: int = 5
    scheduled_sampling_end: int = 25
    scheduled_sampling_max: float = 0.5
    patience: int = 10
    device: str | None = None
    amp: bool = False
    log_every: int = 1
    # Batches used for the free running rollout evaluation. That evaluation
    # steps frame by frame and so costs far more than the teacher forced pass,
    # but it is the regime the model is actually deployed in, so it is
    # measured on a fixed subset every epoch rather than skipped. 0 disables
    # it.
    rollout_eval_batches: int = 8
    loss_weights: LossWeights = field(default_factory=LossWeights)
    calibrate_weights: bool = True

    # -- learning rate schedule ------------------------------------------
    # "cosine" with a one epoch warmup. Constant is available for an ablation
    # but is a poor default at both ends of a run: see mmm/training/schedules.
    lr_schedule: str = "cosine"
    warmup_epochs: float = 1.0
    min_lr_factor: float = 0.05

    # -- selection and stopping ------------------------------------------
    # Which metric decides `best.safetensors` and early stopping.
    #
    # "val_rollout_total" -- free running -- rather than "val_total", because
    # Phase 1's deliverable is a drift resistant motion prior and the teacher
    # forced loss cannot see drift: it hands the model the true previous pose
    # every frame. Falls back to val_total, then train_total, if absent.
    monitor: str = "val_rollout_total"
    min_delta: float = 0.0
    # Epoch from which `monitor` starts deciding `best` and early stopping.
    # None means `scheduled_sampling_start`.
    #
    # Measured on a 3 epoch run: under pure teacher forcing the free running
    # loss got *worse* every epoch (2.688 -> 3.837 -> 3.643) while the teacher
    # forced loss halved (1.002 -> 0.602 -> 0.521). That is exposure bias, and
    # it is expected -- nothing is training the free running regime until
    # scheduled sampling engages. Letting the monitor act before then picks the
    # least trained epoch as "best" and can early stop the run before the
    # mechanism meant to fix drift has done anything at all.
    monitor_start_epoch: int | None = None

    # -- logging ----------------------------------------------------------
    tensorboard: bool = True
    csv_log: bool = True
    progress_every: int = 25

    # torch.compile the policy. Targets the real bottleneck -- the selective
    # scan materialises a (batch, seq, d_inner, d_state) tensor, 134 MB across
    # four layers, and the elementwise work over it is memory bandwidth bound,
    # which is exactly what fusion removes.
    #
    # Only the teacher forced path benefits: compile wraps `forward`, while the
    # scheduled sampling path drives `step` frame by frame. Untestable on
    # Windows, where torch 2.1's inductor backend needs a triton wheel that
    # does not exist, so a failure to compile is reported and ignored rather
    # than fatal.
    compile_policy: bool = False

    # Kills this process if no training progress (a batch or an epoch
    # boundary) is seen for this many minutes. 0 disables it. Exists for
    # unattended runs where a hung process -- a stuck disk write, a wedged
    # NCCL collective -- otherwise burns billed GPU time with nobody
    # watching until the session's own limit. See
    # `mmm.training.callbacks.StallWatchdog` for why a plain callback cannot
    # do this job. 20 minutes comfortably exceeds every observed per-epoch
    # time on Kaggle so far (tens of seconds for a small GPT backbone, up to
    # ~530s for a 10M parameter Mamba backbone with rollout eval), so it
    # should only ever fire on a genuine stall, not a slow but healthy epoch.
    stall_timeout_minutes: float = 20.0

    @property
    def resolved_monitor_start(self) -> int:
        if self.monitor_start_epoch is not None:
            return self.monitor_start_epoch
        return max(self.scheduled_sampling_start, 1)

    def resolved_monitor(self, logs: dict) -> str:
        """The monitored key, falling back if it was not produced.

        A monitor that silently never appears would disable early stopping and
        make every epoch look like the best one, so the fallback is explicit
        and reported once.
        """
        for candidate in (self.monitor, "val_total", "train_total"):
            if candidate in logs:
                if candidate != self.monitor:
                    print(
                        f"[monitor] {self.monitor!r} not produced; "
                        f"falling back to {candidate!r}"
                    )
                return candidate
        raise ValueError(f"no usable monitor in logs: {sorted(logs)}")

    def as_dict(self) -> dict:
        out = {
            key: getattr(self, key)
            for key in (
                "epochs",
                "learning_rate",
                "weight_decay",
                "grad_clip",
                "scheduled_sampling_start",
                "scheduled_sampling_end",
                "scheduled_sampling_max",
                "patience",
                "device",
                "amp",
                "log_every",
                "calibrate_weights",
                "lr_schedule",
                "warmup_epochs",
                "min_lr_factor",
                "monitor",
                "min_delta",
                "monitor_start_epoch",
                "compile_policy",
                "rollout_eval_batches",
                "stall_timeout_minutes",
            )
        }
        out["loss_weights"] = self.loss_weights.as_dict()
        return out


class PreTrainer:
    """Trains an `MmmPolicy` on next pose prediction.

    Args:
        policy: the model.
        database: the database the windows came from, for the skeleton and
            frame time the loss needs.
        config: training settings.
        checkpoint_dir: where checkpoints and the run manifest go.
    """

    def __init__(
        self,
        policy: MmmPolicy,
        database: MotionDatabase,
        config: PreTrainerConfig | None = None,
        checkpoint_dir: str = "runs/phase1",
        distributed: DistributedContext | None = None,
        phase: int = 1,
    ):
        self.config = config or PreTrainerConfig()
        self.database = database
        self.checkpoint_dir = checkpoint_dir
        # Recorded in the checkpoint and the manifest. The training loop below
        # is phase-agnostic next-pose prediction either way -- Phase 2 reuses
        # this class wholesale, pointed at a dataset built from teacher
        # rollouts instead of raw mocap -- so the one thing that has to be
        # told apart is which phase produced the checkpoint, for anyone
        # reading it later.
        self.phase = phase
        self._watchdog = None  # set up in fit(); None outside it is a no-op

        # A single process context behaves as a no-op throughout, so the single
        # GPU path is the same code path rather than a second one.
        self.distributed = distributed or DistributedContext()
        self.device = self.distributed.device

        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        elif not self.distributed.enabled:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.distributed.is_main:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.policy = policy.to(self.device)
        # Every rank must start from identical weights, or they average
        # gradients computed from different models.
        self.distributed.broadcast_parameters(self.policy)

        # `self.module` is always the real policy: `save`, `load` and
        # `describe` must never see a compiled or wrapped object, since its
        # state dict keys are prefixed and would not load back.
        self.module = self.policy
        if self.config.compile_policy:
            self.policy = self._compile(self.policy)
        self.loss_fn = PoseLoss(
            skeleton=database.skeleton,
            frame_time=database.frame_time,
            weights=self.config.loss_weights,
            n_joints=database.skeleton.n_joints,
        )
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history: dict[str, list] = {}
        self._calibrated = False
        self.start_epoch = 1
        self.global_step = 0
        self.schedule = None
        self.steps_per_epoch = 0
        self.callbacks = None

    def _compile(self, policy):
        try:
            compiled = torch.compile(policy)
            if self.distributed.is_main:
                print("[compile] torch.compile enabled for the teacher forced path")
            return compiled
        except Exception as error:  # noqa: BLE001
            if self.distributed.is_main:
                print(f"[compile] unavailable ({type(error).__name__}: {error}); continuing uncompiled")
            return policy

    def _print(self, *args, **kwargs) -> None:
        """Print on the main rank only, so N processes do not log N times."""
        if self.distributed.is_main:
            print(*args, **kwargs)

    # -- schedules --------------------------------------------------------

    def scheduled_sampling_probability(self, epoch: int) -> float:
        """Probability of feeding the model its own previous output.

        Zero until `scheduled_sampling_start`, then ramping linearly to
        `scheduled_sampling_max`. Starting at zero matters: a model fed its own
        untrained output from epoch 0 never learns the one-step mapping in the
        first place, and an SSM carrying a hidden state compounds that.
        """
        config = self.config
        if epoch < config.scheduled_sampling_start:
            return 0.0
        if epoch >= config.scheduled_sampling_end:
            return config.scheduled_sampling_max
        span = max(1, config.scheduled_sampling_end - config.scheduled_sampling_start)
        progress = (epoch - config.scheduled_sampling_start) / span
        return config.scheduled_sampling_max * progress

    # -- one window -------------------------------------------------------

    def _rollout(self, batch: dict, sampling_probability: float, training: bool):
        """Run one window, optionally feeding the model its own predictions.

        With `sampling_probability == 0` the whole window goes through the
        backbone in one parallel pass, which is much faster. Above zero the
        window has to be stepped frame by frame, because frame `t`'s input
        depends on frame `t-1`'s output.

        Returns:
            `(predicted, contacts, correction)`; `correction` is None unless
            the residual head is in use.
        """
        joystick = batch["joystick"]
        previous = batch["previous_pose"]

        if sampling_probability <= 0.0:
            pose, contacts, _ = self.policy(joystick, previous)
        else:
            batch_size, seq_len = joystick.shape[0], joystick.shape[1]
            state = self.module.initial_state(batch_size, device=joystick.device)
            fed = previous[:, 0]
            poses, contact_list = [], []
            for t in range(seq_len):
                pose_t, contacts_t, state = self.module.step(joystick[:, t], fed, state)
                poses.append(pose_t)
                if contacts_t is not None:
                    contact_list.append(contacts_t)

                if t + 1 < seq_len:
                    use_own = (
                        torch.rand(batch_size, 1, device=joystick.device) < sampling_probability
                    ).to(pose_t.dtype)
                    # Detached: scheduled sampling is about the input
                    # distribution, not about backpropagating through the whole
                    # rollout. Keeping the graph here would make the window
                    # cost grow quadratically and is Phase 3's job, not Phase 1's.
                    fed = use_own * pose_t.detach() + (1.0 - use_own) * previous[:, t + 1]
            pose = torch.stack(poses, dim=1)
            contacts = torch.stack(contact_list, dim=1) if contact_list else None

        return pose, contacts, None

    def _batch_to_device(self, batch: dict) -> dict:
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    # -- epochs -----------------------------------------------------------

    def train_epoch(self, loader, epoch: int) -> dict:
        self.policy.train()
        probability = self.scheduled_sampling_probability(epoch)
        totals: dict[str, float] = {}
        n_batches = 0

        grad_norm_total = 0.0

        for index, batch in enumerate(loader):
            batch = self._batch_to_device(batch)

            if self.config.calibrate_weights and not self._calibrated:
                self._calibrate(batch)

            # Set the rate per step, not per epoch: an epoch is hundreds of
            # steps, so a per-epoch schedule would jump the rate at an
            # arbitrary point inside one.
            lr = self._apply_lr()

            pose, contacts, correction = self._rollout(batch, probability, training=True)
            loss, terms = self.loss_fn(
                pose,
                batch["target"],
                predicted_contacts=contacts,
                target_contacts=batch.get("contacts"),
                correction=correction,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Average gradients across ranks here rather than through DDP: the
            # scheduled sampling path drives `step` frame by frame and never
            # calls the wrapped `forward` DDP hooks itself onto. See
            # mmm/training/distributed.py.
            self.distributed.all_reduce_gradients(self.module)

            # The norm before clipping, which is the diagnostic worth having:
            # after clipping it is just `grad_clip` whenever anything is wrong.
            if self.config.grad_clip > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.module.parameters(), self.config.grad_clip
                    )
                )
            else:
                grad_norm = 0.0
            grad_norm_total += grad_norm

            self.optimizer.step()
            self.global_step += 1

            # Every rank, unlike the logging callbacks below: a rank with no
            # validation or checkpointing to do still needs its own proof of
            # life, since it is the one left waiting (and billing GPU time)
            # if the stall is on another rank.
            if self._watchdog is not None:
                self._watchdog.beat()

            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + value
            n_batches += 1

            if self.callbacks is not None and self.distributed.is_main:
                self.callbacks.on_batch_end(
                    index, {**terms, "lr": lr, "grad_norm": grad_norm}
                )

        result = {name: value / max(n_batches, 1) for name, value in totals.items()}
        result["grad_norm"] = grad_norm_total / max(n_batches, 1)
        # Averaged over ranks, or a logged loss is whatever rank 0's shard
        # happened to hold rather than the quantity being optimised.
        result = self.distributed.all_reduce_metrics(result)
        result["sampling_probability"] = probability
        result["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return result

    def _apply_lr(self) -> float:
        """Write the scheduled rate onto the optimizer, and return it.

        No schedule means the configured rate, untouched -- so calling
        `train_epoch` directly, as the tests do, needs no schedule.
        """
        if self.schedule is None:
            return float(self.optimizer.param_groups[0]["lr"])
        lr = self.schedule(self.global_step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    @torch.no_grad()
    def validate(self, loader) -> dict:
        """Teacher forced validation, plus a free running rollout check."""
        self.policy.eval()
        totals: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            batch = self._batch_to_device(batch)

            pose, contacts, correction = self._rollout(batch, 0.0, training=False)
            _, terms = self.loss_fn(
                pose,
                batch["target"],
                predicted_contacts=contacts,
                target_contacts=batch.get("contacts"),
                correction=correction,
            )
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + value
            n_batches += 1

        result = {name: value / max(n_batches, 1) for name, value in totals.items()}
        result.update(self._rollout_eval(loader))
        return result

    def _rollout_eval(self, loader) -> dict:
        """Loss under *free running* rollout, not teacher forcing.

        Teacher forcing hands the model the true previous pose every frame,
        which at 60Hz is an excellent predictor on its own. Free running is
        the regime the model is actually deployed in: its own output is fed
        back and error accumulates, so this is the number that actually
        reflects how the model holds up over a real session.
        """
        limit = self.config.rollout_eval_batches
        if not limit:
            return {}

        self.policy.eval()
        total = 0.0
        n = 0
        for batch in loader:
            if n >= limit:
                break
            batch = self._batch_to_device(batch)

            pose, contacts, correction = self._rollout(batch, 1.0, training=False)
            _, terms = self.loss_fn(
                pose,
                batch["target"],
                predicted_contacts=contacts,
                target_contacts=batch.get("contacts"),
                correction=correction,
            )
            total += terms["total"]
            n += 1

        if not n:
            return {}
        return {"rollout_total": total / n}

    def _calibrate(self, batch: dict) -> None:
        """Set the loss weights so each term has roughly equal magnitude.

        Done once, on the first batch, using the model's untrained output.
        Plan section 6.2 asks for this explicitly, and the terms are in
        different units, so measuring beats guessing.
        """
        pose, _, _ = self._rollout(batch, 0.0, training=False)
        suggested = self.loss_fn.calibrate(pose.detach(), batch["target"])
        self.loss_fn.weights = LossWeights(
            rotation=suggested["rotation"],
            position=suggested["position"],
            velocity=suggested["velocity"],
            root=suggested["root"],
            hips_height=suggested["hips_height"],
            contact=self.config.loss_weights.contact,
            residual=self.config.loss_weights.residual,
        )
        self._calibrated = True

        if self.distributed.is_main:
            weights = self.loss_fn.weights.as_dict()
            print("calibrated loss weights:")
            for name, value in weights.items():
                print(f"    {name:12s} {value:10.4f}")
            # Rewritten now that the weights are known. The first write
            # happened before training began, when they were still the
            # placeholder defaults.
            self._write_manifest()

    # -- the loop ---------------------------------------------------------

    def fit(
        self, train_loader, val_loader=None, on_epoch_end=None, callbacks=None
    ) -> dict:
        """Train, with early stopping on validation loss.

        Args:
            train_loader: yields the dicts `WindowDataset` produces, collated.
            val_loader: optional, same format.
            on_epoch_end: optional `callable(epoch, train_metrics, val_metrics)`.

        Returns:
            The history dict, also written to the checkpoint directory.
        """
        config = self.config
        self._write_manifest()

        # The schedule spans the actual run, so it is built here where the
        # loader length is known rather than guessed in the config.
        self.steps_per_epoch = len(train_loader)
        self.schedule = build_schedule(
            kind=config.lr_schedule,
            base_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=config.warmup_epochs,
            min_lr_factor=config.min_lr_factor,
        )

        stopper = EarlyStopping(
            monitor=config.monitor,
            patience=config.patience,
            min_delta=config.min_delta,
            start_epoch=config.resolved_monitor_start,
        )
        built = [LearningRateLogger(), stopper]
        if config.csv_log:
            built.append(CSVLogger(os.path.join(self.checkpoint_dir, "metrics.csv")))
        if config.tensorboard:
            built.append(TensorBoard(os.path.join(self.checkpoint_dir, "tb")))
        if config.progress_every:
            built.append(
                ProgressLogger(every=config.progress_every, total=self.steps_per_epoch)
            )
        if on_epoch_end is not None:
            # The old single-callable hook, kept working as a callback.
            built.append(_LegacyEpochHook(on_epoch_end))
        built.extend(callbacks or [])

        # Only rank 0 logs and writes files; the others would duplicate every
        # row and race on every handle.
        self.callbacks = CallbackList(
            built if self.distributed.is_main else [], trainer=self
        )
        self.callbacks.on_train_begin(
            {"config": config.as_dict(), "architecture": self.module.describe()}
        )

        # Every rank, not just rank 0: `self.callbacks` above is empty on the
        # others, but a rank stuck waiting at the next collective is itself
        # idle GPU time nobody is watching either.
        self._watchdog = (
            StallWatchdog(timeout=config.stall_timeout_minutes * 60)
            if config.stall_timeout_minutes > 0
            else None
        )
        if self._watchdog is not None:
            self._watchdog.start()

        best_metric = math.inf
        best_epoch = 0
        monitor_key = None
        epoch = self.start_epoch

        try:
            for epoch in range(self.start_epoch, config.epochs + 1):
                started = time.time()
                if self._watchdog is not None:
                    self._watchdog.beat()
                self.callbacks.on_epoch_begin(epoch)

                # A DistributedSampler reshuffles only when told which epoch it
                # is; without this every epoch sees the same shard order and
                # the shuffling silently does nothing.
                sampler = getattr(train_loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

                train_metrics = self.train_epoch(train_loader, epoch)

                # Validation on rank 0 only. It is not sharded, so every rank
                # would compute the identical thing, and the free running
                # rollout is the most expensive part of an epoch.
                val_metrics = None
                if val_loader is not None and self.distributed.is_main:
                    val_metrics = self.validate(val_loader)

                self._record("train", train_metrics)
                if val_metrics is not None:
                    self._record("val", val_metrics)
                self._record("meta", {"epoch_seconds": time.time() - started})

                logs = {f"train_{k}": v for k, v in train_metrics.items()}
                if val_metrics is not None:
                    logs.update({f"val_{k}": v for k, v in val_metrics.items()})
                logs["epoch_seconds"] = time.time() - started

                # Monitor resolution, best tracking and checkpointing are all
                # rank 0's business. Only rank 0 validates, so on other ranks
                # `logs` carries no val_* key at all and resolving a monitor
                # there would fall back to a different metric and print a
                # warning about a misconfiguration that does not exist.
                logs["best"] = 0.0
                counts = epoch >= config.resolved_monitor_start

                if self.distributed.is_main:
                    if monitor_key is None:
                        monitor_key = config.resolved_monitor(logs)
                    metric = float(logs[monitor_key])

                    if (counts and metric < best_metric - 1e-9) or best_epoch == 0:
                        # `best_epoch == 0` keeps a usable best.safetensors
                        # from epoch 1 even while the monitor is still warming
                        # up, so a run killed early is not left without one.
                        if counts:
                            best_metric, best_epoch = metric, epoch
                        else:
                            best_epoch = epoch
                        self.save(
                            os.path.join(self.checkpoint_dir, "best.safetensors"),
                            epoch=epoch,
                        )
                        logs["best"] = 1.0

                if self.distributed.is_main and not counts:
                    # `EarlyStopping.start_epoch` does the withholding, so the
                    # key stays in `logs` for the loggers to record.
                    if epoch == self.start_epoch:
                        print(
                            f"[monitor] {monitor_key} starts deciding best and "
                            f"early stopping at epoch {config.resolved_monitor_start}, "
                            "when scheduled sampling engages"
                        )

                # `last` is written every epoch, not only at the end, so a run
                # killed at hour five can still be resumed from hour five.
                if self.distributed.is_main:
                    self.save(
                        os.path.join(self.checkpoint_dir, "last.safetensors"), epoch=epoch
                    )

                self.callbacks.on_epoch_end(epoch, logs)

                if epoch % config.log_every == 0:
                    self._log(epoch, train_metrics, val_metrics, time.time() - started)

                if self.distributed.is_main:
                    self._write_history()

                # Rank 0's validation, rollout eval and checkpoint saves above
                # are done; beat again before the collective below so a slow
                # but healthy one of those does not eat into next epoch's
                # training-batch budget before the watchdog notices progress.
                if self._watchdog is not None:
                    self._watchdog.beat()

                # The stop decision is made on rank 0 from a validation pass
                # only rank 0 ran. Sharing it is not optional: one process
                # leaving the loop while the others wait in the next collective
                # is a hang, not an error.
                if self.distributed.broadcast_flag(self.callbacks.stop_training):
                    break
        finally:
            self.callbacks.on_train_end({"best_epoch": best_epoch, "best": best_metric})
            if self.distributed.is_main:
                self._write_history()
            # Stopped here, not left running: a slow publish step after fit()
            # returns (uploading a checkpoint) is not a stall.
            if self._watchdog is not None:
                self._watchdog.stop()
            # Nobody tears down until everybody has finished the loop.
            self.distributed.barrier()

        if not best_epoch:
            self._print("no epoch completed")
        elif math.isinf(best_metric):
            # The run ended before `monitor_start_epoch`, so nothing was ever
            # compared. The checkpoint is real; the metric is not.
            self._print(
                f"best.safetensors is epoch {best_epoch}; {monitor_key} never "
                f"started deciding (it does from epoch "
                f"{config.resolved_monitor_start})"
            )
        else:
            self._print(f"best {monitor_key} = {best_metric:.6f} at epoch {best_epoch}")
        return self.history

    def _write_history(self) -> None:
        path = os.path.join(self.checkpoint_dir, "history.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

    def resume(self, path: str) -> int:
        """Continue a run from a checkpoint, returning the next epoch.

        Restores weights, optimizer moments and the calibrated loss weights, so
        the resumed run optimises the same objective it stopped on.
        """
        metadata = self.load(path)
        finished = int(metadata.get("epoch") or 0)
        self.start_epoch = finished + 1
        self._print(f"resumed {path} at epoch {finished}; continuing from {self.start_epoch}")
        return self.start_epoch

    def _record(self, prefix: str, metrics: dict) -> None:
        for name, value in metrics.items():
            self.history.setdefault(f"{prefix}_{name}", []).append(value)

    def _log(self, epoch, train_metrics, val_metrics, seconds) -> None:
        if not self.distributed.is_main:
            return
        line = (
            f"epoch {epoch:3d}/{self.config.epochs} "
            f"| train {train_metrics['total']:.5f} "
            f"(rot {train_metrics['rotation']:.4f} pos {train_metrics['position']:.4f} "
            f"vel {train_metrics['velocity']:.4f} root {train_metrics['root']:.4f}) "
            f"| ss {train_metrics['sampling_probability']:.2f}"
        )
        if val_metrics is not None:
            line += f" | val {val_metrics['total']:.5f}"
            if "rollout_total" in val_metrics:
                line += f" | rollout {val_metrics['rollout_total']:.3f}"
        line += f" | {seconds:.1f}s"
        print(line)

    # -- persistence ------------------------------------------------------

    def save(self, path: str, epoch: int | None = None, optimizer: bool = True) -> str:
        """Write a safetensors checkpoint.

        Args:
            path: destination. A `.pt` suffix is rewritten to `.safetensors`.
            epoch: recorded in the metadata.
            optimizer: include optimizer state. Needed to resume, so it is on
                by default, but it is two thirds of the file and useless to
                anyone running inference, so publishing turns it off.

        Returns:
            The path written.
        """
        from mmm.teacher.logger import database_fingerprint

        return save_checkpoint(
            path,
            policy=self.module,
            optimizer=self.optimizer if optimizer else None,
            metadata={
                "epoch": epoch,
                "phase": self.phase,
                "architecture": self.module.describe(),
                "loss_weights": self.loss_fn.weights.as_dict(),
                "config": self.config.as_dict(),
                # Pins the checkpoint to the data it was trained on, for
                # provenance.
                "database": {
                    "fingerprint": database_fingerprint(self.database),
                    "n_rows": self.database.n_rows,
                    "pose_dim": self.database.pose_dim,
                    "frame_time": self.database.frame_time,
                    "clips": [c.name for c in self.database.clips],
                },
            },
        )

    def load(self, path: str, strict: bool = True) -> dict:
        """Load a checkpoint, refusing an incompatible architecture.

        The three phases run as separate processes days apart, so a checkpoint
        whose pose width no longer matches must fail here rather than load
        with silently reshaped weights.
        """
        blob = load_checkpoint(path, device=self.device)
        if strict and "architecture" in blob:
            saved = blob["architecture"]
            current = self.module.describe()
            for key in ("pose_dim", "input_dim", "output_dim", "d_model", "head"):
                if saved.get(key) != current.get(key):
                    raise ValueError(
                        f"checkpoint {path} has {key}={saved.get(key)!r} but this "
                        f"policy has {current.get(key)!r}. Build the policy to match, "
                        "or pass strict=False to override."
                    )
        # Re-read, now loading the tensors into place. The first pass only
        # decoded the metadata, so that an architecture mismatch is refused
        # before any weights are touched.
        load_checkpoint(
            path,
            policy=self.module,
            optimizer=self.optimizer if blob.get("has_optimizer", True) else None,
            device=self.device,
        )

        # Restore the calibrated loss weights too. `calibrate` sets them from
        # the first batch's magnitudes, so they are part of the objective, not
        # a hyperparameter: a checkpoint evaluated without them is being scored
        # against a different loss than the one it was trained on. Measured on
        # a 3 epoch run, the same checkpoint scored 0.489 with its own weights
        # and 2.648 with the defaults -- a 5.4x difference that made the
        # training log and any later evaluation silently incomparable, and
        # would have made `best.pt` selection meaningless across a resume.
        if "loss_weights" in blob:
            saved = blob["loss_weights"]
            self.loss_fn.weights = LossWeights(
                **{
                    field: float(saved[field])
                    for field in LossWeights.__dataclass_fields__
                    if field in saved
                }
            )
            self._calibrated = True
        return blob

    def _write_manifest(self) -> None:
        if not self.distributed.is_main:
            return
        from mmm.teacher.logger import database_fingerprint

        manifest = {
            "phase": self.phase,
            "architecture": self.module.describe(),
            "config": self.config.as_dict(),
            # The weights actually in force, which are NOT the ones in
            # `config.loss_weights` once `calibrate_weights` is on: those are
            # only the starting point, and calibration replaces them from a
            # measurement on the first batch.
            #
            # They were recorded nowhere before, which made two runs'
            # `val_total` incomparable with no sign that anything was wrong --
            # the weights for an earlier run had to be recovered afterwards by
            # least squares from its logged per-term history.
            "loss_weights_effective": self.loss_fn.weights.as_dict(),
            "loss_weights_calibrated": bool(self._calibrated),
            "device": str(self.device),
            "database": {
                "fingerprint": database_fingerprint(self.database),
                "n_rows": self.database.n_rows,
                "clips": [c.name for c in self.database.clips],
                "frame_time": self.database.frame_time,
            },
        }
        with open(os.path.join(self.checkpoint_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)


class _LegacyEpochHook(Callback):
    """Adapts `fit(on_epoch_end=...)` to the callback protocol.

    The single callable predates the callback list. Kept working rather than
    removed, since scripts already pass it.
    """

    def __init__(self, hook):
        self.hook = hook

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train = {k[len("train_") :]: v for k, v in logs.items() if k.startswith("train_")}
        validation = {k[len("val_") :]: v for k, v in logs.items() if k.startswith("val_")}
        self.hook(epoch, train, validation or None)
