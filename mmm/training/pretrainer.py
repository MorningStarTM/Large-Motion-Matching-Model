"""
`PreTrainer`: the Phase 1 trainer. Learns how bodies move.

Task (plan section 5.2): next pose prediction over long sequences. The joystick
signal is taken from the data's own real future root path, and the retrieved
candidates are included as input from the start, so the model learns to use
retrieval rather than learning to work without it and having it bolted on later.

Three mechanisms force that, and all three are on by default:

1. **Leave one clip out** -- a frame's candidates never come from its own clip,
   applied in `WindowDataset` where the retrieval happens.
2. **Retrieval dropout** -- whole sequences occasionally arrive with the
   candidates blanked, applied in `MmmPolicy.build_input`.
3. **Held out styles** -- whole 100STYLE styles excluded from training, applied
   in the split.

Without them the model can memorise and ignore retrieval, which plan section 11
names as a risk and which would quietly invalidate the main claim. So the
validation pass here also evaluates *with the candidates zeroed* and reports the
gap: `retrieval_gain` is how much worse the model does without retrieval, and it
is tracked every epoch rather than discovered at the end.

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
from mmm.models.heads import ResidualHead
from mmm.models.policy import MmmPolicy
from mmm.training.checkpoints import load_checkpoint, save_checkpoint
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
    # but it is the only regime in which retrieval can show its value at all
    # (see `validate`), so it is measured on a fixed subset every epoch rather
    # than skipped. 0 disables it.
    rollout_eval_batches: int = 8
    loss_weights: LossWeights = field(default_factory=LossWeights)
    calibrate_weights: bool = True

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
    ):
        self.config = config or PreTrainerConfig()
        self.database = database
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = policy.to(self.device)
        self.loss_fn = PoseLoss(
            skeleton=database.skeleton,
            frame_time=database.frame_time,
            weights=self.config.loss_weights,
            n_joints=database.skeleton.n_joints,
        )
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history: dict[str, list] = {}
        self._calibrated = False

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
        candidates = batch["candidates"]
        mask = batch["mask"]

        if sampling_probability <= 0.0:
            pose, contacts, _ = self.policy(
                joystick,
                previous,
                candidates,
                mask,
                apply_retrieval_dropout=training,
            )
        else:
            batch_size, seq_len = joystick.shape[0], joystick.shape[1]
            state = self.policy.initial_state(batch_size, device=joystick.device)
            fed = previous[:, 0]
            poses, contact_list = [], []
            for t in range(seq_len):
                pose_t, contacts_t, state = self.policy.step(
                    joystick[:, t], fed, candidates[:, t], mask[:, t], state
                )
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

        correction = None
        if isinstance(self.policy.head, ResidualHead):
            correction = ResidualHead.correction(pose, candidates, mask)

        return pose, contacts, correction

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

        for batch in loader:
            batch = self._batch_to_device(batch)

            if self.config.calibrate_weights and not self._calibrated:
                self._calibrate(batch)

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
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            self.optimizer.step()

            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + value
            n_batches += 1

        result = {name: value / max(n_batches, 1) for name, value in totals.items()}
        result["sampling_probability"] = probability
        return result

    @torch.no_grad()
    def validate(self, loader) -> dict:
        """Validation, plus the retrieval ablation.

        Reports two retrieval gains, and only the second one means anything:

        - `retrieval_gain`, teacher forced. Near zero by construction, because
          the true previous pose is supplied every frame and is a far better
          predictor than any retrieved candidate.
        - `rollout_retrieval_gain`, free running. The model is fed its own
          output and must rely on retrieval to stay anchored. See
          `_rollout_retrieval_gain`.
        """
        self.policy.eval()
        totals: dict[str, float] = {}
        ablated_total = 0.0
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

            blanked = dict(batch)
            blanked["candidates"] = torch.zeros_like(batch["candidates"])
            blanked["mask"] = torch.zeros_like(batch["mask"])
            ablated_pose, ablated_contacts, _ = self._rollout(blanked, 0.0, training=False)
            _, ablated_terms = self.loss_fn(
                ablated_pose,
                batch["target"],
                predicted_contacts=ablated_contacts,
                target_contacts=batch.get("contacts"),
            )
            ablated_total += ablated_terms["total"]
            n_batches += 1

        result = {name: value / max(n_batches, 1) for name, value in totals.items()}
        result["total_without_retrieval"] = ablated_total / max(n_batches, 1)
        result["retrieval_gain"] = result["total_without_retrieval"] - result["total"]
        result.update(self._rollout_retrieval_gain(loader))
        return result

    def _rollout_retrieval_gain(self, loader) -> dict:
        """Retrieval gain under *free running* rollout, not teacher forcing.

        This is the number the editability claim actually rests on, and the
        teacher forced one above cannot show it.

        Under teacher forcing the model is handed the true previous pose every
        frame, and at 60Hz that pose is an excellent predictor of the next one:
        measured on this database, simply copying it scores 2.29 while copying
        the nearest retrieved candidate scores 16.24, seven times worse,
        because leave one clip out forces the match to come from a different
        subject at a different point in its stride. So a model under teacher
        forcing is *right* to ignore the candidates, and a retrieval gain near
        zero there says nothing about whether retrieval is useful.

        Free running is the regime the model is actually deployed in. Its own
        output is fed back, error accumulates, and the retrieved candidates are
        the only thing in its input still anchored to real motion. If retrieval
        does not help here, it does not help anywhere, and the Phase 1 claim
        has failed in a way no amount of validation loss would reveal.
        """
        limit = self.config.rollout_eval_batches
        if not limit:
            return {}

        self.policy.eval()
        with_retrieval = 0.0
        without = 0.0
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
            with_retrieval += terms["total"]

            blanked = dict(batch)
            blanked["candidates"] = torch.zeros_like(batch["candidates"])
            blanked["mask"] = torch.zeros_like(batch["mask"])
            ablated_pose, ablated_contacts, _ = self._rollout(blanked, 1.0, training=False)
            _, ablated_terms = self.loss_fn(
                ablated_pose,
                batch["target"],
                predicted_contacts=ablated_contacts,
                target_contacts=batch.get("contacts"),
            )
            without += ablated_terms["total"]
            n += 1

        if not n:
            return {}
        return {
            "rollout_total": with_retrieval / n,
            "rollout_total_without_retrieval": without / n,
            "rollout_retrieval_gain": (without - with_retrieval) / n,
        }

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
            contact=self.config.loss_weights.contact,
            residual=self.config.loss_weights.residual,
        )
        self._calibrated = True

    # -- the loop ---------------------------------------------------------

    def fit(self, train_loader, val_loader=None, on_epoch_end=None) -> dict:
        """Train, with early stopping on validation loss.

        Args:
            train_loader: yields the dicts `WindowDataset` produces, collated.
            val_loader: optional, same format.
            on_epoch_end: optional `callable(epoch, train_metrics, val_metrics)`.

        Returns:
            The history dict, also written to the checkpoint directory.
        """
        best_metric = math.inf
        best_epoch = 0
        bad_epochs = 0
        self._write_manifest()

        for epoch in range(1, self.config.epochs + 1):
            started = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader) if val_loader is not None else None

            self._record("train", train_metrics)
            if val_metrics is not None:
                self._record("val", val_metrics)
            self._record("meta", {"epoch_seconds": time.time() - started})

            metric = val_metrics["total"] if val_metrics else train_metrics["total"]
            if metric < best_metric - 1e-9:
                best_metric, best_epoch, bad_epochs = metric, epoch, 0
                self.save(os.path.join(self.checkpoint_dir, "best.safetensors"), epoch=epoch)
            else:
                bad_epochs += 1

            if epoch % self.config.log_every == 0:
                self._log(epoch, train_metrics, val_metrics, time.time() - started)

            if on_epoch_end is not None:
                on_epoch_end(epoch, train_metrics, val_metrics)

            if bad_epochs >= self.config.patience:
                print(
                    f"[early stop] no improvement for {self.config.patience} epochs; "
                    f"best epoch {best_epoch} at {best_metric:.6f}"
                )
                break

        self.save(os.path.join(self.checkpoint_dir, "last.safetensors"), epoch=epoch)
        with open(os.path.join(self.checkpoint_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)
        return self.history

    def _record(self, prefix: str, metrics: dict) -> None:
        for name, value in metrics.items():
            self.history.setdefault(f"{prefix}_{name}", []).append(value)

    def _log(self, epoch, train_metrics, val_metrics, seconds) -> None:
        line = (
            f"epoch {epoch:3d}/{self.config.epochs} "
            f"| train {train_metrics['total']:.5f} "
            f"(rot {train_metrics['rotation']:.4f} pos {train_metrics['position']:.4f} "
            f"vel {train_metrics['velocity']:.4f} root {train_metrics['root']:.4f}) "
            f"| ss {train_metrics['sampling_probability']:.2f}"
        )
        if val_metrics is not None:
            line += (
                f" | val {val_metrics['total']:.5f}"
                f" | gain tf {val_metrics['retrieval_gain']:+.4f}"
            )
            # The free running gain is the one that means anything; it is
            # printed second so the two are never confused in a log.
            if "rollout_retrieval_gain" in val_metrics:
                line += (
                    f" | rollout {val_metrics['rollout_total']:.3f}"
                    f" gain {val_metrics['rollout_retrieval_gain']:+.4f}"
                )
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
            policy=self.policy,
            optimizer=self.optimizer if optimizer else None,
            metadata={
                "epoch": epoch,
                "phase": 1,
                "architecture": self.policy.describe(),
                "loss_weights": self.loss_fn.weights.as_dict(),
                "config": self.config.as_dict(),
                # Pins the checkpoint to the data it was trained on. A policy
                # is only meaningful alongside the database it retrieves from,
                # because the candidate poses are part of its input.
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
        whose pose width or candidate count no longer matches must fail here
        rather than load with silently reshaped weights.
        """
        blob = load_checkpoint(path, device=self.device)
        if strict and "architecture" in blob:
            saved = blob["architecture"]
            current = self.policy.describe()
            for key in ("pose_dim", "input_dim", "output_dim", "n_candidates", "d_model", "head"):
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
            policy=self.policy,
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
        from mmm.teacher.logger import database_fingerprint

        manifest = {
            "phase": 1,
            "architecture": self.policy.describe(),
            "config": self.config.as_dict(),
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
