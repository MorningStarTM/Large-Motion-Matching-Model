"""
Training callbacks, in the Keras sense.

`PreTrainer.fit` previously took a single `on_epoch_end` callable, which is
enough for one thing and awkward for two. These are composable: a list is
passed to `fit`, every callback sees every event, and anything that wants to
stop the run can say so without the trainer knowing which of them did.

The events, and why each exists rather than only having `on_epoch_end`:

- `on_train_begin` / `on_train_end`: opening and closing files, writing the
  run's configuration, flushing a writer. A callback that opens a file in its
  constructor leaks it when a run dies before `fit` is called.
- `on_epoch_begin`: for anything per-epoch that must happen *before* the data
  is touched, such as a schedule that depends on the epoch.
- `on_batch_end`: a six hour run with 424 steps an epoch needs intra-epoch
  feedback. Without it the first sign that a run has diverged is seven minutes
  late.
- `on_epoch_end`: metrics, checkpoints, early stopping.

`stop_training` is a flag a callback sets rather than an exception it raises,
so a stop request still lets every other callback finish its epoch and close
cleanly.

Deliberately kept free of torch: nothing here imports it, so a callback can be
tested without a model.
"""

from __future__ import annotations

import csv
import json
import os
import time

__all__ = [
    "Callback",
    "CallbackList",
    "History",
    "CSVLogger",
    "TensorBoard",
    "ProgressLogger",
    "EarlyStopping",
    "LearningRateLogger",
]


class Callback:
    """Base class. Every hook is a no-op, so a subclass overrides only what it
    needs and gains new events without changing."""

    def set_trainer(self, trainer) -> None:
        self.trainer = trainer

    def on_train_begin(self, logs: dict | None = None) -> None:
        pass

    def on_train_end(self, logs: dict | None = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: dict | None = None) -> None:
        pass


class CallbackList(Callback):
    """Fans every event out to its members.

    A callback that raises must not take the run down with it when the run is
    recoverable -- a TensorBoard writer failing on a full disk should not lose
    six hours of training -- so an exception from a *logging* callback is
    reported and swallowed. `EarlyStopping` and anything else that affects
    control flow is exempt, because silently ignoring a stop request is worse
    than crashing.
    """

    def __init__(self, callbacks=None, trainer=None):
        self.callbacks = list(callbacks or [])
        self.stop_training = False
        if trainer is not None:
            self.set_trainer(trainer)

    def set_trainer(self, trainer) -> None:
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def append(self, callback: Callback) -> None:
        if hasattr(self, "trainer"):
            callback.set_trainer(self.trainer)
        self.callbacks.append(callback)

    def _dispatch(self, event: str, *args, **kwargs) -> None:
        for callback in self.callbacks:
            try:
                getattr(callback, event)(*args, **kwargs)
            except Exception as error:  # noqa: BLE001
                if getattr(callback, "critical", False):
                    raise
                print(
                    f"[callback] {type(callback).__name__}.{event} failed: "
                    f"{type(error).__name__}: {error}"
                )
            if getattr(callback, "stop_training", False):
                self.stop_training = True

    def on_train_begin(self, logs=None):
        self._dispatch("on_train_begin", logs)

    def on_train_end(self, logs=None):
        self._dispatch("on_train_end", logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._dispatch("on_epoch_begin", epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._dispatch("on_epoch_end", epoch, logs)

    def on_batch_end(self, batch, logs=None):
        self._dispatch("on_batch_end", batch, logs)


class History(Callback):
    """Accumulates every logged metric, columnar.

    The trainer keeps its own history for backwards compatibility; this exists
    so a caller can get the same thing without reaching into the trainer.
    """

    def __init__(self):
        self.history: dict[str, list] = {}
        self.epochs: list[int] = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        for name, value in (logs or {}).items():
            self.history.setdefault(name, []).append(value)


class CSVLogger(Callback):
    """One row per epoch, appended as it goes.

    Written incrementally and flushed every epoch, so a run killed at hour
    five still has five hours of metrics. A JSON file dumped at the end does
    not survive that, which is the usual way this data gets lost.
    """

    def __init__(self, path: str):
        self.path = path
        self._handle = None
        self._writer = None
        self._fields: list[str] = []

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._handle = open(self.path, "w", newline="", encoding="utf-8")

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["epoch"] = epoch
        if self._writer is None:
            self._fields = ["epoch"] + sorted(k for k in logs if k != "epoch")
            self._writer = csv.DictWriter(self._handle, fieldnames=self._fields)
            self._writer.writeheader()
        # A metric that appears only from epoch 5 -- the scheduled sampling
        # terms do -- must not break the writer that was built at epoch 1.
        self._writer.writerow({k: logs.get(k, "") for k in self._fields})
        self._handle.flush()

    def on_train_end(self, logs=None):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class TensorBoard(Callback):
    """Scalar logging to TensorBoard, if it is installed.

    Degrades to a no-op with one printed line rather than failing the run: the
    point of a six hour training run is the model, not the graphs.

    Run `tensorboard --logdir runs` to view.
    """

    def __init__(self, log_dir: str, flush_every: int = 1):
        self.log_dir = log_dir
        self.flush_every = flush_every
        self.writer = None
        self._step = 0

    def on_train_begin(self, logs=None):
        # Broad except, not just ImportError. TensorBoard imports fine and then
        # fails on a numpy mismatch -- observed on Kaggle as
        # "module 'numpy.dtypes' has no attribute 'StringDType'", which is a
        # build expecting numpy >= 2 beside numpy 1.x. Either way the answer is
        # the same: say so once and carry on, because metrics.csv and
        # history.json already hold everything.
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.log_dir)
        except Exception as error:  # noqa: BLE001
            print(
                f"[TensorBoard] unavailable ({type(error).__name__}: {error}); "
                "skipping. Metrics still go to history.json and metrics.csv."
            )
            self.writer = None
            return

        if logs:
            # Hyperparameters as text, because `add_hparams` insists on pairing
            # them with final metrics, which do not exist yet at this point.
            try:
                self.writer.add_text(
                    "config", f"```json\n{json.dumps(logs, indent=2)}\n```", 0
                )
            except Exception as error:  # noqa: BLE001
                print(f"[TensorBoard] could not write the config block: {error}")

    def on_batch_end(self, batch, logs=None):
        self._step += 1
        if self.writer is None or not logs:
            return
        for name, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"batch/{name}", value, self._step)

    def on_epoch_end(self, epoch, logs=None):
        if self.writer is None:
            return
        for name, value in (logs or {}).items():
            if isinstance(value, (int, float)):
                # "train/total" rather than "train_total", so TensorBoard
                # groups the terms instead of listing twenty flat scalars.
                self.writer.add_scalar(name.replace("_", "/", 1), value, epoch)
        if epoch % self.flush_every == 0:
            self.writer.flush()

    def on_train_end(self, logs=None):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class ProgressLogger(Callback):
    """Intra-epoch progress, so a diverging run shows up in seconds.

    Prints on one rewritten line, and only when attached to a terminal: in a
    redirected log a carriage return produces one unreadable line of tens of
    thousands of characters.
    """

    def __init__(self, every: int = 25, total: int | None = None):
        self.every = every
        self.total = total
        self._started = 0.0
        self._interactive = None

    def on_train_begin(self, logs=None):
        import sys

        self._interactive = sys.stdout.isatty()

    def on_epoch_begin(self, epoch, logs=None):
        self._started = time.time()
        self._epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if not self._interactive or batch % self.every:
            return
        elapsed = time.time() - self._started
        rate = (batch + 1) / max(elapsed, 1e-9)
        line = f"  epoch {self._epoch}  step {batch + 1}"
        if self.total:
            line += f"/{self.total}"
            remaining = (self.total - batch - 1) / max(rate, 1e-9)
            line += f"  eta {remaining:5.0f}s"
        if logs and "total" in logs:
            line += f"  loss {logs['total']:.4f}"
        if logs and "lr" in logs:
            line += f"  lr {logs['lr']:.2e}"
        print(line.ljust(78), end="\r", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        if self._interactive:
            print(" " * 78, end="\r")


class EarlyStopping(Callback):
    """Stop when a monitored metric stops improving.

    Control flow, so it is `critical`: an exception here must not be
    swallowed, or a run would train past its stopping point in silence.

    Args:
        monitor: metric key, e.g. "val_total".
        patience: epochs without improvement before stopping.
        min_delta: improvement smaller than this does not count. Guards
            against a metric that creeps down by 1e-9 a epoch and defeats
            patience entirely.
        mode: "min" or "max".
        start_epoch: epochs before this are ignored entirely -- neither
            improvement nor patience. For a monitor that measures a regime
            training has not begun optimising yet, counting those epochs picks
            the least trained one as best and can stop the run before the
            mechanism meant to improve the metric has engaged.
    """

    critical = True

    def __init__(
        self,
        monitor: str = "val_total",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        start_epoch: int = 1,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.start_epoch = start_epoch
        self.stop_training = False
        self.best = None
        self.best_epoch = 0
        self.bad_epochs = 0

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_train_begin(self, logs=None):
        self.stop_training = False
        self.best = None
        self.bad_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch < self.start_epoch:
            return
        if self.monitor not in logs:
            # Naming the available keys, because a typo'd monitor otherwise
            # disables early stopping for the whole run without a word.
            print(
                f"[EarlyStopping] {self.monitor!r} not in logs; available: "
                f"{sorted(logs)}"
            )
            return

        value = float(logs[self.monitor])
        if self._improved(value):
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.stop_training = True
            print(
                f"[EarlyStopping] {self.monitor} has not improved for "
                f"{self.patience} epochs; best {self.best:.6f} at epoch "
                f"{self.best_epoch}"
            )


class LearningRateLogger(Callback):
    """Records the optimizer's actual learning rate.

    The *actual* rate read off the optimizer, not the schedule's prediction of
    it: those differ whenever the wiring is wrong, which is exactly the bug
    worth catching.
    """

    def on_epoch_end(self, epoch, logs=None):
        trainer = getattr(self, "trainer", None)
        if trainer is None or logs is None:
            return
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None and optimizer.param_groups:
            logs["lr"] = float(optimizer.param_groups[0]["lr"])
