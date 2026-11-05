"""
Multi-GPU training, kept in one place.

Phase 1 is compute bound, which is the precondition for data parallelism being
worth anything. Measured on a 1650 Ti: step time scales almost exactly linearly
with batch size (474 ms at 8, 3542 ms at 64) and the time per window is flat,
so the GPU is saturated rather than waiting on kernel launches. Splitting the
batch across two GPUs therefore divides real work, and the gradients that have
to be exchanged are only 1,833,625 parameters x 4 bytes = 7.3 MB against a step
of roughly 900 ms -- about a millisecond of communication per second of
compute. That ratio is why the scaling is close to linear.

Why this does not use `torch.nn.parallel.DistributedDataParallel
-------------------------------------------------------------------
DDP installs its gradient reduction around the wrapped module's **`forward`**.
`prepare_for_backward` runs inside that call, and the reducer's bookkeeping
depends on it.

Phase 1's scheduled sampling path does not call `forward`. It calls
`policy.step()` once per frame, 64 times, feeding each output back in as the
next input -- that is the whole point of it. Wrapping the policy in DDP and
then driving it through `step()` bypasses the hook that makes DDP work, and the
failure is silent: training runs, loss decreases, and the two GPUs quietly
optimise two diverging models.

So the reduction is done explicitly in `all_reduce_gradients`, after
`backward()`, where it is correct no matter how the forward pass was driven.
The cost of being explicit is losing DDP's overlap of communication with the
backward pass, which is worth about a millisecond a step here. A silent
correctness trap is not worth a millisecond.

Launching
---------
One process per GPU, which `torchrun` starts and tags via the environment:

    torchrun --nproc_per_node=2 scripts/train_phase1.py --dataset ... --epochs 50

With no such environment this degrades to a single process and every function
below becomes a no-op, so the single GPU path is the same code path.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch

__all__ = ["DistributedContext"]


@dataclass
class DistributedContext:
    """Who this process is, and what it may do.

    Attributes:
        enabled: whether more than one process is taking part.
        rank: global index of this process.
        local_rank: index within this machine, which selects the GPU.
        world_size: how many processes in total.
        device: the device this process owns.
        backend: "nccl" on Linux with CUDA, "gloo" otherwise. NCCL has no
            Windows support, so a local two GPU test on Windows needs gloo.
    """

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    backend: str = "gloo"

    # -- construction -----------------------------------------------------

    @classmethod
    def from_env(cls, device: str | None = None) -> "DistributedContext":
        """Read the launcher's environment, or fall back to one process.

        `torchrun` sets RANK, LOCAL_RANK and WORLD_SIZE. Their absence is the
        normal single GPU case, not an error.
        """
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if device is not None:
            resolved = torch.device(device)
        elif torch.cuda.is_available():
            resolved = torch.device(f"cuda:{local_rank}")
        else:
            resolved = torch.device("cpu")

        if world_size <= 1:
            return cls(enabled=False, device=resolved)

        if resolved.type == "cuda" and sys.platform != "win32":
            backend = "nccl"
        else:
            backend = "gloo"

        context = cls(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=resolved,
            backend=backend,
        )
        context.initialize()
        return context

    def initialize(self) -> None:
        if not self.enabled:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    def shutdown(self) -> None:
        if not self.enabled:
            return
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    # -- roles ------------------------------------------------------------

    @property
    def is_main(self) -> bool:
        """Whether this process writes files, logs and decides when to stop.

        Everything with a side effect outside this process belongs behind this:
        two ranks writing one checkpoint simultaneously corrupts it, and two
        ranks logging produces a duplicated history.
        """
        return self.rank == 0

    def __str__(self) -> str:
        if not self.enabled:
            return f"single process on {self.device}"
        return (
            f"rank {self.rank}/{self.world_size} on {self.device} "
            f"via {self.backend}"
        )

    # -- collectives ------------------------------------------------------

    def barrier(self) -> None:
        if not self.enabled:
            return
        import torch.distributed as dist

        dist.barrier()

    def broadcast_parameters(self, module: torch.nn.Module) -> None:
        """Make every rank start from rank 0's weights.

        Each process builds its own model, and `torch.manual_seed` is not
        enough on its own: any difference in initialisation order would leave
        the ranks averaging gradients computed from *different* models, which
        trains something that is not a model at all. DDP does this
        automatically; doing the reduction by hand means doing this by hand
        too.
        """
        if not self.enabled:
            return
        import torch.distributed as dist

        with torch.no_grad():
            for tensor in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(tensor.data, src=0)

    def all_reduce_gradients(self, module: torch.nn.Module) -> None:
        """Average gradients across ranks, in place.

        Averaged, not summed: each rank computed a mean over its own shard, so
        the mean of those means is the mean over the global batch. Summing
        would scale the gradient by `world_size` and silently multiply the
        effective learning rate.
        """
        if not self.enabled:
            return
        import torch.distributed as dist

        for parameter in module.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad.data, op=dist.ReduceOp.SUM)
                parameter.grad.data /= self.world_size

    def all_reduce_metrics(self, metrics: dict) -> dict:
        """Average a dict of scalars across ranks.

        Without this, a logged loss is whatever rank 0's shard happened to
        contain, which is a different number from the one being optimised.
        """
        if not self.enabled or not metrics:
            return metrics
        import torch.distributed as dist

        keys = sorted(k for k, v in metrics.items() if isinstance(v, (int, float)))
        if not keys:
            return metrics

        values = torch.tensor(
            [float(metrics[k]) for k in keys], dtype=torch.float64, device=self.device
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size

        out = dict(metrics)
        out.update({k: float(v) for k, v in zip(keys, values.tolist())})
        return out

    def broadcast_flag(self, value: bool) -> bool:
        """Share one boolean from rank 0 to all ranks.

        Early stopping is decided on rank 0, from a validation pass only rank 0
        runs. If that decision is not shared, one process leaves the training
        loop and the others block forever in the next collective -- a hang, not
        an error.
        """
        if not self.enabled:
            return value
        import torch.distributed as dist

        flag = torch.tensor([1 if value else 0], dtype=torch.int32, device=self.device)
        dist.broadcast(flag, src=0)
        return bool(flag.item())
