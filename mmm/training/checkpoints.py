"""
Checkpoint I/O on safetensors, with no pickle anywhere.

`torch.save` writes a pickle, and unpickling runs arbitrary code in the
process that loads it. For a file that is going to be published to a model hub
and downloaded by other people -- which is where these are headed -- that is
the wrong format regardless of how much one trusts the producer. safetensors is
a flat map of name to tensor plus a string-to-string metadata header: it can be
memory mapped, and loading it cannot execute anything.

The cost is that safetensors holds *only* tensors, so the two parts of a
checkpoint that are not tensors need somewhere to live:

- Everything describing the run -- architecture, config, calibrated loss
  weights, epoch -- is JSON encoded into the metadata header. That keeps it
  inside the same file, and `huggingface_hub` and the hub's own viewer can read
  it without loading the weights.
- The optimizer's `param_groups` is plain Python (floats, bools, `None`), so it
  is JSON encoded too, while its per parameter tensors (`exp_avg`,
  `exp_avg_sq`, `step`) are flattened into the tensor map under
  `optim.state.<index>.<name>`. The alternative -- a second `.pt` beside the
  safetensors file -- would have quietly reintroduced the pickle this module
  exists to avoid.

Loading a legacy `.pt` still works, so checkpoints written before this change
are not orphaned, but it warns and it is the only path here that unpickles.
"""

from __future__ import annotations

import json
import os
import warnings

import torch
from safetensors.torch import load_file, save_file

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "read_metadata",
    "CHECKPOINT_SUFFIX",
    "LEGACY_SUFFIX",
]

CHECKPOINT_SUFFIX = ".safetensors"
LEGACY_SUFFIX = ".pt"

# Bumped if the key scheme below changes incompatibly.
FORMAT_VERSION = "1"

_OPTIM_PREFIX = "optim.state."
_POLICY_PREFIX = "policy."


def _flatten_optimizer(state: dict) -> tuple[dict, str]:
    """Split an optimizer `state_dict` into tensors and a JSON header.

    Returns `(tensors, param_groups_json)`. The `state` keys are parameter
    indices, which JSON would turn into strings, so they are carried in the
    tensor names and parsed back to ints on load.
    """
    tensors: dict[str, torch.Tensor] = {}
    non_tensor: dict[str, dict] = {}

    for index, entry in state.get("state", {}).items():
        for name, value in entry.items():
            if torch.is_tensor(value):
                tensors[f"{_OPTIM_PREFIX}{index}.{name}"] = value.detach().cpu().clone()
            else:
                # `step` is a plain int on older torch versions.
                non_tensor.setdefault(str(index), {})[name] = value

    header = json.dumps(
        {
            "param_groups": state.get("param_groups", []),
            "non_tensor_state": non_tensor,
        }
    )
    return tensors, header


def _rebuild_optimizer(tensors: dict, header: str) -> dict:
    parsed = json.loads(header)
    state: dict[int, dict] = {}

    for key, value in tensors.items():
        if not key.startswith(_OPTIM_PREFIX):
            continue
        index, name = key[len(_OPTIM_PREFIX) :].split(".", 1)
        state.setdefault(int(index), {})[name] = value

    for index, entry in parsed.get("non_tensor_state", {}).items():
        state.setdefault(int(index), {}).update(entry)

    return {"state": state, "param_groups": parsed.get("param_groups", [])}


def save_checkpoint(
    path: str,
    policy,
    optimizer=None,
    metadata: dict | None = None,
) -> str:
    """Write a checkpoint.

    Args:
        path: destination. A `.pt` suffix is replaced with `.safetensors`
            rather than honoured, so an old call site cannot silently keep
            writing pickles.
        policy: the module whose `state_dict` to store.
        optimizer: optional, needed only to resume training. Published
            artefacts leave it out, which also cuts the file to a third:
            AdamW keeps two moment tensors per parameter, so the weights are
            7.96 MB of a 23.89 MB Phase 1 checkpoint.
        metadata: JSON encodable values describing the run. Each is stored as
            its own JSON string in the safetensors header.

    Returns:
        The path actually written.
    """
    if path.endswith(LEGACY_SUFFIX):
        path = path[: -len(LEGACY_SUFFIX)] + CHECKPOINT_SUFFIX
    elif not path.endswith(CHECKPOINT_SUFFIX):
        path = path + CHECKPOINT_SUFFIX

    tensors = {
        f"{_POLICY_PREFIX}{name}": value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }

    header = {key: json.dumps(value) for key, value in (metadata or {}).items()}
    header["format_version"] = FORMAT_VERSION
    header["has_optimizer"] = json.dumps(optimizer is not None)

    if optimizer is not None:
        optimizer_tensors, optimizer_header = _flatten_optimizer(optimizer.state_dict())
        # A name collision would silently drop one of the two.
        clash = set(optimizer_tensors) & set(tensors)
        if clash:
            raise ValueError(f"optimizer and policy tensors collide on {sorted(clash)}")
        tensors.update(optimizer_tensors)
        header["optimizer"] = optimizer_header

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    # Write then rename, so an interrupted save cannot leave a truncated file
    # that a later run would load as if it were complete.
    temporary = path + ".partial"
    save_file(tensors, temporary, metadata=header)
    os.replace(temporary, path)
    return path


def read_metadata(path: str) -> dict:
    """The decoded metadata header, without loading any weights.

    Used to inspect a checkpoint cheaply -- which architecture, which epoch,
    which loss weights -- and by the publishing script to build a model card.
    """
    if path.endswith(LEGACY_SUFFIX):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return {key: value for key, value in blob.items() if key not in ("policy", "optimizer")}

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        raw = handle.metadata() or {}

    out = {}
    for key, value in raw.items():
        try:
            out[key] = json.loads(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def load_checkpoint(
    path: str,
    policy=None,
    optimizer=None,
    device=None,
    strict: bool = True,
) -> dict:
    """Read a checkpoint, optionally loading it into a policy and optimizer.

    Returns the decoded metadata. Accepts a legacy `.pt` file, warning that it
    is being unpickled.
    """
    if path.endswith(LEGACY_SUFFIX):
        warnings.warn(
            f"{path} is a legacy pickle checkpoint. It is being unpickled, "
            "which executes code from the file. Re-save it as safetensors.",
            stacklevel=2,
        )
        blob = torch.load(path, map_location=device or "cpu", weights_only=False)
        if policy is not None:
            policy.load_state_dict(blob["policy"])
        if optimizer is not None and "optimizer" in blob:
            optimizer.load_state_dict(blob["optimizer"])
        return {k: v for k, v in blob.items() if k not in ("policy", "optimizer")}

    tensors = load_file(path, device="cpu")
    metadata = read_metadata(path)

    if policy is not None:
        weights = {
            key[len(_POLICY_PREFIX) :]: value
            for key, value in tensors.items()
            if key.startswith(_POLICY_PREFIX)
        }
        policy.load_state_dict(weights, strict=strict)

    if optimizer is not None:
        if "optimizer" not in metadata:
            raise ValueError(
                f"{path} holds no optimizer state, so training cannot resume "
                "from it. It was most likely saved for publication."
            )
        header = metadata["optimizer"]
        if not isinstance(header, str):
            header = json.dumps(header)
        optimizer.load_state_dict(_rebuild_optimizer(tensors, header))

    return metadata
