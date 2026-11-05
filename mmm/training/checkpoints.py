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
    "load_backbone",
    "extend_input_layer",
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


# ---------------------------------------------------------------- phase handoff


def load_backbone(path: str, policy, strict: bool = True) -> dict:
    """Load only the backbone from a checkpoint, leaving the adapters alone.

    The backbone is 1,752,320 of 1,989,273 parameters -- 88% of the model --
    and it is the part that holds the motion dynamics. The input and output
    layers are per phase and per skeleton adapters (plan section 15.3): the
    input width changes between phases because the inputs do, and the output
    width changes between skeletons because `pose_dim = n_joints * 6 + 7`.

    So this is the transfer path: train the backbone once, carry it forward,
    and relearn only the 12% that has to change.

    Returns the checkpoint metadata.
    """
    metadata = read_metadata(path)

    if path.endswith(LEGACY_SUFFIX):
        warnings.warn(f"{path} is a legacy pickle checkpoint.", stacklevel=2)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        tensors = {f"{_POLICY_PREFIX}{k}": v for k, v in blob["policy"].items()}
    else:
        tensors = load_file(path, device="cpu")

    prefix = f"{_POLICY_PREFIX}backbone."
    weights = {
        key[len(prefix) :]: value
        for key, value in tensors.items()
        if key.startswith(prefix)
    }
    if not weights:
        raise ValueError(f"{path} contains no backbone tensors")

    saved_d_model = (metadata.get("architecture") or {}).get("d_model")
    current_d_model = getattr(policy, "d_model", None)
    if strict and saved_d_model is not None and current_d_model is not None:
        if saved_d_model != current_d_model:
            raise ValueError(
                f"{path} has d_model={saved_d_model} but this policy has "
                f"{current_d_model}. The backbone cannot transfer across a "
                "width change; pass strict=False only if you know why."
            )

    policy.backbone.load_state_dict(weights, strict=strict)
    return metadata


def extend_input_layer(old_weight, old_bias, new_in_features: int):
    """Widen an input layer's weight, zero-filling the new columns.

    `nn.Linear` holds its weight as `(out_features, in_features)`, so widening
    the input adds *columns*.

    The new columns are zero, not random, and that is the whole point. Phase 1
    trains on `joystick + previous_pose`; Phase 2 adds the retrieved candidates
    on the end. Zero-filling means that on Phase 2's first step the candidate
    block contributes exactly nothing and the model reproduces the finished
    Phase 1 model bit for bit. It then learns to use retrieval from a known
    good starting point rather than from a disturbed one.

    Random initialisation instead would inject noise through 608 new inputs
    into a model whose recurrence was tuned without them, and the first thing
    Phase 2 would have to do is recover the prior it was given.

    This relies on the two input layouts sharing a prefix, which
    `IoSpec.block_offsets` guarantees: joystick and previous_pose occupy
    `0:163` whether or not retrieval is in use.

    Returns:
        `(weight, bias)` for the wider layer. The bias is unchanged, since it
        is per output and the outputs have not changed.
    """
    out_features, old_in = old_weight.shape
    if new_in_features < old_in:
        raise ValueError(
            f"cannot extend {old_in} inputs to {new_in_features}: that is "
            "narrower. Widening is lossless; narrowing is not, and would need "
            "a decision about which columns to drop."
        )

    weight = torch.zeros(
        (out_features, new_in_features), dtype=old_weight.dtype, device=old_weight.device
    )
    weight[:, :old_in] = old_weight
    return weight, old_bias.clone()
