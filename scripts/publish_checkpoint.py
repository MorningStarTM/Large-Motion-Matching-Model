"""
Publish a Phase 1 checkpoint to a Hugging Face model repository.

Deliberately a separate, manually invoked script rather than something
training does on its own. Pushing to a hub is publishing: it leaves this
machine, it can be indexed, and deleting it later does not reliably unpublish
it. That should be an explicit act with a human behind it, not a side effect of
an epoch finishing.

Usage:
    # see exactly what would be uploaded, touching nothing
    python scripts/publish_checkpoint.py --repo-id you/mmm-phase1 --dry-run

    # upload, private by default
    python scripts/publish_checkpoint.py --repo-id you/mmm-phase1

    # make it public -- read the licence note below first
    python scripts/publish_checkpoint.py --repo-id you/mmm-phase1 --public

Authentication comes from the environment, in the order `huggingface_hub`
itself looks: `HF_TOKEN`, then a token saved by `huggingface-cli login`. No
token is ever read from an argument, so it cannot end up in a shell history.

A note on licensing, because it is easy to get wrong
----------------------------------------------------
These weights are trained on LaFAN1 and the retrieval database is built from
it. Publishing a model trained on a dataset does not relicense that dataset,
and LaFAN1 carries Ubisoft's own terms. Check them before passing `--public`.
The default here is private for that reason, not out of caution about the code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmm.training.checkpoints import read_metadata  # noqa: E402

CARD_TEMPLATE = """---
library_name: mmm
tags:
- motion-matching
- character-animation
- state-space-model
- mamba
{license_line}---

# {title}

State space model for real time character animation: a Mamba backbone that
takes a joystick request and its own previous pose, and predicts the next
pose. There is no retrieval in the input.

This is a **Phase {phase}** checkpoint ({phase_name}).

The database this checkpoint was trained against:

| | |
| --- | --- |
| fingerprint | `{db_fingerprint}` |
| rows (frames) | {db_rows} |
| pose width | {db_pose_dim} |
| frame time | {db_frame_time} s |
| clips | {db_clips} |

The database is built from LaFAN1 and is **not** distributed here. Rebuild it
with `scripts/build_databases.py`; the fingerprint above must match, or the
retrieved poses are in a different space from the ones the model learned on.

## Architecture

| | |
| --- | --- |
{architecture_rows}

## Training

{metrics_section}

## Provenance

- Rotation maths cross validated against GenoViewPython's `quat.py`
  (MIT, Copyright (c) 2024 Daniel Holden) to 1e-9.
- Spring damper after Holden, *Spring-It-On*; inertialization after Bollo,
  GDC 2018; 6D rotations after Zhou et al., CVPR 2019.
- Learned Motion Matching (Holden et al., SIGGRAPH 2020) is background and
  baseline for this work, not its architecture.

See `NOTICE.md` in the source repository for the full attribution.

## Format

`safetensors`, with the run's architecture, config and calibrated loss weights
JSON encoded in the metadata header. No pickle is involved in loading.

```python
from safetensors import safe_open

with safe_open("{filename}", framework="pt") as f:
    print(f.metadata()["architecture"])
```
"""

PHASE_NAMES = {
    0: "classic KNN teacher, no learning",
    1: "pre-training on mocap",
    2: "distillation from the KNN teacher",
    3: "reinforcement learning polish",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="e.g. yourname/mmm-phase1")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("runs", "phase1", "best.safetensors"),
        help="the checkpoint to publish",
    )
    parser.add_argument(
        "--history",
        default="",
        help="history.json to summarise in the card. Defaults to one beside "
        "the checkpoint.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="create the repository public. Default is private; read the "
        "licence note in this file's docstring first.",
    )
    parser.add_argument(
        "--license",
        default="",
        help="SPDX id for the card's metadata, e.g. 'mit'. Left blank "
        "deliberately: the weights derive from LaFAN1, so this is yours to "
        "decide, not a default to inherit.",
    )
    parser.add_argument(
        "--with-optimizer",
        action="store_true",
        help="publish the optimizer state too. Triples the file and is only "
        "useful for resuming, so it is off by default.",
    )
    parser.add_argument("--revision", default="", help="branch to push to")
    parser.add_argument(
        "--subfolder",
        default="",
        help="upload under this folder inside the repo instead of the repo "
        "root, e.g. 'phase1_v1'. Lets several versions of a checkpoint "
        "(v1, v2, a Phase 2 round, ...) live as folders in one repo -- "
        "'ErnestBeckham/MMM-versions' -- rather than needing a separate "
        "repo per version. The model card is still written and uploaded "
        "per version, at '<subfolder>/README.md', since each version's "
        "architecture and metrics differ.",
    )
    parser.add_argument("--message", default="", help="commit message")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the card and the file list, upload nothing, and do not "
        "require a token",
    )
    return parser.parse_args(argv)


def _architecture_rows(architecture: dict) -> str:
    order = [
        "backbone",
        "n_layers",
        "d_model",
        "d_state",
        "d_conv",
        "d_inner",
        "head",
        "pose_dim",
        "joystick_dim",
        "input_dim",
        "output_dim",
        "predict_contacts",
        "parameters",
    ]
    rows = []
    for key in order:
        if key in architecture:
            value = architecture[key]
            if key == "parameters" and isinstance(value, int):
                value = f"{value:,}"
            rows.append(f"| {key} | {value} |")
    for key, value in architecture.items():
        if key not in order:
            rows.append(f"| {key} | {value} |")
    return "\n".join(rows)


def _metrics_section(history) -> str:
    """The training/validation curve table.

    `history.json` is columnar -- `{metric: [value per epoch]}` -- as
    `PreTrainer._record` builds it, not a list of per epoch rows.
    """
    if not history or "train_total" not in history:
        return (
            "No usable `history.json` was found beside the checkpoint, so no "
            "training curve is reported."
        )

    def column(name):
        return history.get(name) or []

    train = column("train_total")
    validation = column("val_total")
    n_epochs = len(train)

    def at(values, i, default=float("nan")):
        return values[i] if 0 <= i < len(values) else default

    best = (
        min(range(len(validation)), key=lambda i: validation[i])
        if validation
        else n_epochs - 1
    )

    lines = ["| epoch | train | validation |", "| --- | --- | --- |"]
    # First, best and last, rather than every epoch.
    seen = set()
    for label, i in (("first", 0), ("best", best), ("last", n_epochs - 1)):
        if i in seen:
            continue
        seen.add(i)
        lines.append(
            f"| {i + 1} ({label}) | {at(train, i):.4f} | {at(validation, i):.4f} |"
        )
    return "\n".join(lines) + f"\n\nTrained for {n_epochs} epochs."


def build_card(args, metadata: dict, history, filename: str) -> str:
    architecture = metadata.get("architecture", {}) or {}
    database = metadata.get("database", {}) or {}
    phase = metadata.get("phase", 1)
    training = _metrics_section(history)

    clips = database.get("clips") or []
    return CARD_TEMPLATE.format(
        license_line=f"license: {args.license}\n" if args.license else "",
        title=args.repo_id.split("/")[-1],
        phase=phase,
        phase_name=PHASE_NAMES.get(phase, "unknown phase"),
        db_fingerprint=database.get("fingerprint", "unknown"),
        db_rows=f"{database.get('n_rows', 0):,}",
        db_pose_dim=database.get("pose_dim", "unknown"),
        db_frame_time=database.get("frame_time", "unknown"),
        db_clips=f"{len(clips)}" + (f" ({', '.join(clips[:4])}...)" if len(clips) > 4 else ""),
        architecture_rows=_architecture_rows(architecture),
        metrics_section=training,
        filename=filename,
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1
    if args.checkpoint.endswith(".pt"):
        print(
            f"{args.checkpoint} is a pickle checkpoint. Re-save it as "
            "safetensors before publishing; a published pickle executes code "
            "in whatever process downloads it.",
            file=sys.stderr,
        )
        return 1

    metadata = read_metadata(args.checkpoint)

    history_path = args.history or os.path.join(
        os.path.dirname(args.checkpoint), "history.json"
    )
    history = None
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as handle:
            history = json.load(handle)

    filename = os.path.basename(args.checkpoint)

    if not args.with_optimizer and metadata.get("has_optimizer"):
        print(
            f"note: {filename} carries optimizer state, which triples its size "
            "and is useless for inference. Re-save with optimizer=False, or "
            "pass --with-optimizer to publish it anyway."
        )

    card = build_card(args, metadata, history, filename)

    uploads = [(args.checkpoint, filename), (None, "README.md")]
    for extra in (
        "last.safetensors",
        "metrics.csv",
        "history.json",
        "manifest.json",
        "run_args.json",
    ):
        if extra == filename:
            continue  # already queued as the checkpoint itself
        candidate = os.path.join(os.path.dirname(args.checkpoint), extra)
        if os.path.exists(candidate):
            uploads.append((candidate, extra))

    subfolder = args.subfolder.strip("/")
    print(f"\nrepository : {args.repo_id}  ({'PUBLIC' if args.public else 'private'})")
    if subfolder:
        print(f"subfolder  : {subfolder}/")
    print("files      :")
    for source, name in uploads:
        size = os.path.getsize(source) / 1e6 if source else len(card) / 1e6
        path_in_repo = f"{subfolder}/{name}" if subfolder else name
        print(f"  {path_in_repo:30} {size:8.2f} MB")

    # Write the card beside the checkpoint whatever happens next. A dry run is
    # the natural way to produce it for a manual upload, and an upload that
    # fails on authentication should still leave the card behind rather than
    # discard work that is already done.
    card_path = os.path.join(os.path.dirname(args.checkpoint) or ".", "README.md")
    try:
        with open(card_path, "w", encoding="utf-8") as handle:
            handle.write(card)
        print(f"wrote {card_path}")
    except OSError as error:
        print(f"could not write {card_path}: {error}", file=sys.stderr)
        card_path = None

    if args.dry_run:
        print("\n--- README.md ---")
        print(card)
        print("--- end ---\n")
        print("dry run: nothing was uploaded.")
        return 0

    from huggingface_hub import HfApi

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
    print(f"authenticated as {whoami.get('name', '?')}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )

    if card_path is None:
        print("the model card could not be written, so it will not be uploaded",
              file=sys.stderr)
        uploads = [entry for entry in uploads if entry[1] != "README.md"]

    for source, name in uploads:
        path_in_repo = f"{subfolder}/{name}" if subfolder else name
        api.upload_file(
            path_or_fileobj=source or card_path,
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision or None,
            commit_message=args.message
            or f"Phase {metadata.get('phase', 1)} checkpoint, epoch {metadata.get('epoch')}",
        )
        print(f"uploaded {path_in_repo}")

    url = f"https://huggingface.co/{args.repo_id}"
    print(f"\n{url}" + (f"/tree/main/{subfolder}" if subfolder else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
