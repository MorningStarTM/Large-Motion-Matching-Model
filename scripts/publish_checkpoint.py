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

Retrieval augmented state space model for real time character animation: a
Mamba backbone that takes a joystick request, its own previous pose, and `k`
poses retrieved from a motion matching database, and predicts the next pose.

This is a **Phase {phase}** checkpoint ({phase_name}).

## This model is not usable on its own

The retrieved candidate poses are part of the input, not an optional extra.
Without the matching database the model is being fed zeros in {candidate_share}
of its input width and will not produce correct motion.

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

## Retrieval gain

The number this project is actually judged on. It is the validation loss with
the retrieved candidates zeroed, minus the loss with them present, so positive
means retrieval is helping.

It is reported in two regimes and **only the free running one is meaningful**.
Under teacher forcing the model is handed the true previous pose every frame,
and at 60Hz that is a far better predictor than any retrieved candidate --
measured on this database, copying the previous pose scores 2.29 while copying
the nearest candidate scores 16.24. A teacher forced model is therefore right
to ignore retrieval, and its gain sits at zero whatever the model has learned.

{gain_section}

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
        "n_candidates",
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


def _metrics_section(history) -> tuple[str, str]:
    """Returns `(training table, retrieval gain table)`.

    `history.json` is columnar -- `{metric: [value per epoch]}` -- as
    `PreTrainer._record` builds it, not a list of per epoch rows.
    """
    if not history or "train_total" not in history:
        return (
            "No usable `history.json` was found beside the checkpoint, so no "
            "training curve is reported.",
            "Not reported: no history file was available.",
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
    training = "\n".join(lines) + f"\n\nTrained for {n_epochs} epochs."

    gain_lines = [
        "| regime | with retrieval | without | gain |",
        "| --- | --- | --- | --- |",
    ]
    if validation and column("val_total_without_retrieval"):
        gain_lines.append(
            f"| teacher forced | {at(validation, best):.4f} "
            f"| {at(column('val_total_without_retrieval'), best):.4f} "
            f"| {at(column('val_retrieval_gain'), best):+.4f} |"
        )
    if column("val_rollout_total"):
        gain_lines.append(
            f"| **free running** | {at(column('val_rollout_total'), best):.4f} "
            f"| {at(column('val_rollout_total_without_retrieval'), best):.4f} "
            f"| **{at(column('val_rollout_retrieval_gain'), best):+.4f}** |"
        )

    if len(gain_lines) > 2:
        gain = "\n".join(gain_lines)
        if not column("val_rollout_total"):
            gain += (
                "\n\n**This checkpoint predates the free running evaluation**, so "
                "only the uninformative teacher forced gain is available for it."
            )
    else:
        gain = "Not reported: the history file carries no ablation columns."
    return training, gain


def build_card(args, metadata: dict, history, filename: str) -> str:
    architecture = metadata.get("architecture", {}) or {}
    database = metadata.get("database", {}) or {}
    phase = metadata.get("phase", 1)
    training, gain = _metrics_section(history)

    pose_dim = architecture.get("pose_dim")
    input_dim = architecture.get("input_dim")
    candidates = architecture.get("n_candidates")
    if pose_dim and input_dim and candidates:
        share = f"{100 * candidates * pose_dim / input_dim:.0f}%"
    else:
        share = "most"

    clips = database.get("clips") or []
    return CARD_TEMPLATE.format(
        license_line=f"license: {args.license}\n" if args.license else "",
        title=args.repo_id.split("/")[-1],
        phase=phase,
        phase_name=PHASE_NAMES.get(phase, "unknown phase"),
        candidate_share=share,
        db_fingerprint=database.get("fingerprint", "unknown"),
        db_rows=f"{database.get('n_rows', 0):,}",
        db_pose_dim=database.get("pose_dim", "unknown"),
        db_frame_time=database.get("frame_time", "unknown"),
        db_clips=f"{len(clips)}" + (f" ({', '.join(clips[:4])}...)" if len(clips) > 4 else ""),
        architecture_rows=_architecture_rows(architecture),
        metrics_section=training,
        gain_section=gain,
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
    for extra in ("manifest.json", "history.json"):
        candidate = os.path.join(os.path.dirname(args.checkpoint), extra)
        if os.path.exists(candidate):
            uploads.append((candidate, extra))

    print(f"\nrepository : {args.repo_id}  ({'PUBLIC' if args.public else 'private'})")
    print("files      :")
    for source, name in uploads:
        size = os.path.getsize(source) / 1e6 if source else len(card) / 1e6
        print(f"  {name:22} {size:8.2f} MB")

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
        api.upload_file(
            path_or_fileobj=source or card_path,
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision or None,
            commit_message=args.message
            or f"Phase {metadata.get('phase', 1)} checkpoint, epoch {metadata.get('epoch')}",
        )
        print(f"uploaded {name}")

    print(f"\nhttps://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
