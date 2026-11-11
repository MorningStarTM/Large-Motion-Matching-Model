"""
Push a directory of collected rollouts (round 1 or a DAgger round) to a
Kaggle dataset, so a session that is about to end does not lose hours of
collection to `/kaggle/working/` being wiped.

A separate, manually invoked script, for the same reason
`scripts/publish_checkpoint.py` is: leaving this machine is a deliberate act,
not something collection should do on its own.

Usage (inside a Kaggle notebook, or anywhere with the Kaggle API configured):

    # first time for this dataset slug -- creates it
    python scripts/publish_rollouts.py --dir /kaggle/working/rollouts/round1 --dataset-id yourname/mmm-rollouts-round1 --new

    # a later shard, or a new round, pushed as a new version of the same dataset
    python scripts/publish_rollouts.py --dir /kaggle/working/rollouts/round1 --dataset-id yourname/mmm-rollouts-round1

    # see what would be uploaded, touch nothing
    python scripts/publish_rollouts.py --dir /kaggle/working/rollouts/round1 --dataset-id yourname/mmm-rollouts-round1 --dry-run

Authentication is whatever the `kaggle` package itself reads: `KAGGLE_USERNAME`
/ `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`. On Kaggle itself, add your own
Kaggle API token as a notebook secret and export those two variables from it
before running this; nothing here reads or stores a token directly.

Public by default, unlike the model publisher -- rollouts are a synthetic
joystick driving a model against LaFAN1, not LaFAN1 itself, so there is no
third party licence question the way there is for `publish_checkpoint.py`.
Pass `--private` if you would rather it was not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir", required=True, help="the rollout directory to upload, e.g. "
        "/kaggle/working/rollouts/round1"
    )
    parser.add_argument(
        "--dataset-id", required=True,
        help="Kaggle dataset slug, 'yourname/some-title'. Reused across runs "
        "of the same round so later shards become new versions."
    )
    parser.add_argument("--title", default="", help="defaults to the slug's second half")
    parser.add_argument(
        "--new", action="store_true",
        help="create the dataset. Omit this once it exists -- passing it "
        "again just fails with 'dataset already exists'."
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="", help="version notes")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print what would be uploaded, upload nothing, require no credentials"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.isdir(args.dir):
        print(f"{args.dir} is not a directory", file=sys.stderr)
        return 1

    shards = sorted(f for f in os.listdir(args.dir) if f.endswith(".npz"))
    if not shards:
        print(f"no .npz shards in {args.dir} -- nothing to publish", file=sys.stderr)
        return 1

    total_bytes = sum(os.path.getsize(os.path.join(args.dir, f)) for f in shards)
    title = args.title or args.dataset_id.split("/")[-1]

    print(f"directory  : {args.dir}")
    print(f"shards     : {len(shards)}")
    print(f"size       : {total_bytes / 1e6:.1f} MB")
    print(f"dataset-id : {args.dataset_id}  ({'private' if args.private else 'public'})")

    # The Kaggle API uploads a folder by reading a dataset-metadata.json
    # inside it, rather than taking the slug as an argument -- written here,
    # into the rollout directory itself, right next to the shards and
    # collection_summary.json it is already publishing.
    metadata = {
        "title": title,
        "id": args.dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    metadata_path = os.path.join(args.dir, "dataset-metadata.json")

    if args.dry_run:
        print(f"\nwould write {metadata_path}:")
        print(json.dumps(metadata, indent=2))
        print("\ndry run: nothing was uploaded.")
        return 0

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:  # noqa: BLE001 - the kaggle package raises several types
        print(
            f"not authenticated ({error}). Set KAGGLE_USERNAME and KAGGLE_KEY, "
            "or place a kaggle.json at ~/.kaggle/kaggle.json.",
            file=sys.stderr,
        )
        return 1

    if args.new:
        api.dataset_create_new(
            folder=args.dir, convert_to_csv=False, dir_mode="zip",
            public=not args.private,
        )
    else:
        api.dataset_create_version(
            folder=args.dir, version_notes=args.message or "updated rollouts",
            convert_to_csv=False, dir_mode="zip",
        )

    print(f"\nhttps://www.kaggle.com/datasets/{args.dataset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
