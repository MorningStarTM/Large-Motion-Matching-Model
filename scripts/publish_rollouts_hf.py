"""
Push a directory of collected rollouts (round 1 or a DAgger round) to a
Hugging Face dataset repository.

Unlike `scripts/publish_rollouts.py` (Kaggle datasets), this uploads each
shard file individually rather than replacing the whole repo with one
folder snapshot. That matters when several separate sessions each collect
their own slice of rollouts and need to accumulate into *one* repo over
time: a Kaggle dataset version is a full-folder replace (the previous
version's files are not kept unless you re-include them), so N independent
sessions pushing to the same Kaggle dataset slug silently lose everything
but the last push. HF file uploads are additive by path, so as long as each
session uploads under its own --subfolder, any number of sessions -- 3, 5,
10, scaling over time -- can push into the same repo without colliding or
overwriting each other.

Usage (inside a Kaggle/Colab notebook, or anywhere with an HF token):

    # session 1
    python scripts/publish_rollouts_hf.py --dir /kaggle/working/rollouts/dagger2_v2 \
        --repo-id yourname/mmm-rollouts --subfolder dagger2_v2/run1

    # session 2, 3, ... N -- same repo, a fresh --subfolder each time
    python scripts/publish_rollouts_hf.py --dir /kaggle/working/rollouts/dagger2_v2 \
        --repo-id yourname/mmm-rollouts --subfolder dagger2_v2/run2

    # see what would be uploaded, touch nothing
    python scripts/publish_rollouts_hf.py --dir ... --repo-id ... --subfolder ... --dry-run

--subfolder must be unique per session. Reusing one just overwrites that
session's own shards (safe), but using the same --subfolder from two
different sessions loses one of them, the same way two notebooks writing
shard_00000.npz into the same folder would.

Authentication comes from the environment, in the order `huggingface_hub`
itself looks: `HF_TOKEN`, then a token saved by `huggingface-cli login`. No
token is ever read from an argument, so it cannot end up in a shell history.

Public by default, unlike the model publisher -- rollouts are a synthetic
joystick driving a model against LaFAN1, not LaFAN1 itself, so there is no
third party licence question the way there is for `publish_checkpoint.py`.
Pass `--private` if you would rather it was not.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir", required=True, help="the rollout directory to upload, e.g. "
        "/kaggle/working/rollouts/dagger2_v2"
    )
    parser.add_argument(
        "--repo-id", required=True, help="HF dataset repo, e.g. yourname/mmm-rollouts. "
        "Reused across every session -- that is the point."
    )
    parser.add_argument(
        "--subfolder", required=True,
        help="upload under this folder inside the repo, e.g. 'dagger2_v2/run1'. "
        "Must be unique per session -- see the module docstring.",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="", help="commit message")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print what would be uploaded, upload nothing, require no credentials",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.isdir(args.dir):
        print(f"{args.dir} is not a directory", file=sys.stderr)
        return 1

    files = sorted(
        f for f in os.listdir(args.dir) if os.path.isfile(os.path.join(args.dir, f))
    )
    shards = [f for f in files if f.endswith(".npz")]
    if not shards:
        print(f"no .npz shards in {args.dir} -- nothing to publish", file=sys.stderr)
        return 1

    subfolder = args.subfolder.strip("/")
    total_bytes = sum(os.path.getsize(os.path.join(args.dir, f)) for f in files)

    print(f"directory  : {args.dir}")
    print(f"files      : {len(files)} ({len(shards)} shards)")
    print(f"size       : {total_bytes / 1e6:.1f} MB")
    print(f"repository : {args.repo_id}  ({'private' if args.private else 'public'})")
    print(f"subfolder  : {subfolder}/")

    if args.dry_run:
        print("\nwould upload:")
        for name in files:
            print(f"  {subfolder}/{name}")
        print("\ndry run: nothing was uploaded.")
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
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    for name in files:
        path_in_repo = f"{subfolder}/{name}"
        api.upload_file(
            path_or_fileobj=os.path.join(args.dir, name),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.message or f"rollouts: {subfolder}",
        )
        print(f"uploaded {path_in_repo}")

    url = f"https://huggingface.co/datasets/{args.repo_id}"
    print(f"\n{url}/tree/main/{subfolder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
