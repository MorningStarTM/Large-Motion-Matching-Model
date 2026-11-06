"""
Download a published checkpoint from the Hugging Face Hub into a versioned
local folder.

The counterpart to `scripts/publish_checkpoint.py`. That one pushes a run up;
this one pulls a run down, and the only real design decision is where to put
it.

Why versioned folders
---------------------
`runs/phase_1/` already holds a run, and a download that overwrote it would
destroy the only copy of whatever was there. Worse, it would do so silently
and produce a `runs/phase_1/` whose contents no longer match its own
`run_args.json`. So a download never writes into an existing folder: it
allocates the next free `runs/<name>_v<n>/` and records where the contents
came from in `source.json`.

That file is the point of the exercise. A checkpoint with no provenance is
nearly worthless a month later -- you cannot tell which run produced it, which
commit of the Hub repo it came from, or whether the local copy has been
edited. `source.json` holds the repo, the exact revision, the commit date, the
download time and a SHA-256 of every file, so all four questions have answers.

Re-running is a no-op. The script checks each existing version's `source.json`
for the revision it is about to fetch and stops if it is already there, which
means it can be run in a loop or a script without accumulating copies.

Usage:

    # the current revision of the default repo
    python scripts/fetch_checkpoint.py

    # a specific repo and revision
    python scripts/fetch_checkpoint.py --repo ErnestBeckham/MMM --revision d809325

    # see what the Hub holds without downloading
    python scripts/fetch_checkpoint.py --list

A private repo needs a token in `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`; it is
never read from a file or an argument, so it cannot end up in shell history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_REPO = "ErnestBeckham/MMM"
DEFAULT_NAME = "phase_1"
DEFAULT_ROOT = "runs"

# What a published run consists of. Anything else in the repo -- the model
# card, .gitattributes -- is Hub bookkeeping and is not part of the run.
RUN_FILES = (
    "published.safetensors",
    "history.json",
    "metrics.csv",
    "manifest.json",
    "run_args.json",
)

VERSION_PATTERN = re.compile(r"^(?P<name>.+)_v(?P<number>\d+)$")


def token_from_environment():
    """The Hub token, from the environment only.

    Deliberately not an argument: a token passed on the command line lands in
    shell history and in the process list, where anyone on the machine can
    read it.
    """
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def existing_versions(root: str, name: str) -> list:
    """Every `<root>/<name>_v<n>` directory, in version order."""
    if not os.path.isdir(root):
        return []
    found = []
    for entry in sorted(os.listdir(root)):
        match = VERSION_PATTERN.match(entry)
        if match and match.group("name") == name:
            found.append((int(match.group("number")), os.path.join(root, entry)))
    return [path for _, path in sorted(found)]


def read_source(directory: str) -> dict | None:
    path = os.path.join(directory, "source.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def find_revision(root: str, name: str, revision: str):
    """Whether this revision has already been downloaded, and where."""
    for directory in existing_versions(root, name):
        source = read_source(directory)
        if source and source.get("revision") == revision:
            return directory
    return None


def next_version_path(root: str, name: str) -> str:
    used = set()
    if os.path.isdir(root):
        for entry in os.listdir(root):
            match = VERSION_PATTERN.match(entry)
            if match and match.group("name") == name:
                used.add(int(match.group("number")))
    number = 1
    while number in used:
        number += 1
    return os.path.join(root, f"{name}_v{number}")


def describe_repo(repo: str, token) -> None:
    """Print the repo's files and commits without downloading anything."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.model_info(repo, files_metadata=True)
    print(f"{repo} @ {info.sha}")
    print(f"\n{'file':32s} {'bytes':>12s}")
    print("-" * 46)
    for sibling in sorted(info.siblings, key=lambda s: s.rfilename):
        size = sibling.size if sibling.size is not None else "?"
        print(f"{sibling.rfilename:32s} {str(size):>12s}")

    print("\ncommits, newest first:")
    for commit in api.list_repo_commits(repo):
        print(
            f"  {commit.commit_id[:10]}  "
            f"{commit.created_at:%Y-%m-%d %H:%M}  {commit.title}"
        )


def fetch(
    repo: str,
    name: str,
    root: str,
    revision: str | None,
    token,
    files=RUN_FILES,
    force: bool = False,
) -> int:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi(token=token)
    info = api.model_info(repo, revision=revision)
    resolved = info.sha

    commits = api.list_repo_commits(repo)
    commit = next((c for c in commits if c.commit_id == resolved), None)
    created = commit.created_at if commit else None

    already = find_revision(root, name, resolved)
    if already and not force:
        print(
            f"revision {resolved[:10]} is already in {already}; nothing to do.\n"
            "Pass --force to download it again as a new version."
        )
        return 0

    available = set(api.list_repo_files(repo, revision=resolved))
    wanted = [f for f in files if f in available]
    missing = [f for f in files if f not in available]
    if not wanted:
        print(f"none of {list(files)} are in {repo} at {resolved[:10]}")
        return 1

    destination = next_version_path(root, name)
    os.makedirs(destination, exist_ok=True)
    print(f"{repo} @ {resolved[:10]} -> {destination}")

    records = {}
    for filename in wanted:
        try:
            cached = hf_hub_download(
                repo_id=repo, filename=filename, revision=resolved, token=token
            )
        except EntryNotFoundError:
            print(f"  {filename:28s} not found, skipped")
            continue
        target = os.path.join(destination, filename)
        # Copied out of the Hub cache rather than symlinked: the cache is
        # shared and can be cleared, and a run folder that evaporates when
        # someone runs `huggingface-cli delete-cache` is not a stored run.
        shutil.copyfile(cached, target)
        size = os.path.getsize(target)
        digest = sha256(target)
        records[filename] = {"bytes": size, "sha256": digest}
        print(f"  {filename:28s} {size:>10,d} bytes  {digest[:12]}")

    if missing:
        print(f"  not in this revision: {', '.join(missing)}")

    source = {
        "repo": repo,
        "repo_type": "model",
        "revision": resolved,
        "commit_title": commit.title if commit else None,
        "commit_date": created.isoformat() if created else None,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    with open(
        os.path.join(destination, "source.json"), "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(source, handle, indent=2)
        handle.write("\n")

    print(f"\nwrote {destination}/source.json")
    _summarise(destination)
    return 0


def _summarise(directory: str) -> None:
    """Print what the downloaded run actually contains.

    A downloaded checkpoint is opaque until something reads it, and the most
    useful thing to know immediately is how far the run got.
    """
    history_path = os.path.join(directory, "history.json")
    if not os.path.isfile(history_path):
        return
    try:
        with open(history_path, encoding="utf-8") as handle:
            history = json.load(handle)
    except (OSError, ValueError):
        return

    if isinstance(history, dict):
        keys = sorted(history)
        epochs = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
    else:
        keys = sorted(history[0]) if history else []
        epochs = len(history)

    print(f"  {epochs} epochs, {len(keys)} metrics")
    for key in ("val_total", "val_rollout_total", "train_total"):
        series = history.get(key) if isinstance(history, dict) else None
        if series:
            best = min(series)
            print(
                f"    {key:20s} first {series[0]:.4f}  last {series[-1]:.4f}  "
                f"best {best:.4f} @ epoch {series.index(best) + 1}"
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a published checkpoint into a versioned folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo id")
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="local run name; folders are <root>/<name>_v<n>",
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="where runs live")
    parser.add_argument(
        "--revision",
        default=None,
        help="a branch, tag or commit sha. Defaults to the repo's current head.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="show the repo's files and commits, download nothing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even if this revision is already present",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=list(RUN_FILES),
        help="which files make up a run",
    )

    args = parser.parse_args(argv)
    token = token_from_environment()

    try:
        if args.list:
            describe_repo(args.repo, token)
            return 0
        return fetch(
            repo=args.repo,
            name=args.name,
            root=args.root,
            revision=args.revision,
            token=token,
            files=args.files,
            force=args.force,
        )
    except Exception as error:  # noqa: BLE001
        message = str(error)
        print(f"{type(error).__name__}: {message}")
        if "401" in message or "403" in message or "gated" in message.lower():
            print(
                "\nThis looks like an authentication failure. Set HF_TOKEN in "
                "your environment -- the script never takes a token as an "
                "argument, because that would put it in your shell history."
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
