"""
Where a run may read from and write to, resolved rather than assumed.

The same training script has to work on a workstation, in a Kaggle notebook and
in Colab, and those disagree about something basic: **the dataset directory is
not writable**. On Kaggle `/kaggle/input` is mounted read only, so the habit of
putting a built database, a retrieval cache and checkpoints next to the source
data fails at the first write, several minutes into a session.

This module answers three questions explicitly:

- which platform is this (by *testing*, not by trusting an environment
  variable alone)
- where is it safe to write
- which of those paths did the user override

Nothing here imports torch or numpy, so it stays cheap and testable.

Paths by platform:

| platform | dataset input | writable scratch |
| --- | --- | --- |
| kaggle | /kaggle/input/... (read only) | /kaggle/working, /kaggle/temp |
| colab | /content/... | /content |
| local | anywhere | the working directory |

A Kaggle session also ends on a wall clock limit, so anything expensive to
recompute -- the 190 MB database, the seven minute retrieval cache -- belongs
under `/kaggle/working`, which is preserved as the notebook's output, rather
than under `/kaggle/temp`, which is not.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

__all__ = ["detect_platform", "is_writable", "Workspace"]

KAGGLE_WORKING = "/kaggle/working"
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_TEMP = "/kaggle/temp"
COLAB_ROOT = "/content"


def detect_platform() -> str:
    """"kaggle", "colab" or "local".

    Checks the filesystem as well as the environment, because a notebook image
    can carry the environment variables without the mounts, and because the
    variables have been renamed between Kaggle versions. The directory either
    exists or it does not.
    """
    if os.path.isdir(KAGGLE_WORKING) or any(
        key.startswith("KAGGLE_") for key in os.environ
    ):
        if os.path.isdir(KAGGLE_WORKING):
            return "kaggle"

    if os.path.isdir(COLAB_ROOT) and "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except ImportError:
        pass

    return "local"


def is_writable(path: str) -> bool:
    """Whether a directory can actually be written to.

    Tested by writing, not by `os.access`, which reports the permission bits
    and not the read only mount that Kaggle actually uses.
    """
    if not os.path.isdir(path):
        parent = os.path.dirname(os.path.abspath(path)) or "."
        return os.path.isdir(parent) and is_writable(parent)
    try:
        with tempfile.NamedTemporaryFile(dir=path):
            return True
    except OSError:
        return False


@dataclass
class Workspace:
    """Resolved paths for one run.

    Attributes:
        platform: as `detect_platform`.
        root: the writable base for anything this run produces.
        out: checkpoints, history, TensorBoard logs.
        database: where a built database lives, or is built to.
        cache: BVH parse caches and the retrieval cache.

    Every one of these can be overridden; the defaults only apply to what the
    caller left unset.
    """

    platform: str
    root: str
    out: str
    database: str
    cache: str

    @classmethod
    def resolve(
        cls,
        out: str | None = None,
        database: str | None = None,
        cache: str | None = None,
        root: str | None = None,
        run_name: str = "phase1",
    ) -> "Workspace":
        platform = detect_platform()

        if root is None:
            if platform == "kaggle":
                # /kaggle/working is preserved as the notebook's output;
                # /kaggle/temp is discarded when the session ends, which would
                # throw away the database and the retrieval cache.
                root = KAGGLE_WORKING
            elif platform == "colab":
                root = COLAB_ROOT
            else:
                root = os.getcwd()

        if not is_writable(root):
            fallback = KAGGLE_TEMP if os.path.isdir(KAGGLE_TEMP) else tempfile.gettempdir()
            print(f"[workspace] {root} is not writable; falling back to {fallback}")
            root = fallback

        resolved = cls(
            platform=platform,
            root=root,
            out=out or os.path.join(root, "runs", run_name),
            database=database or os.path.join(root, "data", "databases", run_name),
            cache=cache or os.path.join(root, "data", "cache"),
        )
        resolved.create()
        return resolved

    def create(self) -> None:
        for path in (self.out, self.database, self.cache):
            os.makedirs(path, exist_ok=True)

    def describe(self) -> dict:
        return {
            "platform": self.platform,
            "root": self.root,
            "out": self.out,
            "database": self.database,
            "cache": self.cache,
        }

    def __str__(self) -> str:
        lines = [f"workspace ({self.platform})"]
        for key, value in self.describe().items():
            if key != "platform":
                lines.append(f"  {key:9} {value}")
        return "\n".join(lines)
