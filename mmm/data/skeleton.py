"""
Skeleton definition: the hidden tree of joints that drives the character mesh.

LaFAN1 resolved ships 75 joints, which includes a full five finger hand on each
side. Fingers are irrelevant to locomotion motion matching and would triple the
pose width for nothing, so this module also defines the locomotion subset the
model actually uses: 24 joints, keeping the spine, neck, head, arms and legs
down to the toe base.

That subset is what makes the plan's numbers work out. 24 joints x 6D rotation
is 144, plus 6 numbers of root velocity, giving a 150 wide pose -- the "about
150" of plan section 4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Skeleton", "LOCOMOTION_JOINTS", "LEFT_FOOT", "RIGHT_FOOT", "HIPS"]


# The 24 joints kept for locomotion, in the order they appear in LaFAN1
# resolved. Order matters: it fixes the layout of every pose vector, every
# database row and every exported weight, so it is declared once here and
# never derived.
LOCOMOTION_JOINTS = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Neck1",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
)

HIPS = "Hips"
LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"


@dataclass
class Skeleton:
    """A joint tree.

    Attributes:
        names: joint names, in the canonical order. Index 0 must be the root.
        parents: `parents[i]` is the index of joint `i`'s parent, and `-1` for
            the root. Guaranteed topologically sorted: `parents[i] < i`, which
            is what lets forward kinematics run as a single forward pass.
        offsets: `(n_joints, 3)` rest offset of each joint from its parent, in
            the file's units (centimetres for LaFAN1).
    """

    names: list[str]
    parents: np.ndarray
    offsets: np.ndarray
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.parents = np.asarray(self.parents, dtype=np.int32)
        self.offsets = np.asarray(self.offsets, dtype=np.float32)

        if self.parents.shape != (len(self.names),):
            raise ValueError(
                f"parents has shape {self.parents.shape}, expected ({len(self.names)},)"
            )
        if self.offsets.shape != (len(self.names), 3):
            raise ValueError(
                f"offsets has shape {self.offsets.shape}, expected ({len(self.names)}, 3)"
            )
        if self.parents[0] != -1:
            raise ValueError("joint 0 must be the root, with parent -1")
        if np.any(self.parents[1:] >= np.arange(1, len(self.names))):
            raise ValueError(
                "parents must be topologically sorted so that parents[i] < i; "
                "forward kinematics relies on it"
            )
        self._index = {name: i for i, name in enumerate(self.names)}

    # -- lookups ----------------------------------------------------------

    @property
    def n_joints(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(
                f"joint {name!r} not in this skeleton. Available: {self.names}"
            ) from None

    def indices(self, names) -> np.ndarray:
        return np.array([self.index(n) for n in names], dtype=np.int64)

    def has(self, name: str) -> bool:
        return name in self._index

    # -- derived quantities ----------------------------------------------

    def bone_lengths(self) -> np.ndarray:
        """Distance from each joint to its parent. The root's entry is 0."""
        lengths = np.linalg.norm(self.offsets, axis=-1)
        lengths[0] = 0.0
        return lengths

    def leg_length(self) -> float:
        """Hip to foot distance along the chain, used to scale root
        translation when retargeting between skeletons."""
        chain = ["LeftUpLeg", "LeftLeg", "LeftFoot"]
        if not all(self.has(j) for j in chain):
            raise KeyError(f"leg_length needs the joints {chain}")
        lengths = self.bone_lengths()
        return float(sum(lengths[self.index(j)] for j in chain))

    def children(self) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in range(self.n_joints)]
        for i, p in enumerate(self.parents):
            if p >= 0:
                out[int(p)].append(i)
        return out

    # -- subsetting -------------------------------------------------------

    def subset(self, names) -> tuple["Skeleton", np.ndarray]:
        """Restrict to `names`, re-rooting any joint whose parent was dropped.

        This is how the 75 joint LaFAN1 skeleton becomes the 24 joint
        locomotion skeleton. A dropped intermediate joint would break the
        chain, so each kept joint's parent is taken to be its nearest kept
        ancestor, and its offset is accumulated along the dropped span -- the
        rest pose is therefore preserved exactly, even though the tree is
        shallower.

        Returns:
            The new skeleton, and `(n_kept,)` indices into this skeleton's
            joint order, for slicing motion arrays.
        """
        keep = [n for n in names if self.has(n)]
        missing = [n for n in names if not self.has(n)]
        if missing:
            raise KeyError(f"joints not present in this skeleton: {missing}")

        kept_set = {self.index(n) for n in keep}
        source_indices = np.array([self.index(n) for n in keep], dtype=np.int64)

        # Rest position of every joint in the original skeleton, so accumulated
        # offsets can be read off directly rather than summed by hand.
        rest = np.zeros((self.n_joints, 3), dtype=np.float32)
        for i in range(self.n_joints):
            p = int(self.parents[i])
            rest[i] = self.offsets[i] if p < 0 else rest[p] + self.offsets[i]

        new_parents = []
        new_offsets = []
        position_in_subset = {src: j for j, src in enumerate(source_indices)}
        for src in source_indices:
            # Walk up to the nearest kept ancestor.
            ancestor = int(self.parents[src])
            while ancestor >= 0 and ancestor not in kept_set:
                ancestor = int(self.parents[ancestor])

            if ancestor < 0:
                new_parents.append(-1)
                new_offsets.append(rest[src] if src == source_indices[0] else self.offsets[src])
            else:
                new_parents.append(position_in_subset[ancestor])
                new_offsets.append(rest[src] - rest[ancestor])

        # The subset root keeps its own offset rather than an accumulated one:
        # for the root the offset is unused by FK anyway (its position comes
        # from the motion channels), and preserving it keeps round trips clean.
        new_offsets[0] = self.offsets[source_indices[0]]

        return (
            Skeleton(names=list(keep), parents=np.array(new_parents), offsets=np.array(new_offsets)),
            source_indices,
        )

    def locomotion_subset(self) -> tuple["Skeleton", np.ndarray]:
        """`subset(LOCOMOTION_JOINTS)`, the 24 joint skeleton the model uses."""
        return self.subset(LOCOMOTION_JOINTS)

    # -- validation -------------------------------------------------------

    def assert_matches(self, other: "Skeleton", relative_tol: float = 0.01) -> None:
        """Fail loudly if two skeletons differ.

        A silently wrong joint order produces motion that looks almost right,
        which is far more expensive to debug than a refused load.

        Two deliberate choices about what counts as a difference:

        - **The root offset is ignored.** In LaFAN1 resolved the root's OFFSET
          is the character's world position in that clip, not a bone, so it
          differs by hundreds of centimetres between clips that otherwise share
          an identical skeleton. Forward kinematics takes the root's position
          from the motion channels and never reads this offset, so it carries
          no information worth comparing.
        - **Bone offsets are compared relatively**, as a fraction of bone
          length. Measured across LaFAN1 resolved subjects 1, 2, 3 and 5, every
          bone agrees to within 0.15% and the leg length is 89.42 cm for all of
          them -- the dataset is retargeted to a single skeleton. The 1% default
          therefore accepts that residue while still catching a genuinely
          different character.

        Raises:
            ValueError: naming the joints responsible, so a mismatch is
                actionable rather than just fatal.
        """
        if self.names != other.names:
            only_self = set(self.names) - set(other.names)
            only_other = set(other.names) - set(self.names)
            raise ValueError(
                "skeleton joint names differ.\n"
                f"  only in first:  {sorted(only_self)}\n"
                f"  only in second: {sorted(only_other)}"
            )
        if not np.array_equal(self.parents, other.parents):
            raise ValueError("skeleton joint names match but the tree structure differs")

        if self.n_joints == 1:
            return

        difference = np.abs(self.offsets[1:] - other.offsets[1:]).max(axis=-1)
        lengths = np.linalg.norm(self.offsets[1:], axis=-1)
        # A zero length bone has no scale to be relative to, so fall back to an
        # absolute comparison for it.
        relative = np.where(lengths > 1e-6, difference / np.maximum(lengths, 1e-9), difference)

        worst = int(np.argmax(relative))
        if relative[worst] > relative_tol:
            offenders = [
                f"{self.names[1 + i]} ({relative[i]:.2%})"
                for i in np.argsort(relative)[::-1][:5]
                if relative[i] > relative_tol
            ]
            raise ValueError(
                f"skeleton bone offsets differ by up to {relative[worst]:.2%} of "
                f"bone length, above the {relative_tol:.0%} tolerance. "
                f"Worst: {', '.join(offenders)}. These are different characters; "
                "retarget one onto the other rather than raising the tolerance."
            )

    def __repr__(self) -> str:
        return f"Skeleton({self.n_joints} joints, root={self.names[0]!r})"
