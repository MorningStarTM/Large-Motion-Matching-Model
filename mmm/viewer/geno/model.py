"""
The Geno character: loading the mesh, and posing it from our motion.

`Geno.bin` is Daniel Holden's skinned character, exported from Maya by
`resources/export_geno.py` in GenoView. It is the same rig LaFAN1 was resolved
onto, which is the fact that makes this whole thing cheap: all 24 of our
locomotion joints are present in its 75 bones **by name**, and not one of them
has a parent that we dropped. So our local rotations are already Geno's local
rotations. No retargeting, no bone mapping table to hand-write, no rest-pose
correction.

Verified rather than assumed -- `tests/test_geno_model.py` asserts both of
those properties against the real file.

Deliberately split in two
-------------------------
`GenoMesh` and every pose computation here are plain NumPy and import no
raylib. `GenoModel` is the thin part that uploads to the GPU. A renderer cannot
be unit tested without a window, but the skinning matrices can, and they are
where the bugs live: a transposed rotation or a wrong quaternion order
produces a character that is inside out or exploded, and finding that by
staring at a window is miserable.

Units
-----
`Geno.bin` is in metres; LaFAN1 is in centimetres. The conversion happens in
exactly one place, `METRES_PER_UNIT`, and the rest of the file works in the
mesh's own units.

File format
-----------
Little endian throughout, as written by `export_geno.py`::

    u32   vertex_count
    u32   triangle_count
    u32   bone_count
    f32   [vertex_count][3]   positions, metres
    f32   [vertex_count][2]   texture coordinates
    f32   [vertex_count][3]   normals
    u8    [vertex_count][4]   bone indices
    f32   [vertex_count][4]   bone weights, summing to 1
    u16   [triangle_count][3] triangle indices
    repeated bone_count times:  char[32] name, i32 parent
    repeated bone_count times:  f32 position[3], rotation[4] xyzw, scale[3]

Note the quaternion order: the exporter writes **xyzw**, while
`mmm.data.rotations` uses **wxyz**. That reorder is done once, at load.

Licensing
---------
`Geno.bin` and the shaders beside it are Daniel Holden's, MIT licensed. The
copy in `resources/` and the attribution are recorded in `NOTICE.md` and
`licenses/GenoViewPython-LICENSE`. This module is our own code reading his
format.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

from mmm.data.rotations import quat_to_matrix

__all__ = [
    "GenoMesh",
    "BoneMapping",
    "load_geno_mesh",
    "default_resource_dir",
    "default_model_path",
    "BindReference",
    "load_bind_reference",
    "skinning_matrices_for_bones",
    "default_bind_path",
    "skinning_matrices",
    "METRES_PER_UNIT",
    "GENO_BONE_COUNT",
]

# LaFAN1 is in centimetres, Geno.bin in metres. One constant, one direction:
# multiply a centimetre quantity by this to get the mesh's units.
METRES_PER_UNIT = 0.01

# What the file is expected to contain. Checked on load rather than trusted,
# because a truncated or mis-exported file otherwise shows up as a garbled
# mesh rather than as an error.
GENO_BONE_COUNT = 75

_NAME_BYTES = 32


def default_resource_dir() -> str:
    """Where the shaders and the character live, inside this package."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def default_model_path() -> str:
    return os.path.join(default_resource_dir(), "Geno.bin")


@dataclass
class GenoMesh:
    """The character, as plain arrays.

    Attributes:
        positions: `(n_vertices, 3)` bind pose vertex positions, metres.
        texcoords: `(n_vertices, 2)`.
        normals: `(n_vertices, 3)` bind pose normals.
        bone_ids: `(n_vertices, 4)` uint8, which bones influence each vertex.
        bone_weights: `(n_vertices, 4)` float32, summing to 1.
        triangles: `(n_triangles, 3)` uint16 indices.
        bone_names: the 75 bone names, in file order.
        bone_parents: `(n_bones,)` int32, -1 at the root.
        bind_positions: `(n_bones, 3)` **global** bind positions, metres.
        bind_rotations: `(n_bones, 3, 3)` **global** bind rotations.
    """

    positions: np.ndarray
    texcoords: np.ndarray
    normals: np.ndarray
    bone_ids: np.ndarray
    bone_weights: np.ndarray
    triangles: np.ndarray
    bone_names: list
    bone_parents: np.ndarray
    bind_positions: np.ndarray
    bind_rotations: np.ndarray

    @property
    def n_vertices(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def n_bones(self) -> int:
        return len(self.bone_names)

    def index(self, name: str) -> int:
        try:
            return self.bone_names.index(name)
        except ValueError:
            raise KeyError(
                f"bone {name!r} is not in the Geno skeleton. "
                f"Available: {self.bone_names}"
            ) from None

    def __repr__(self) -> str:
        return (
            f"GenoMesh({self.n_vertices:,} vertices, {self.n_triangles:,} "
            f"triangles, {self.n_bones} bones)"
        )


def load_geno_mesh(path: str | None = None) -> GenoMesh:
    """Read `Geno.bin`.

    Validated as it goes. The checks are cheap and each one corresponds to a
    failure that is otherwise silent and visually baffling: weights that do not
    sum to 1 shrink the character toward the origin, a bone index past the end
    reads whatever matrix happens to be next in memory, and a file with
    trailing bytes means the layout assumed here is not the layout written.
    """
    path = path or default_model_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. The Geno character is Daniel Holden's, from "
            "GenoView; copy resources/Geno.bin from a GenoViewPython checkout "
            "into mmm/viewer/geno/resources/."
        )

    with open(path, "rb") as handle:
        n_vertices, n_triangles, n_bones = struct.unpack("<III", handle.read(12))

        def block(count, dtype, columns):
            raw = handle.read(count * columns * np.dtype(dtype).itemsize)
            array = np.frombuffer(raw, dtype).reshape(count, columns)
            # A copy, because frombuffer is read-only and callers normalise in
            # place.
            return np.array(array)

        positions = block(n_vertices, np.float32, 3)
        texcoords = block(n_vertices, np.float32, 2)
        normals = block(n_vertices, np.float32, 3)
        bone_ids = block(n_vertices, np.uint8, 4)
        bone_weights = block(n_vertices, np.float32, 4)
        triangles = block(n_triangles, np.uint16, 3)

        names, parents = [], []
        for _ in range(n_bones):
            raw_name, parent = struct.unpack(f"<{_NAME_BYTES}si", handle.read(36))
            names.append(raw_name.split(b"\0")[0].decode("ascii"))
            parents.append(parent)

        transforms = block(n_bones, np.float32, 10)
        trailing = handle.read()

    if trailing:
        raise ValueError(
            f"{path}: {len(trailing)} unread bytes at the end. The layout this "
            "module assumes does not match the file."
        )
    if n_bones != GENO_BONE_COUNT:
        raise ValueError(f"{path}: expected {GENO_BONE_COUNT} bones, found {n_bones}")
    if int(bone_ids.max()) >= n_bones:
        raise ValueError(
            f"{path}: a vertex references bone {int(bone_ids.max())} but there "
            f"are only {n_bones}"
        )
    sums = bone_weights.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise ValueError(
            f"{path}: bone weights do not sum to 1 "
            f"(min {sums.min():.4f}, max {sums.max():.4f})"
        )

    parents = np.asarray(parents, np.int32)
    if parents[0] != -1:
        raise ValueError(f"{path}: bone 0 must be the root")
    if np.any(parents[1:] >= np.arange(1, n_bones)):
        raise ValueError(
            f"{path}: bones are not topologically sorted; forward kinematics "
            "here relies on every parent preceding its child"
        )

    bind_positions = transforms[:, 0:3].copy()
    # The exporter writes xyzw; mmm.data.rotations is wxyz. Reordered once,
    # here, so nothing downstream has to remember.
    xyzw = transforms[:, 3:7]
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, 0:3]], axis=-1)
    bind_rotations = quat_to_matrix(wxyz).astype(np.float32)

    scales = transforms[:, 7:10]
    if not np.allclose(scales, 1.0, atol=1e-5):
        raise ValueError(
            f"{path}: bone scales are not all 1, which the skinning here does "
            "not handle"
        )

    return GenoMesh(
        positions=positions,
        texcoords=texcoords,
        normals=normals,
        bone_ids=bone_ids,
        bone_weights=bone_weights,
        triangles=triangles,
        bone_names=names,
        bone_parents=parents,
        bind_positions=bind_positions,
        bind_rotations=bind_rotations,
    )


# ---------------------------------------------------------------------------
# mapping our skeleton onto Geno's
# ---------------------------------------------------------------------------

@dataclass
class BoneMapping:
    """Which Geno bone each of our joints drives.

    Attributes:
        geno_index: `(n_ours,)` the Geno bone each of our joints maps to.
        driven: `(n_geno,)` bool, True where a Geno bone is animated by us.
        source: `(n_geno,)` for a driven bone, the index of our joint that
            drives it; -1 elsewhere.
    """

    geno_index: np.ndarray
    driven: np.ndarray
    source: np.ndarray

    @classmethod
    def by_name(cls, mesh: GenoMesh, our_names) -> "BoneMapping":
        """Match by name, which works because it is the same rig.

        Raises rather than skipping an unmatched joint. A silently dropped
        joint leaves that limb frozen in the bind pose, which reads as a
        stiff-armed character rather than as an error -- exactly the kind of
        thing that gets debugged for an hour.
        """
        missing = [n for n in our_names if n not in mesh.bone_names]
        if missing:
            raise KeyError(
                f"these joints are not in the Geno skeleton: {missing}. The "
                "character and the motion are not the same rig."
            )

        geno_index = np.array([mesh.index(n) for n in our_names], np.int32)
        driven = np.zeros(mesh.n_bones, bool)
        source = np.full(mesh.n_bones, -1, np.int32)
        driven[geno_index] = True
        source[geno_index] = np.arange(len(our_names), dtype=np.int32)
        return cls(geno_index=geno_index, driven=driven, source=source)

    def __len__(self) -> int:
        return len(self.geno_index)


def default_bind_path() -> str:
    """The bind pose, as a BVH on the LaFAN1 skeleton."""
    return os.path.join(default_resource_dir(), "Geno_bind.bvh")


@dataclass
class BindReference:
    """What the mesh is bound to, expressed in *our* convention.

    The problem this solves, stated plainly: the mesh is bound to Geno's bind
    pose, where the arms are out to the side, while our BVH rest pose -- the
    pose you get with every local rotation set to identity -- has the arms
    straight up. Measured, the two differ by up to 97 cm at the hands. The
    bone *lengths* agree to a millimetre, so it is the same skeleton in two
    different poses, but feeding our rotations in as though the rest poses
    matched puts the character's arms a metre from where they belong.

    `Geno_bind.bvh`, which ships with GenoView, is that bind pose written as a
    BVH on the LaFAN1 skeleton. Running our own forward kinematics over it
    gives the bind in exactly the convention our animation uses, and it
    reproduces `Geno.bin`'s stored bind positions to 0.0001 cm -- so it really
    is the same pose, not an approximation.

    Attributes:
        positions: `(n_ours, 3)` world bind positions, metres.
        rotations: `(n_ours, 3, 3)` world bind rotations, our convention.
        ancestor: `(n_bones,)` for every Geno bone, the index into our joints
            of the nearest driven ancestor. See `skinning_matrices_for_bones`.
    """

    positions: np.ndarray
    rotations: np.ndarray
    ancestor: np.ndarray


def load_bind_reference(
    mesh: GenoMesh,
    mapping: BoneMapping,
    joint_names=None,
    path: str | None = None,
    tolerance: float = 0.01,
) -> BindReference:
    """Build the bind reference from `Geno_bind.bvh`.

    Checked against the bind positions stored in `Geno.bin` rather than
    trusted. If the two ever disagree the mesh and the reference are different
    poses, and the character would be skinned into a shape that is wrong
    everywhere but obviously broken nowhere.

    Args:
        mesh: the character.
        mapping: our joints to Geno bones.
        joint_names: our joint order; defaults to the locomotion subset.
        path: the bind BVH.
        tolerance: how far, in centimetres, the two bind poses may differ.
    """
    from mmm.data import bvh as bvh_loader
    from mmm.data.fk import forward_kinematics
    from mmm.data.skeleton import LOCOMOTION_JOINTS

    joint_names = list(joint_names or LOCOMOTION_JOINTS)
    path = path or default_bind_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Geno_bind.bvh ships with GenoViewPython and is "
            "what reconciles its bind pose with the BVH rest pose; without it "
            "the character cannot be posed correctly."
        )

    motion = bvh_loader.load(path).subset(joint_names)
    positions, rotations = forward_kinematics(
        motion.skeleton, motion.rotations[:1], motion.root_positions[:1]
    )
    positions, rotations = positions[0], rotations[0]

    stored = mesh.bind_positions[mapping.geno_index] / METRES_PER_UNIT
    difference = np.linalg.norm(positions - stored, axis=-1)
    if difference.max() > tolerance:
        worst = joint_names[int(np.argmax(difference))]
        raise ValueError(
            f"{path} does not match the bind pose stored in Geno.bin: up to "
            f"{difference.max():.3f} cm, worst at {worst}. These are different "
            "poses, and skinning with them would deform the character."
        )

    # For every Geno bone, the nearest ancestor we actually animate. The
    # fingers and the `*End` tips have none of their own, and they must follow
    # the bone they hang off rigidly.
    ancestor = np.zeros(mesh.n_bones, np.int32)
    for bone in range(mesh.n_bones):
        walk = bone
        while walk >= 0 and not mapping.driven[walk]:
            walk = int(mesh.bone_parents[walk])
        ancestor[bone] = int(mapping.source[walk]) if walk >= 0 else 0

    return BindReference(
        positions=(positions * METRES_PER_UNIT).astype(np.float32),
        rotations=rotations.astype(np.float32),
        ancestor=ancestor,
    )


def skinning_matrices_for_bones(
    reference: BindReference,
    pose_positions: np.ndarray,
    pose_rotations: np.ndarray,
) -> np.ndarray:
    """Skinning matrices for all 75 bones, from our 24 posed joints.

    The transform that takes a bone from its bind pose to its posed pose is
    `A . B^-1`, and because both are expressed in the same convention here --
    ours, via `load_bind_reference` -- it reduces to a rotation of
    `G Gb^T` and a translation of `P - G Gb^T Pb`.

    The undriven bones are handled by an identity that is worth seeing rather
    than special-casing. A finger is rigidly attached to the hand, so its
    posed transform is `A_hand . B_hand^-1 . B_finger`, and its skinning
    matrix is therefore `A_hand . B_hand^-1 . B_finger . B_finger^-1`, which
    is just the hand's own skinning matrix. **A bone that is not animated
    shares the matrix of its nearest driven ancestor**, exactly. So the 75
    matrices are a gather from the 24, with no extra forward kinematics at all.

    Args:
        reference: from `load_bind_reference`.
        pose_positions: `(n_ours, 3)` world joint positions, **metres**.
        pose_rotations: `(n_ours, 3, 3)` world joint rotations, our
            convention -- both straight out of `mmm.data.fk.forward_kinematics`.

    Returns:
        `(n_bones, 4, 4)` row-major.
    """
    pose_positions = np.asarray(pose_positions, np.float32)
    pose_rotations = np.asarray(pose_rotations, np.float32)

    rotation = pose_rotations @ np.swapaxes(reference.rotations, -1, -2)
    translation = pose_positions - np.einsum(
        "jik,jk->ji", rotation, reference.positions
    )

    joints = np.zeros((len(pose_positions), 4, 4), np.float32)
    joints[:, :3, :3] = rotation
    joints[:, :3, 3] = translation
    joints[:, 3, 3] = 1.0
    return joints[reference.ancestor]


def skinning_matrices(
    bind_positions: np.ndarray,
    bind_rotations: np.ndarray,
    pose_positions: np.ndarray,
    pose_rotations: np.ndarray,
) -> np.ndarray:
    """The matrix that takes each bind-pose vertex to its posed position.

    For a bone with global bind transform `(Pb, Rb)` and global posed transform
    `(Pa, Ra)`, the skinning matrix is the posed transform composed with the
    inverse of the bind transform::

        M = T(Pa) R(Ra) . [T(Pb) R(Rb)]^-1

    which multiplies out to a rotation of `Ra Rb^T` and a translation of
    `Pa - Ra Rb^T Pb`. In the bind pose it is the identity, by construction --
    which `tests/test_geno_model.py` checks, because it is the one assertion
    that catches a transposed rotation, a swapped multiplication order and a
    wrong quaternion convention all at once.

    Returns:
        `(n_bones, 4, 4)` row-major, ready to hand to the vertex shader.
    """
    rotation = pose_rotations @ np.swapaxes(bind_rotations, -1, -2)
    translation = pose_positions - np.einsum("bij,bj->bi", rotation, bind_positions)

    matrices = np.zeros((len(bind_positions), 4, 4), np.float32)
    matrices[:, :3, :3] = rotation
    matrices[:, :3, 3] = translation
    matrices[:, 3, 3] = 1.0
    return matrices


def skin_vertices(mesh: GenoMesh, matrices: np.ndarray) -> np.ndarray:
    """Apply the skinning matrices on the CPU.

    The GPU does this in `skinnedBasic.vs` at runtime; this exists so a test
    can check that a posed mesh is the shape it should be without a window, and
    so a headless render can be produced at all.
    """
    homogeneous = np.concatenate(
        [mesh.positions, np.ones((mesh.n_vertices, 1), np.float32)], axis=-1
    )
    # (v, 4 influences, 4x4) applied to (v, 4) then weighted and summed.
    per_influence = np.einsum(
        "vkij,vj->vki", matrices[mesh.bone_ids], homogeneous
    )
    return np.einsum("vki,vk->vi", per_influence, mesh.bone_weights)[:, :3]
