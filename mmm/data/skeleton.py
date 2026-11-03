"""
Skeleton definition: the hidden tree of joints that drives the character mesh.

To implement
------------
- Skeleton dataclass: joint names, parent index per joint, rest offsets
  (n_joints, 3), and the indices that matter downstream (hips, left and right
  foot, toes).
- GENO: the skeleton used by LaFAN1 resolved, 100STYLE retargeted and
  GenoViewPython. The single runtime skeleton: the KB has to be on it.
- SMPL_22: the 22 body joint skeleton used by the AMASS pretraining route, so
  retargeting and the skeleton specific adapter ablation have something to map
  from.
- Helpers: bone lengths from offsets, leg length (for the root translation
  scale when retargeting), topological order for FK, and a name based joint map
  between two skeletons.

Also validate a loaded BVH against the expected skeleton and fail loudly on a
mismatch, since a silently wrong joint order produces motion that looks almost
right.
"""
