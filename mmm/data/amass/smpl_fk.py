"""
Rest pose joint offsets for one subject, from betas.

Bone lengths come from the subject's body shape, so each AMASS subject needs
its own skeleton before a BVH can be written.

To implement
------------
- rest_joint_positions(betas, gender) -> (22, 3), by running the SMPL body
  model with those betas in rest pose. Requires the SMPL+H body model files
  downloaded from the AMASS site, and the smplx package.
- offsets_from_positions(positions, parents): each joint's offset is its
  position minus its parent's position, which is what the BVH HIERARCHY section
  stores.
- Cache per subject, since many clips share a subject.
"""
