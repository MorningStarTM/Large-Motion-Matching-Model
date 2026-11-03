"""
Write an AMASS clip as BVH on the SMPL skeleton.

To implement
------------
- Convert axis angle rotations to euler angles with scipy, using
  Rotation.from_rotvec(r).as_euler("zyx", degrees=True), matching the channel
  order written into the HIERARCHY section.
- Write HIERARCHY from the per subject offsets of smpl_fk, then MOTION with one
  line per frame: root position followed by each joint's euler angles.
- Verify the written file by parsing it back and comparing its FK joint
  positions against the FK positions computed directly from the axis angle
  input. An existing open source smpl2bvh script can do this step; either way
  the output is checked, not trusted, and spot checked visually in
  GenoViewPython or Blender.
"""
