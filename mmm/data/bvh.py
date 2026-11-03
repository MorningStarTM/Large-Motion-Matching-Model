"""
BVH reading and writing. MMM's working format (plan section 14).

To implement
------------
- parse(path) -> BvhFile: parse the HIERARCHY section into a joint tree
  (names, parent indices, offsets, channel order per joint) and the MOTION
  section into an array of shape (n_frames, n_channels), plus the frame time.
- BvhFile.to_motion() -> Motion: resolve channels into root translation
  (n_frames, 3) and joint euler angles (n_frames, n_joints, 3), then hand the
  rotation conversion to rotations.py.
- write(path, skeleton, root_positions, euler_angles, frame_time): emit
  HIERARCHY (tree and offsets) then MOTION (one line per frame: root position
  followed by each joint's euler angles), so round tripping a file reproduces
  it.
- Tolerate the channel orders and End Site blocks found in LaFAN1 resolved and
  in 100STYLE, and resample to 60 fps when frame_time disagrees.

Notes
-----
Plain text and easy to parse, which is why the plan keeps it as the working
format. FBX import is a later addition so studios can drop their own clips into
the KB.
"""
