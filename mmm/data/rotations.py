"""
Rotation format conversions.

6D rotation, the first two columns of a rotation matrix, is the network facing
format: it is continuous, which is what neural networks need. Everything the
model reads or writes is 6D; everything on disk is euler or axis angle.

To implement
------------
- matrix_to_sixd / sixd_to_matrix, the latter re-orthonormalising with
  Gram Schmidt so a raw network output becomes a valid rotation.
- euler_to_matrix / matrix_to_euler with an explicit channel order argument,
  since BVH files differ.
- axis_angle_to_matrix, for the AMASS route where poses are axis angle.
- quaternion helpers, used by inertialization.
- All functions batched, shape preserving, and available in both NumPy and
  torch so the same code serves data preparation and the loss.

Round trip tests live in tests/test_rotations.py: format conversion bugs are
invisible until the character is inside out.
"""
