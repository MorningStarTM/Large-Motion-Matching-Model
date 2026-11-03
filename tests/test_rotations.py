"""
Rotation conversion round trips.

To implement
------------
- matrix -> 6D -> matrix is the identity, and 6D -> matrix re-orthonormalises a
  perturbed, non orthogonal 6D vector into a valid rotation.
- euler -> matrix -> euler for every channel order used, and axis angle ->
  matrix against scipy as the reference.
- The NumPy and torch implementations agree, since the data pipeline uses one
  and the losses use the other.
"""
