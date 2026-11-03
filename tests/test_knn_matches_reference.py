"""
The KNN teacher against Holden's reference behaviour.

To implement
------------
- The cKDTree backend and the brute force NumPy backend return the same top K
  on real data; brute force is the honest reference.
- Clip exclusion works: rows from an excluded clip never appear in results,
  which is what leave one clip out in Phase 1 relies on.
- Given a fixed database, a fixed joystick sequence and a fixed seed, the
  controller produces a deterministic sequence of chosen frames, so later
  changes to features or search are detected as a behaviour change rather than
  discovered in the viewer.
"""
