"""
KNN search over the Matching Database.

To implement
------------
- An index interface with two backends: scipy cKDTree, and brute force NumPy
  which is the honest reference and fast enough for locomotion. FAISS is an
  option if a dataset gets large.
- query(features, k) -> candidate row indices and distances, on the weighted
  and normalized feature vector, so the distance metric matches the one the
  features were built for.
- Exclusions: forbid rows from a given clip (leave one clip out for Phase 1),
  and optionally forbid rows very close to the currently playing frame so the
  search does not simply return where the character already is.
- Return the top K, about 4, not only the best: the top K candidates are the
  model's retrieved context in plan section 4.1, while the teacher's own
  playback uses K equal to 1.
- Keep the row to clip mapping attached to results, so candidates can be
  followed forward a frame.
"""
