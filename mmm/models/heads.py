"""
Output heads: full pose, residual correction, and foot contacts.

To implement
------------
- FullPoseHead: the output layer's result is the next pose directly.
- ResidualHead (plan section 4.4, retrieve then refine): the output is a
  correction, and the final pose is the best KNN pose plus that correction.
  With a zero correction this is exactly classic motion matching, which makes
  the model stable from day one, leaves RL only small fixes to learn, and lets
  new KB clips flow straight through. The invariant is guarded by
  tests/test_residual_zero_equals_knn.py.
- ContactHead: foot contact probabilities for IK cleanup, optional, trained
  with the BCE term of plan section 6.2.
- Both pose heads expose the same interface so training and evaluation can
  switch between them from config, since full pose against residual is one of
  the section 9 experiments.
- Rotation outputs are re-orthonormalised through rotations.sixd_to_matrix
  before FK, in the head rather than in the loss, so evaluation and runtime do
  the same thing as training.
"""
