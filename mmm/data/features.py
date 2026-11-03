"""
Matching Database features: the short vector per frame used for searching.

Twenty seven numbers per frame for locomotion (plan section 3.1), all in
character space:

- future root positions at 20, 40 and 60 frames ahead     (3 x 2 = 6)
- future root facing directions at 20, 40 and 60 frames   (3 x 2 = 6)
- left and right foot positions                           (2 x 3 = 6)
- left and right foot velocities                           (2 x 3 = 6)
- hip velocity                                             (3)

To implement
------------
- compute_features(motion, skeleton) -> (n_frames, 27), with the trailing
  frames of a clip masked out because their future trajectory does not exist.
- Per dimension normalization (mean and standard deviation over the training
  set) plus the per feature weights that set how much each term counts in the
  KNN distance. Store both with the database so search and query agree.
- build_query(current_pose, desired_trajectory): the runtime counterpart, which
  takes the feet and hips from the character's current state and the future
  path from the joystick, and must produce a vector in exactly the same layout
  and normalization as the rows.
- joystick_signal(motion): the 12 number future root position and facing block
  used as the model's joystick input, taken from the data's own real future
  path during Phase 1 pretraining.

The layout is defined once here and imported everywhere else. A mismatch
between row layout and query layout is the single most likely cause of a
teacher that retrieves nonsense.
"""
