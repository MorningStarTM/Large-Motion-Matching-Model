"""
Jitter: high frequency noise in the motion.

To implement
------------
- jerk(joint_positions): the second difference | p_t - 2 p_{t-1} + p_{t-2} |,
  that is acceleration, matching the P_jerk reward term so training and
  evaluation measure the same quantity.
- Aggregate per joint as well as overall: raw pose output tends to jitter in the
  extremities first, and plan section 11 names this as a risk that the FK and
  velocity losses exist to control.
- Compare against the mocap data's own jitter level as the floor, since the
  target is motion as smooth as the source, not motion smoother than reality.
"""
