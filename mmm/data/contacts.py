"""
Foot contact labelling.

A contact label says a foot is planted. Foot sliding, a planted foot moving
along the ground, is the most visible animation bug, so these labels drive both
the optional contact loss and the RL foot sliding penalty.

To implement
------------
- detect_contacts(joint_positions, skeleton) -> (n_frames, n_feet) booleans,
  from low foot height and low foot speed with hysteresis, so a foot does not
  flicker in and out of contact between frames.
- Per dataset thresholds, since capture setups differ in floor height and
  scale; expose them in the data config rather than hard coding.
- Short state removal: drop contacts and gaps below a few frames.
- A diagnostic that reports, per clip, the fraction of frames in contact and
  the sliding distance while planted, used to drop broken clips in the cleaning
  step.
"""
