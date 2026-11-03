"""
Spring damper: raw joystick input to a smooth desired path.

To implement
------------
- A critically damped spring damper on the desired velocity and on the desired
  facing direction, with a half life parameter rather than raw stiffness, which
  is the form Holden's Spring Roll Call uses and is far easier to tune.
- predict_trajectory(state, frame_time, horizons=(20, 40, 60)): integrate the
  smoothed velocity forward to produce the future root positions and facing
  directions the query needs, in character space.
- Separate handling of facing from movement direction, so the character can
  strafe and can turn on the spot.
- The same code must serve the teacher's query and the model's joystick input,
  so that at runtime both see the identical 12 numbers.
"""
