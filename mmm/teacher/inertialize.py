"""
Inertialization: smooth blending when playback jumps between clips.

Without it, every 10 frame jump is a visible pop. With it, the offset between
the pose before the jump and the pose after it is decayed away over a short
blend time.

To implement
------------
- An inertializer that stores, per joint, the rotation offset and its velocity
  at the moment of transition, then decays both towards zero with a spring like
  half life.
- Separate handling for root velocity, so the character does not lurch.
- transition(current_pose, target_pose): called on a jump; step(): called every
  frame to apply the decaying offset to whatever the source is currently
  producing.
- The same inertializer serves the teacher and, at runtime, the residual
  variant of the model, so it lives here rather than inside the controller.
"""
