"""
Gamepad input.

To implement
------------
- Read the left stick as desired movement direction and magnitude and the right
  stick or the stick direction as desired facing, with a dead zone and a
  configurable response curve.
- Feed the raw input into teacher/trajectory.py's spring damper, which is what
  turns it into the smooth desired path the query and the model's joystick input
  are built from.
- Keyboard fallback, so the project is usable without a pad, with the caveat
  that the plan's own getting started step wants a real gamepad for the web demo.
- The same interface as envs/joystick.py.
"""
