"""
Path following error: does the character go where the joystick asked.

To implement
------------
- error(root_trajectory, desired_trajectory): root position distance to the
  desired path point plus facing angle error, matching the P_path reward term.
- Report responsiveness separately from accuracy: lag between an input change
  and the character reacting, since a controller can track a path well and still
  feel unresponsive, which is what players notice.
- Evaluate on the held out hard joystick subset, where turns and reversals make
  the trade off between smoothness and responsiveness visible.
"""
