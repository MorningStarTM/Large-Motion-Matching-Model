"""
Foot sliding: the most visible animation bug, and the headline quality metric.

To implement
------------
- sliding(joint_positions, contacts, skeleton): horizontal displacement of a
  foot while it is labelled planted, reported as mean and as 95th percentile per
  sequence, since a rare bad transition matters more than the average.
- Use detected contacts rather than predicted ones when comparing methods, so
  that a model which simply predicts fewer contacts does not score better.
- Report per locomotion mode, walk, run, turn, stop, and separately on the hard
  joystick subset of sharp reversals and stop starts.
"""
