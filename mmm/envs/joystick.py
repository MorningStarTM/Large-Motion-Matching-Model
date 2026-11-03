"""
Synthetic joystick generator: turns, stops, sprints and reversals.

Used to drive the teacher for hours in Phase 2 and to drive episodes in Phase 3.
The distribution of inputs here decides what the model is good at, so it is a
deliberate design object rather than uniform noise.

To implement
------------
- A sampler over input episodes built from named primitives: hold a direction,
  turn by an angle, stop, start, sprint, reverse sharply, strafe, and circle,
  with durations drawn from configured ranges.
- Deterministic under a seed, and the seed recorded in the rollout manifest, so
  a Phase 2 dataset or a Phase 3 episode can be reproduced exactly.
- A hard case subset, sharp reversals and stop starts, kept aside for
  evaluation, since those are where motion matching transitions look worst.
- The same interface as the gamepad in viewer/gamepad.py, so the controller does
  not know whether a human or the generator is driving.
"""
