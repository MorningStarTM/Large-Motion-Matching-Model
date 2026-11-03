"""
Collect Phase 2 SFT data: drive the teacher with synthetic joystick for hours.

Plan section 5.3 step 1.

To implement
------------
- Run the teacher headless under the synthetic joystick generator for a target
  number of frames, recording joystick, retrieved candidates and output poses
  through teacher/logger.py.
- Cover the input distribution deliberately: turns, stops, sprints and
  reversals, in the configured mix, and report the realised mix rather than
  assuming it.
- Shard the output and record the seed and database manifest per shard, so
  collection can be parallelised and resumed.
"""
