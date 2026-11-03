"""
Teacher rollout logging. Set up on day one, per plan section 5.1.

Every query, the top K candidates and the chosen frame are recorded from the
first time the teacher runs. Phase 2 is built entirely out of these logs, so
logging that is added later means collecting the data twice.

To implement
------------
- A writer that appends per frame records, as one array per field in a single
  compressed .npz per rollout: joystick input, query vector, top K candidate
  rows and distances, chosen row, search frame flag, and the output pose.
- A reader that yields these as training sequences for SFT, able to stream
  rather than load everything, since hours of 60 Hz rollouts are large.
- A manifest per rollout: database manifest hash, controller config, joystick
  generator seed, so a rollout can be reproduced and so a dataset built against
  an old database is detected rather than silently reused.
"""
