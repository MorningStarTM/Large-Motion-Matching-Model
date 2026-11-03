"""
The classic motion matching controller: the 10 frame loop.

To implement
------------
- KnnController.step(joystick_input) -> pose, running the loop:
    every 10 frames: build the query from the current feet and hips plus the
                     desired path, search, and if the best candidate is not the
                     continuation of the current frame, jump there and start an
                     inertialization transition;
    every frame:     advance one row of the Animation Database, apply the
                     decaying transition offset, return the pose.
- Expose the intermediate quantities the rest of the project needs, not only
  the final pose: the query, the top K candidate rows, the chosen row, and
  whether this frame was a search frame.
- reference_pose(state, joystick): a stateless query used by Phase 3, which
  asks the teacher what it would do at a state MMM reached, without disturbing
  the teacher's own playback. This is what the P_teach reward term compares
  against, and what DAgger labels with.
- Configurable search interval, so the learned re grounding experiment of plan
  section 10.3 has a fixed interval baseline to beat.
"""
