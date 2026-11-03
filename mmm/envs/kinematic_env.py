"""
The kinematic world used for Phase 3.

To implement
------------
- A gym style environment: reset() starts an episode from a random database
  frame and a fresh joystick, step(action) applies the pose, integrates the
  root velocity into a world trajectory, and returns the observation, reward and
  per term reward breakdown.
- Observation assembly through models/policy.build_input, so the environment and
  training cannot disagree about the input layout.
- KNN in the loop: every step also asks the teacher for its reference pose at
  the current state, which feeds P_teach. This is the single most important
  wiring in Phase 3.
- Retrieval on search frames, by fixed interval or by the learned gate.
- Episode termination on divergence, for example a root that has left the
  plausible region or a pose that has gone non finite, so a broken rollout does
  not poison a batch.
- A differentiable mode that keeps the torch graph across the horizon for the
  direct gradient variant, alongside the detached mode PPO uses.
"""
