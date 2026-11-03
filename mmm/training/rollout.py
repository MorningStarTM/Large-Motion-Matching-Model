"""
Autoregressive rollout, shared by all three phases.

The model's own previous output is one of its inputs, so training and
evaluation both need a loop that feeds predictions back in while carrying the
Mamba hidden state. Keeping one implementation means Phase 1 drift, Phase 2
DAgger state collection and Phase 3 episodes cannot diverge in subtle ways.

To implement
------------
- rollout(policy, source, horizon, state): step the policy, feed its output
  pose back as the next frame's current pose, retrieve candidates on search
  frames, and carry and return the hidden state.
- Teacher forcing and scheduled sampling, selected by a schedule that moves
  from feeding ground truth to feeding the model's own predictions, which is the
  Phase 1 answer to drift (plan section 5.2).
- Retrieval policy during rollout: the fixed 10 frame interval, or the learned
  gate from models/regrounding.py, plus the leave one clip out restriction and
  retrieval dropout from the config.
- Differentiability: keep the graph across the horizon, for example 32 frames,
  for the direct gradient variant of Phase 3, and allow detaching for PPO and
  for long Phase 1 windows.
"""
