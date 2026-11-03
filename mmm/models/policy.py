"""
The assembled model: input assembly, backbone, head, and one frame of state.

This is what training, the environment and the exporter all talk to, so that
the exact input layout is defined in one place and cannot drift between the
training loop and the runtime.

To implement
------------
- MmmPolicy.build_input(joystick, previous_pose, candidates, candidate_mask):
  concatenate the three blocks of plan section 4.3 in a fixed documented order,
  in character space. Handle fewer than K candidates by masking, and apply
  retrieval dropout here, so the Phase 1 mechanism that forces retrieval use
  lives next to the layout it perturbs.
- step(obs, state) -> (pose, new_state): one frame. The hidden state carries
  over to the next frame.
- rollout(obs_sequence, state): the parallel path for training, equivalent to
  repeated step calls.
- Expose pose_dim, input_dim and the input block offsets as attributes, since
  the C++ engine and the ONNX export both need them.
- Keep the retrieved candidates as plain pose numbers: no encoder sits between
  the Animation Database and the model, which is what allows a clip appended to
  the KB to be usable immediately.
"""
