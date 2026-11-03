"""
ONNX export, for runtime timing.

To implement
------------
- Export the single frame recurrent step, not the parallel training path, since
  that is what runs in a game: inputs are the assembled input vector and the
  per layer conv buffers and SSM states, outputs are the pose and the updated
  states.
- Keep the state tensors as explicit inputs and outputs rather than hiding them
  inside the graph, so the caller owns per character state exactly as the C++
  engine does.
- Verify after export: the same 1000 frames through PyTorch and through the
  ONNX runtime must agree closely, before any timing number is quoted.
- Export the GRU and transformer baselines the same way, since the section 9
  cost comparison is only meaningful if all three are measured identically.
"""
