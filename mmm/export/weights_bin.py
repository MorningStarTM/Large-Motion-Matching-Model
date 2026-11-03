"""
Write every weight matrix into one float32 binary for the C++ engine.

Plan section 13.3, step 1. This file and engine/src/weights.cpp are two halves
of one contract: a fixed order, agreed once and asserted on both sides.

To implement
------------
- A small header: a magic number, a format version, and the architecture
  parameters d_model, n_layers, d_inner, d_state, d_conv, pose_dim, input_dim
  and K, so the engine can validate rather than trust.
- Then every tensor in a fixed documented order, float32, contiguous, with the
  layout the engine's matrix multiplies want, which may be the transpose of
  PyTorch's.
- A manifest listing every tensor with its name, shape and byte offset, written
  alongside, which is what makes a mismatch debuggable.
- A dump of 1000 reference input frames and their PyTorch outputs, so the
  engine's parity test of plan section 13.3 step 5 has fixtures to check
  against within about 1e-5.
"""
