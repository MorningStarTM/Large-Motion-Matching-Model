"""
The Mamba backbone: 4 layers at width 256, in two equivalent modes.

Parallel mode is used for training over long sequences. Recurrent mode is used
at runtime and is the form the C++ engine implements. The two must agree, which
tests/test_mamba_recurrent_equals_parallel.py enforces, because debugging that
disagreement in C++ is far more expensive than in Python.

To implement
------------
- MambaBackbone(d_model=256, n_layers=4): forward(x, state=None) handling both
  a full sequence and a single step, returning the updated state in the step
  case.
- Per layer recurrent state, exactly the two small memories the engine keeps
  per character (plan section 13.2): a conv buffer holding the last 4 inputs,
  and an SSM state of d_inner x d_state numbers, 512 x 16.
- The recurrent step, written to mirror the engine one to one:
    1. x = RMSNorm(h)
    2. split in_proj(x) into u and gate
    3. u = SiLU(conv1d_step(conv_buffer, u))
    4. split x_proj(u) into dt, B and C; dt = softplus(dt_proj(dt))
    5. s[i][n] = exp(dt[i] * A[i][n]) * s[i][n] + dt[i] * B[n] * u[i]
    6. y[i] = C . s[i] + D[i] * u[i], then y = y * SiLU(gate)
    7. h = h + out_proj(y)
- Use the reference mamba-ssm implementation for the parallel path where it is
  available, and keep a pure torch fallback so Phase 0 and early Phase 1 can run
  on CPU.
- State management across frames: detach and carry the hidden state between
  training windows, and reset it on episode boundaries.

Cost at width 256 and 4 layers is about 1 to 2 million multiply adds per frame,
similar to LMM's Stepper plus Decompressor, which the LMM paper measures at
about 80 to 200 microseconds per frame on one CPU core.
"""
