/*
 * One recurrent Mamba layer step (plan section 13.2).
 *
 * The recurrent form of the same math used in training, per layer:
 *   1. x = RMSNorm(h)
 *   2. split in_proj(x) into u and gate
 *   3. u = SiLU(conv1d_step(conv_buffer, u))
 *   4. split x_proj(u) into dt, B and C; dt = softplus(dt_proj(dt))
 *   5. s[i][n] = exp(dt[i] * A[i][n]) * s[i][n] + dt[i] * B[n] * u[i]
 *   6. y[i] = C . s[i] + D[i] * u[i], then y = y elementwise* SiLU(gate)
 *   7. h = h + out_proj(y)
 *
 * Cost for width 256 and 4 layers is about 1 to 2 million multiply adds per
 * frame, expected to run in tens of microseconds on one CPU core.
 *
 * This file and mmm/models/mamba_backbone.py must stay in step: the parity test
 * in tests/ is what proves they are.
 */
