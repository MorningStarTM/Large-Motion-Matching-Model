/*
 * Parity against PyTorch (plan section 13.3 step 5).
 *
 * Feed the same 1000 frames to the engine and to PyTorch and check the outputs
 * match within about 1e-5. This happens before any optimization: once the
 * compiler flags and memory layout change, a mismatch becomes far harder to
 * attribute.
 *
 * To implement:
 *   - Load the reference frames and expected outputs written by
 *     scripts/export_weights.py.
 *   - Report the first diverging frame, the layer and the tensor, rather than
 *     only a pass or fail.
 *   - Re-run after every optimization step, including the int8 quantization
 *     experiment, with its own looser tolerance.
 */
