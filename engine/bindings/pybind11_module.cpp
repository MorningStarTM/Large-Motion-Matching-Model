/*
 * pybind11 bindings, so GenoViewPython can drive the C++ engine (plan section
 * 13.3 step 7).
 *
 * To implement:
 *   - Expose load_weights, create_state, reset_state and step.
 *   - Accept and return NumPy arrays without copying where possible.
 *   - Keep the Python facing interface identical in shape to MmmPolicy.step, so
 *     the viewer can switch between the PyTorch model and the engine and the
 *     two can be compared directly.
 *
 * The alternative deployment path is calling the engine directly from Holden's
 * C++ raylib demo.
 */
