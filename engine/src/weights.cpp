/*
 * Load the exported weights (plan section 13.3 steps 1 and 2).
 *
 * Reads the single float32 binary written by mmm/export/weights_bin.py: a
 * header with a magic number, a format version and the architecture parameters,
 * then every tensor in a fixed order.
 *
 * To implement:
 *   - Validate the header against the engine's compiled expectations and refuse
 *     to run on a mismatch, rather than producing quietly wrong motion.
 *   - Load once at startup and preallocate every buffer, so no allocation
 *     happens per frame.
 *
 * This file and weights_bin.py are two halves of one contract.
 */
