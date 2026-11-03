/*
 * Basic neural network operations for the MMM C++ inference engine.
 *
 * MMM needs no attention, KV cache, tokenizer or sampling, so this engine is a
 * few hundred lines: the same approach Holden used for LMM (the nnet.h file in
 * his Motion Matching repo).
 *
 * To implement:
 *   - linear(out, in, weight, bias): a plain matrix multiply, with matrices
 *     kept contiguous in the layout the multiply wants.
 *   - rms_norm, silu, softplus.
 *   - conv1d_step(buffer, in): one step of the depthwise causal convolution,
 *     advancing a buffer that holds the last 4 inputs.
 *   - Preallocated buffers only. No allocation per frame.
 *
 * Write this only after the PyTorch architecture is frozen, since every
 * architecture change means changing the engine too.
 */
