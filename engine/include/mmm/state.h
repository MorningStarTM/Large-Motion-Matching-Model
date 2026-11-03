/*
 * Per character runtime state.
 *
 * Each layer keeps two small memories per character between frames: a conv
 * buffer holding its last 4 inputs, and an SSM state of d_inner x d_state
 * numbers, 512 x 16. About 130 KB for 4 layers, so many NPCs can run at once.
 *
 * To implement:
 *   - A state struct owning both buffers for all layers, allocated once.
 *   - reset(), for a character entering the world or teleporting.
 *   - The character's own previous pose, which is one of the model's inputs.
 *   - No shared mutable state between characters, so the engine can be called
 *     from several threads.
 */
