/*
 * Engine entry point: the per frame loop and the parity harness.
 *
 * What runs per frame (plan section 13.1):
 *   - linear in: one matrix multiply
 *   - 4 Mamba layers in recurrent mode: small matrix multiplies plus element
 *     wise math
 *   - linear out: one matrix multiply
 *   - KNN search every 10 frames over the Matching Database
 *
 * To implement:
 *   - A frame loop reading the joystick, the previous pose and the retrieved
 *     candidates, and producing the next pose.
 *   - A harness that replays the 1000 exported reference frames for the parity
 *     test, before any optimization.
 *   - Timing output for 1, 10 and 100 characters.
 */
