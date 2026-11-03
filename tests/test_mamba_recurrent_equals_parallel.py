"""
Mamba recurrent mode equals parallel mode.

Training runs the parallel path; the runtime and the C++ engine run the
recurrent step. If the two disagree, a model that trained well will behave
differently in the game, and the plan's section 13.3 step 5 parity check will
fail with no way to tell whether the engine or the architecture is at fault.

To implement
------------
- Run a sequence through the parallel path, then step the same sequence one
  frame at a time carrying conv buffers and SSM state, and require agreement to
  tight tolerance.
- Check state carry over across window boundaries: two consecutive windows with
  the state passed between them equal one long window.
- Check the conv buffer warm up at the start of a sequence, where the last four
  inputs do not yet exist.
"""
