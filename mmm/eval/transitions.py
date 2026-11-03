"""
Transition smoothness: what happens at the moment playback jumps.

To implement
------------
- Locate transitions: search frames where the teacher jumped, and for MMM the
  frames where the retrieved context changed.
- Measure the discontinuity at those frames, in pose, in velocity and in
  acceleration, against the surrounding frames, which is what a visible pop is.
- Compare the teacher with inertialization, MMM full pose, and MMM residual,
  since this metric is where retrieve then refine should either pay off or show
  its cost.
- Report the worst transitions with their frame indices, so they can be replayed
  in the viewer rather than only counted.
"""
