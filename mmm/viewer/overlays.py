"""
Debug overlays. The main instrument for diagnosing both the teacher and MMM.

To implement
------------
- The desired path from the spring damper, and the future query points at 20, 40
  and 60 frames, drawn in the world.
- The retrieved candidates: the top K poses as ghosts, their distances, which
  one was chosen, and which clip each came from. Watching this is how a broken
  feature layout is spotted in seconds rather than days.
- Foot contact state and sliding, drawn at the feet, plus a jitter indicator.
- For the residual variant, the magnitude and direction of the correction, so it
  is visible whether the model is making small fixes or fighting the retrieval.
- A search frame marker and, for learned re grounding, the predicted search
  probability over time.
"""
