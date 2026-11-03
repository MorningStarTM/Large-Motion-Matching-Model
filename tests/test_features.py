"""
Matching Database features: layout, and row against query agreement.

To implement
------------
- The feature vector has the documented size and block layout, 27 for
  locomotion.
- The crucial test: a query built from a database frame's own state and its own
  real future path must retrieve that frame as its nearest neighbour. If row
  layout and query layout disagree even slightly, the teacher retrieves nonsense
  while still appearing to run.
- Trailing frames of a clip, whose future does not exist, are masked out and
  never retrieved.
- Normalization and feature weights are applied identically on both sides.
"""
