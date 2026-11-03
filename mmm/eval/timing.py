"""
Runtime cost: microseconds per frame and megabytes of memory.

A game has about 16 ms per frame for everything, and animation per character
gets microseconds. Plan section 11 warns that Python timing is misleading, so
measurement here is on exported models only.

To implement
------------
- Time the ONNX exported model, and the C++ engine once it exists, never raw
  PyTorch, on one CPU core.
- Separate the per frame model cost from the KNN search cost, which is paid only
  on search frames, and report both the per frame average and the search frame
  spike, because the spike is what blows a frame budget.
- Scale: cost for 1, 10 and 100 characters, which the C++ engine makes
  meaningful since per character state is only about 130 KB for 4 layers.
- Memory: model weights plus the resident Animation Database, reported honestly
  against LMM's 5.3 MB and classic motion matching's 52 MB for locomotion. MMM
  is expected to be larger than LMM here; the selling point is editability, not
  memory.
"""
