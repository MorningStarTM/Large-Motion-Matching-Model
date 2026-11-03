"""
Export paths out of PyTorch.

Two separate consumers:

- onnx_export.py, for honest timing, because plan section 11 warns that Python
  timing is misleading.
- weights_bin.py, for the C++ engine of plan section 13, which loads one flat
  float32 file.

Both are written against the frozen architecture description that
training/checkpoints.py emits. Plan section 13 is explicit that the engine is
written only after the PyTorch architecture is frozen, since every architecture
change means changing the engine too.
"""
