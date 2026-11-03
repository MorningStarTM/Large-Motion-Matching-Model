"""
Export a frozen checkpoint for the C++ engine.

Writes the float32 weight blob and its manifest through export/weights_bin.py,
plus the 1000 reference frames and their PyTorch outputs that the engine's
parity test checks against.

Run this only once the PyTorch architecture is frozen.
"""
