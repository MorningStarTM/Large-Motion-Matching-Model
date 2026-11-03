# MMM C++ inference engine (plan section 13).
#
# Build only after the PyTorch architecture is frozen.
#
# To fill in:
#   - C++17, the mmm_engine library from src/, the parity test from tests/, and
#     the optional pybind11 module from bindings/.
#   - Release flags: full optimization and native CPU instructions, GCC -O3 and
#     -march=native, applied only after the parity test passes.
#   - Optional nanoflann for KNN.
