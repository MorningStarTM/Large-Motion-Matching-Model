"""
GenoViewPython integration: the raylib viewer with the Geno character.

Plan section 8: fork GenoViewPython, add an upstream remote, and keep the venv
and the BVH files out of git. This package is the thin layer that drives that
viewer with either the teacher or the model, and that draws the debug overlays
needed to tell whether either is working.

The viewer is also the Phase 3 world, so what it draws and what
envs/kinematic_env.py simulates must be the same thing.
"""
