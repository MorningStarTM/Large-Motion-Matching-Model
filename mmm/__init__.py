"""
Mamba Motion Matching (MMM).

Retrieval augmented state space model for real time character motion in games.

A single tiny model (Mamba backbone) takes the player's joystick input and
outputs a full body pose every frame, while the motion knowledge base (KB)
stays outside the weights so new clips become usable immediately with no
retraining.

Package map
-----------
data/      BVH, skeleton, rotations, FK, Matching/Animation databases.
teacher/   Phase 0: classic KNN motion matching. The teacher, not a learner.
models/    Linear in -> 4 Mamba layers -> linear out. No tokenizer.
training/  Phase 1 pretrain, Phase 2 distill (SFT + DAgger), Phase 3 RL polish.
envs/      Kinematic world and synthetic joystick for Phase 3.
viewer/    GenoViewPython / raylib integration.
eval/      Foot sliding, jitter, path error, timing, reports.
export/    ONNX export and the float32 weight blob for the C++ engine.

See docs/research-plan.md for the full plan; section numbers quoted in module
docstrings refer to it.
"""
