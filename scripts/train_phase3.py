"""
Phase 3 entry point: RL polish with KNN in the loop.

Thin wrapper over mmm.training.phase3_rl. Starts from a Phase 2 checkpoint,
selects PPO or the direct gradient variant from config, and renders rollouts
periodically for the visual inspection the plan requires.
"""
