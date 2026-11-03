"""
Phase 2 entry point: SFT on teacher rollouts.

Thin wrapper over mmm.training.phase2_distill. Starts from a Phase 1 checkpoint
and trains the model to reproduce the KNN teacher. DAgger rounds are run by
dagger_round.py so that each round is a separate, inspectable step.
"""
