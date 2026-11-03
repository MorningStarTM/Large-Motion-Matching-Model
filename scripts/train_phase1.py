"""
Phase 1 entry point: pretrain on raw mocap.

Thin wrapper over mmm.training.phase1_pretrain. Loads the config, asserts the
split is clean, builds the model and the databases, and runs training.
"""
