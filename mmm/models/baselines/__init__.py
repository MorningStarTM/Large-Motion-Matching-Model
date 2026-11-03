"""
Backbone baselines for the section 9 ablation: Mamba against GRU against a
small transformer.

The GRU is not optional. At runtime Mamba behaves like a strong RNN, so the
plan requires a small GRU baseline to establish whether Mamba is actually
needed. The transformer is included to show what attention costs when history
grows, which is the argument for an SSM in the first place.

All three expose the same interface as MambaBackbone, parallel and recurrent,
so only the config changes between runs.
"""
