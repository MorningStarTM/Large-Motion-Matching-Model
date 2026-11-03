"""
Metrics for the section 9 experiments.

The experiments, and what each one proves:

  hold out 10 styles, add them to the KB only      no retraining needed,
                                                   the main claim
  compare against KNN motion matching and LMM      quality on par
  microseconds per frame and MB of memory          fits a game budget
  Mamba against GRU against a small transformer    whether Mamba is needed
  full pose output against residual output         value of retrieve then refine
  with and without Phase 3; PPO against direct     RL improves on the teacher
  fixed 10 frame retrieval against learned timing  learned re grounding

Every metric here is computed identically for MMM, for the KNN teacher and, where
available, for LMM, so the comparison tables are honest.
"""
