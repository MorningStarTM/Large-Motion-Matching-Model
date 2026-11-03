"""
Dataset splits, including the held out styles that carry the main claim.

Plan section 7: train on LaFAN1 resolved plus 90 styles of 100STYLE, and keep
10 styles fully held out. Those ten enter neither pretraining nor the KB until
the editability experiment runs.

To implement
------------
- The named split as data: which 100STYLE styles are held out, chosen once,
  written down, and never silently changed.
- make_splits(datasets, config) -> train / validation / heldout clip lists,
  deterministic under a seed.
- leave_one_clip_out(): the retrieval restriction used in Phase 1, which forbids
  retrieving from the sequence's own clip so the model cannot trivially copy
  itself.
- An assertion used by the training entry points that no held out style clip
  reached the training set or the KB. This guard is worth more than the
  experiment it protects: a leak here invalidates the headline result.
"""
