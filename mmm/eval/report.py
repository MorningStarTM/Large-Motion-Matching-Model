"""
Experiment reports: the comparison tables and figures for the thesis.

To implement
------------
- Collect metrics across runs into one table per section 9 experiment, with the
  KNN teacher and LMM as columns wherever they apply.
- The editability experiment gets its own report: the ten held out styles, each
  one's metrics with the style in the KB against with it absent, plus the
  assertion that no held out clip was in training.
- Record for every row the provenance that makes it reproducible: commit,
  config, checkpoint, database manifest.
- Emit both machine readable results and the formatted tables, so a number in
  the thesis can always be traced back to the run that produced it.
"""
