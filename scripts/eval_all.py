"""
Run the section 9 experiment suite and emit the comparison tables.

To implement
------------
- Evaluate MMM, the KNN teacher and where available LMM on the same sequences
  with the same metrics.
- Run the named experiments: held out styles editability, quality comparison,
  backbone ablation, full pose against residual, with and without Phase 3, and
  fixed against learned retrieval timing.
- Write results through eval/report.py with full provenance.
"""
