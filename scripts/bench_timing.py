"""
Measure microseconds per frame and megabytes of memory.

Exported models only, never raw PyTorch, per plan section 11. Reports the model
cost, the KNN search cost on search frames, and the totals for 1, 10 and 100
characters, alongside LMM's published 80 to 200 microseconds per frame on one
CPU core.
"""
