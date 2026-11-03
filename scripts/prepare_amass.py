"""
Prepare AMASS for Phase 1 pretraining. Optional path, plan section 15.

To implement
------------
- Run the pipeline in order: load_npz, resample to 60 fps and z up to y up,
  rest pose offsets from betas, write BVH on the SMPL skeleton, retarget to
  Geno, clean and label, compute features.
- Start with subsets rich in locomotion and everyday movement, for example CMU,
  KIT and BMLmovi, and record the licence and citation of every subset used.
- Clean: drop broken, jittery or very short clips, and clips with unrealistic
  root motion such as treadmill walking.
- Validate: render random clips next to LaFAN1 clips and compare foot sliding
  and jitter statistics, and report both rather than only writing files.
"""
