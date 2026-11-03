"""
AMASS pretraining pipeline (plan section 15). Optional, Phase 1 only.

AMASS clips are SMPL+H body model parameters in .npz files, not BVH. This
subpackage converts them to BVH on the SMPL skeleton and retargets them onto
Geno, so AMASS can sit alongside LaFAN1 resolved and 100STYLE in pretraining.

Phases 2 and 3 stay on Geno skeleton data, since the KNN teacher and the KB run
on Geno.

Order: load_npz -> resample -> smpl_fk -> smpl2bvh -> retarget, then the normal
contact labelling and feature computation.

Licence: AMASS is for noncommercial research and each subset keeps its own
licence. Record the licence and citation of every subset used.
"""
