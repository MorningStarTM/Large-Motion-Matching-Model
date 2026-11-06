"""
Environment aware motion matching: obstacle geometry, and the extra matching
features that let the search see it.

This package is additive. Nothing in `mmm.data`, `mmm.teacher` or `mmm.models`
is modified by it: the base 27 feature vector in `mmm/data/features.py` and the
151 wide pose in `mmm/data/database.py` keep their layouts exactly, and the
obstacle features are appended alongside rather than inserted. A database built
without obstacles searches exactly as it did before.

The three pieces, in the order they run:

1. `geometry`: boxes, and the queries the features ask of them.
2. `recover`: gets the boxes back out of the mocap, since LaFAN1's obstacle
   clips were captured with real props that the BVH does not record.
3. `features`: the per frame obstacle block, and the matching query to go with
   it, so KNN can retrieve a vault when there is something to vault.
"""

from mmm.obstacles.geometry import Box, BoxField

__all__ = ["Box", "BoxField"]
