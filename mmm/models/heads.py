"""
Output heads: full pose and residual correction.

Both expose the same interface, so training and evaluation switch between them
from config -- full pose against residual is one of the plan section 9
experiments.

The residual head is plan section 4.4, retrieve then refine. Per the build
phases document it is a Phase 6 extension rather than part of the core model,
so it is implemented here but is not the default; `tests/
test_residual_zero_equals_knn.py` guards its defining invariant from the start,
because an invariant that is only checked once the feature is "really" needed
tends to have been broken long before.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmm.models.io_layers import IoSpec

__all__ = ["FullPoseHead", "ResidualHead"]


class FullPoseHead(nn.Module):
    """The output layer's result is the next pose directly."""

    def __init__(self, spec: IoSpec):
        super().__init__()
        self.spec = spec

    def forward(self, raw_pose, candidates=None, candidate_mask=None):
        return raw_pose


class ResidualHead(nn.Module):
    """`final pose = best retrieved pose + correction` (plan section 4.4).

    With a zero correction this is exactly classic motion matching, which is
    what makes the variant stable from day one, leaves RL only small fixes to
    learn, and lets a clip newly appended to the KB flow straight through.

    The base is the *first* candidate, which the search returns nearest first.
    When no candidate is valid -- under retrieval dropout, or at a frame where
    every candidate was excluded -- the base is zero and the head degenerates
    to full pose output. That is the honest behaviour: there is nothing to
    refine, so the correction has to carry the whole pose.
    """

    def __init__(self, spec: IoSpec):
        super().__init__()
        self.spec = spec

    def forward(self, raw_pose, candidates=None, candidate_mask=None):
        if candidates is None:
            return raw_pose

        base = candidates[..., 0, :]
        if candidate_mask is not None:
            base = base * candidate_mask[..., 0:1]
        return base + raw_pose

    @staticmethod
    def correction(final_pose, candidates, candidate_mask=None):
        """Recover the correction from a final pose, for the `L_res` term.

        Computed rather than returned alongside the pose so the head keeps the
        same single-tensor interface as `FullPoseHead` and nothing downstream
        needs to know which head is in use.
        """
        base = candidates[..., 0, :]
        if candidate_mask is not None:
            base = base * candidate_mask[..., 0:1]
        return final_pose - base
