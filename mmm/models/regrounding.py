"""
Learned re grounding: the model decides when to retrieve.

Plan section 10.3, a novel contribution. Classic motion matching searches every
10 frames because that is a reasonable fixed guess. Here the model predicts
whether this frame should trigger a search, so re grounding happens when the
state actually needs it.

To implement
------------
- A small head on the backbone's hidden state producing a search probability
  per frame.
- A gating rule at runtime: search when the probability passes a threshold, with
  a minimum interval between searches so a KNN query cannot run every frame and
  blow the runtime budget.
- A cost term so searching is not free, otherwise the optimum is to search
  always; and a training signal, which can be supervised from when the teacher's
  chosen frame actually changed, or learned in Phase 3 where the reward already
  prices quality against the search budget.
- The fixed 10 frame interval stays available as the baseline this is measured
  against (section 9).
"""
