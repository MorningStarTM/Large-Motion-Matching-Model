"""
Small transformer backbone baseline, for the section 9 ablation: whether
attention's cost growing with history (plan section 4.5's argument for an
SSM in the first place) is visible in practice, not only in theory.

The backbone itself -- `GptBackbone`, a causal transformer with a KV cache --
is implemented once, in `mmm.models.gpt.backbone`, where it is explored as an
architecture candidate in its own right, not only as this ablation's control.
This module should import it from there rather than keep a second
implementation; the two questions (is attention worth its cost growing with
history / does attention capture this problem's structure better than a
recurrent state) share one answer.

To implement
------------
- Once `mmm.models.gpt.backbone.GptBackbone` exists, build this ablation on
  top of it: measure per frame cost at several context lengths (not one), so
  attention's cost actually growing with history is visible in the results
  table, the specific claim plan section 4.5 makes.
- No part of the C++ engine depends on this: the engine needs no attention, KV
  cache, tokenizer or sampling. This baseline stays in PyTorch.
"""
