"""
Small transformer backbone baseline.

To implement
------------
- A causal transformer with the same forward(x, state) interface, where state
  is a KV cache, over a bounded context window.
- Report explicitly what plan section 4.5 claims: per frame cost grows with
  history, where the SSM's does not. Measure cost at several context lengths
  rather than at one, so the growth is visible in the results table.
- No part of the C++ engine depends on this: the engine needs no attention, KV
  cache, tokenizer or sampling. This baseline stays in PyTorch.
"""
