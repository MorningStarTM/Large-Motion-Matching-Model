"""
GRU backbone baseline. Required by plan section 4.5.

To implement
------------
- GruBackbone(d_model=256, n_layers=...) with the same forward(x, state)
  interface as MambaBackbone, sized to a comparable parameter and multiply add
  count per frame so the comparison is about the architecture rather than about
  capacity.
- The same hidden state carry over and reset behaviour.
- Timing measured the same way as Mamba, on an exported model, since the whole
  point of the comparison is cost against quality.
"""
