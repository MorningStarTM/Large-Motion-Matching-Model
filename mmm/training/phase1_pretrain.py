"""
Phase 1: pretrain on raw mocap. Learns how bodies move.

Task (plan section 5.2): next pose prediction over long sequences. The joystick
signal is taken from the data's own real future root path, and the retrieved
candidates are included as input from the start, so the model learns to use
retrieval rather than learning to work without it and having it bolted on later.

To implement
------------
- Supervised training over long windows with the losses of section 6.2, target
  y_t being the mocap data.
- Force use of retrieval, all three mechanisms:
    leave one clip out, never retrieving from the sequence's own clip;
    retrieval dropout, so the model must cope with missing candidates;
    whole 100STYLE styles held out.
  Without these the model can memorise and ignore retrieval, which is a named
  risk in plan section 11 and would quietly invalidate the main claim.
- Drift: teacher forcing first, then scheduled sampling, feeding the model's own
  predictions.
- Data: LaFAN1 resolved plus the 90 training styles of 100STYLE, and optionally
  AMASS through the section 15 pipeline.
- Validation that reports the eval metrics, not only the loss, and a retrieval
  ablation at validation time, evaluating with candidates zeroed, so the degree
  to which the model actually depends on retrieval is tracked throughout
  training rather than discovered at the end.
"""
