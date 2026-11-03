"""
Prepare 100STYLE: 100 locomotion styles, over 4 million frames, CC BY 4.0.

Use the version already retargeted to Geno, which is the same skeleton as LaFAN1
resolved and therefore needs no retargeting work.

To implement
------------
- Load and validate all styles, process them exactly as LaFAN1 is processed, and
  record the style label per clip.
- Apply the split of plan section 7: 90 styles for training, 10 held out
  entirely. The held out ten must be written to a separate directory, not merely
  flagged, so that an accidental glob cannot pull them into a training run.
- Print the held out style names, so every run of this script restates what the
  main experiment depends on.
"""
