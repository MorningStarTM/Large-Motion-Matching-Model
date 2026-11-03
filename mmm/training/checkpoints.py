"""
Checkpoints and run metadata.

To implement
------------
- save / load of model weights, optimizer state, the schedule position and the
  phase.
- Record with every checkpoint what is needed to reproduce and to trust it: the
  config, the git commit, the database manifest hash, the split identity
  including which ten styles are held out, and the phase lineage, that is which
  checkpoint this run started from.
- A guard that refuses to load a checkpoint into an incompatible
  architecture, comparing pose_dim, input_dim, K and the backbone
  configuration, since the three phases run as separate processes days apart.
- Export the frozen architecture description that export/ and the C++ engine
  read, so the engine is built against an explicit frozen contract.
"""
