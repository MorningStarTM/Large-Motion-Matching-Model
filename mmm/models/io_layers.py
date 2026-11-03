"""
The trainable linear input and output layers.

To implement
------------
- InputLayer: one linear map from the concatenated input vector to width 256.
  The input vector is assembled by policy.py and is, per plan section 4.3:
    joystick         12        desired future root positions and facing at 20,
                              40 and 60 frames, in character space
    current pose     about 150 the model's own previous output: joint rotations
                              in 6D plus root velocity
    KNN candidates   K x 150  the top K retrieved poses, or their next frames,
                              from the Animation Database
- OutputLayer: one linear map from width 256 to pose size, where the pose is
  joint rotations in 6D plus root translational and rotational velocity, with
  optional foot contact logits.
- Skeleton specific adapters (plan section 15.3): the input and output layers
  are swappable while the backbone is kept, so the backbone can be pretrained on
  the SMPL skeleton and then fine tuned on Geno with fresh linear layers. Keep
  these layers free of any other state so that swap stays trivial.

These two layers are also the first and last matrix multiply of the C++ engine,
so their shapes and weight order are part of the export contract.
"""
