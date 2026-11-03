"""
Retarget SMPL motion onto the Geno skeleton.

To implement
------------
- A joint name map: pelvis to Hips, left knee to LeftLeg, and so on, declared
  explicitly rather than inferred.
- Rest pose correction: the two skeletons do not share a rest pose, so each
  mapped rotation needs the difference between the two rest orientations
  applied before it means the same thing on Geno.
- Scale root translation by the leg length ratio of the two skeletons, so stride
  length matches the new body.
- Fix planted feet with inverse kinematics: find contacts on the source motion,
  then pin the Geno foot to the contact point so retargeting does not introduce
  sliding.
- Check results visually in GenoViewPython, and numerically by comparing foot
  sliding and jitter statistics against LaFAN1 clips.

Plan section 15.3 offers an alternative to all of this: pretrain the backbone on
the SMPL skeleton with its own linear input and output layers and swap in Geno
layers afterwards. That route is an ablation, implemented in models/io_layers.py,
not here.
"""
