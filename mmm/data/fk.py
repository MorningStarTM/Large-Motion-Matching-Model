"""
Forward kinematics: chain local joint rotations into 3D joint positions.

FK is differentiable, so it sits inside the loss. The position and velocity
loss terms of plan section 6.2 are what make the error measurable where the eye
sees it rather than in rotation space.

To implement
------------
- forward_kinematics(skeleton, local_rotations, root_position) -> global joint
  positions and rotations, walking the joint tree in topological order.
- A torch implementation that keeps the graph intact, batched over
  (batch, time, joints), used by the losses.
- A NumPy implementation for data preparation and for the teacher, which does
  not need gradients.
- fk_character_space(): the variant the losses actually call, with the root
  transform removed so the comparison does not double count root error that
  L_root already covers.

Performance matters here: FK runs on every frame of every rollout in training,
so prefer batched matrix operations over per joint Python loops.
"""
