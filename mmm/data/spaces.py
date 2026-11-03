"""
World, character and local space conversions.

- World space: relative to the game world.
- Character space: relative to the character's root, with position and facing
  removed. This is the model's input normalization: every feature and every
  pose the network sees is in character space.
- Local space: each joint relative to its parent, which is what BVH stores and
  what FK chains together.

To implement
------------
- root_transform(root_position, root_rotation): the ground plane transform,
  position plus facing direction only, no pitch or roll.
- to_character_space / from_character_space for points, directions and
  velocities, batched over frames.
- Root velocity extraction: translational velocity in character space plus
  rotational (turning) velocity, the quantities the model predicts.
- Integrating predicted root velocities back into a world space trajectory at
  runtime, the inverse of the above.
"""
