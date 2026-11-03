"""
The Phase 3 environment: a kinematic world and a synthetic joystick.

No physics. The character is moved by whatever pose the policy outputs, and the
reward measures whether that pose sequence looks right and follows the path.
Because the whole world is kinematic, every reward term is differentiable,
which is what makes the direct gradient variant of Phase 3 possible.
"""
