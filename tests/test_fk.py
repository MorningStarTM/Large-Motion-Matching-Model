"""
Forward kinematics correctness and differentiability.

To implement
------------
- A hand built two and three joint skeleton with known answers.
- The torch and NumPy implementations agree on real motion data.
- Gradients flow through FK and match a finite difference check, since the whole
  point of differentiable FK is that it sits inside a loss.
- Character space FK removes the root transform, verified by moving and rotating
  the root and requiring the output not to change.
"""
