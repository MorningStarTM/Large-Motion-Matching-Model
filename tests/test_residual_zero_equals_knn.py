"""
The residual variant invariant (plan section 4.4).

With a zero correction, the residual model's output must equal the best KNN
pose exactly, that is classic motion matching. This is what makes the variant
stable from day one and what lets a new KB clip flow straight through.

To implement
------------
- Force the residual head's output to zero and require the pose to be bit for
  bit the retrieved pose, through the whole policy including rotation
  re-orthonormalisation, not only at the head's boundary.
- The same over a rollout with transitions, so inertialization does not break
  the invariant.
- A freshly initialised residual head produces small corrections, so a Phase 3
  run genuinely starts near the teacher.
"""
