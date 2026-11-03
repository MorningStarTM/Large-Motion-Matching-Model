"""
Phase 3: RL polish with the KNN teacher in the loop. Learns to beat the teacher.

Plan section 5.4.

Environment  the Python viewer as a kinematic world, no physics, with a random
             joystick generator.
Policy       MMM. Observation is the joystick, its own previous pose, the hidden
             state and the KNN candidates. Action is the next pose, or the
             residual correction.
KNN in loop  at every step the KNN controller also produces its reference pose,
             used as a regulariser in the reward through P_teach.

To implement
------------
- PPO: the clipped objective maximising expected discounted reward, the general
  option that makes no assumption about differentiability.
- Direct gradients: maximise the sum of rewards over a horizon H, for example 32
  frames, by backpropagation through time. Available because FK and every reward
  term are differentiable in a kinematic world; this is the Dreamer style route
  and is usually more sample efficient. The two are a section 9 comparison, so
  both share the environment, reward and rollout code and differ only in the
  update.
- Reward hacking watch: the known failure modes are tiny steps and frozen arms.
  Keep the teacher distance term, log every reward term separately, and render
  rollouts for visual inspection every few iterations rather than trusting the
  curves.
- Start from the Phase 2 checkpoint, and with the residual head start from a
  zero correction, which is exactly classic motion matching, so the run begins
  from a stable policy.
"""
