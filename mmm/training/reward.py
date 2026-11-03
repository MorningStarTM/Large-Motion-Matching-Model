"""
Phase 3 reward, per frame (plan section 6.4).

To implement
------------
- P_slide = sum over feet of contact * | horizontal foot velocity |
            foot sliding, the most visible animation bug.
- P_jerk  = | p_t - 2 p_{t-1} + p_{t-2} | over joint positions
            the second difference, that is acceleration.
- P_path  = | root position - desired path point | + facing angle error
- P_teach = | FK(yhat_t) - FK(y_t from KNN) |
            the teacher distance, which acts like the KL penalty in RLHF and is
            the main defence against reward hacking.
- reward_t = -(alpha P_slide + beta P_jerk + gamma P_path + lambda P_teach)
- Every term differentiable, since the direct gradient variant backpropagates
  through them.
- Per term logging, because a single scalar reward hides which term the policy
  is actually optimising, and the known failure modes, tiny steps and frozen
  arms, show up as one term improving while another collapses.
"""
