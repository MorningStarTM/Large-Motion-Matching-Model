"""
Pose losses for Phases 1 and 2 (plan section 6.2), plus the residual term.

Notation: y_t true pose, yhat_t predicted pose, R(.) joint rotations as
matrices, FK(.) joint positions in character space, r_t root velocity, c_t foot
contacts, dt the frame time of 1/60 s.

To implement
------------
- L_rot     = mean | R(yhat_t) - R(y_t) |                               (L1)
- L_pos     = mean | FK(yhat_t) - FK(y_t) |
              measured after forward kinematics, where the eye sees the error.
- L_vel     = mean | (FK(yhat_t) - FK(yhat_{t-1})) - (FK(y_t) - FK(y_{t-1})) |
              / dt, and the same term on rotations. This removes jitter; plain
              MSE gave jittery motion in LMM, so this term is not optional.
- L_root    = mean | rhat_t - r_t |
- L_contact = BCE(chat_t, c_t), optional.
- Total     = w_rot L_rot + w_pos L_pos + w_vel L_vel + w_root L_root
              + w_c L_contact, with weights set so each term has roughly equal
              magnitude. Provide a calibration helper that measures each term's
              scale on a batch and reports suggested weights, rather than
              leaving the weights to guesswork.
- L_res     = w_res * | correction | squared, for the residual variant (plan
              section 6.3), keeping the correction small so the output stays
              close to real mocap.

The target y_t is mocap in Phase 1, and the KNN teacher's output in Phase 2, on
teacher rollouts and on DAgger states. The loss code does not care which, which
is why it is shared.
"""
