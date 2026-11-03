"""
The three training phases.

  Phase 1  Pretrain            how bodies move
           Supervised next pose prediction on raw mocap.
  Phase 2  Distill from KNN    game style joystick response
           SFT plus DAgger, with the KNN teacher as the target.
  Phase 3  RL polish           better than the teacher
           PPO or direct gradients, with KNN in the loop.

Phase 0, the classic KNN controller, learns nothing and lives in mmm/teacher/.
It is the teacher these phases distil from and the reference the Phase 3 reward
regularises towards.

Shared pieces live in losses.py, reward.py and rollout.py so the three phase
scripts stay thin and comparable.
"""
