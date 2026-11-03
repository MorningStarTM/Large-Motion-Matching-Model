"""
Phase 2: distil from the KNN teacher. Learns game style joystick response.

Plan section 5.3, in three steps:

1. Drive the Phase 0 controller with random synthetic joystick input, turns,
   stops, sprints and reversals, for hours, recording joystick, retrieved
   candidates and output poses.
2. SFT the model to reproduce the teacher, with the section 6.2 losses and the
   teacher's output as the target.
3. DAgger rounds: run MMM, query the KNN teacher at the states MMM reaches, add
   those pairs to the dataset, fine tune again, and repeat a few rounds.

To implement
------------
- The SFT loop over logged teacher rollouts, streamed from teacher/logger.py.
- A DAgger round as one callable step: roll MMM out in the kinematic
  environment, label every visited state with teacher.reference_pose, append to
  the aggregated dataset, fine tune.
- Dataset bookkeeping across rounds: keep the original teacher data and all
  previous rounds, record which round each sample came from, and report the mix,
  since DAgger's whole point is correcting the distribution the model actually
  visits.
- Per round evaluation, so a round that makes things worse is visible
  immediately rather than after the last round.
"""
