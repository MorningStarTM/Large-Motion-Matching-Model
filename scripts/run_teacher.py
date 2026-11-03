"""
Run the Phase 0 KNN motion matching controller interactively.

The first milestone of the plan's getting started roadmap, and the thing to get
visibly right before any training starts.

To implement
------------
- Load a database, start the viewer, drive the teacher with the gamepad, and
  draw the overlays: desired path, query points, retrieved candidates and the
  chosen frame.
- Log every query, the top K candidates and the chosen frame from day one, since
  Phase 2 is built out of these logs.
- A headless mode driven by the synthetic joystick generator, which is how the
  hours of Phase 2 data are collected.
"""
