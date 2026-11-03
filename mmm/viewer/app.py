"""
The viewer application loop.

To implement
------------
- A raylib window drawing the Geno character from a stream of poses, at 60 Hz.
- A source abstraction with three implementations behind one interface: the KNN
  teacher, the MMM model, and a plain clip player for inspecting data. Switchable
  at runtime with a key, since side by side comparison is how most animation
  bugs are actually found.
- Drive the source with the gamepad, or with the synthetic joystick generator
  for unattended recording.
- Recording: dump the pose stream and a video or GIF, for the visual inspection
  that plan section 5.4 asks for every few RL iterations.
- Pose application through data/fk.py and data/spaces.py, never a second copy of
  that maths, so what the viewer shows is what the losses measured.
"""
