"""
Phase 0: classic KNN motion matching. The teacher, which learns nothing.

This package is deliberately a first class part of the library rather than a
demo script, because the teacher is used in four places:

1. Phase 0 itself, as the baseline controller, driven by a gamepad.
2. Phase 2, as the source of SFT and DAgger targets.
3. Phase 3, as the in the loop reference pose behind the P_teach reward term.
4. Section 9, as the quality baseline MMM is compared against.

The loop: every 10 frames build a query from the character's current feet and
hips plus the desired path from the joystick, find the nearest frame in the
Matching Database (K equal to 1 for playback, top K for the model's context),
jump there and blend with inertialization; every frame play the next row of the
Animation Database.

Port of Holden's Motion Matching demo, in Python on GenoViewPython.
"""
