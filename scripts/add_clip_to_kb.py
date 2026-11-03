"""
Append a clip to the knowledge base. The editability demo (plan section 4.6).

Compute the clip's features, append its rows to the Matching Database and the
Animation Database. No encoder, no retraining. This script is the main claim in
executable form, and it is also how a held out style is introduced for the
section 9 experiment and how a self recorded IMU suit capture is tried.

To implement
------------
- Append in place, update the index and the manifest, and verify that the
  existing rows are untouched.
- Report the time taken and the memory added, since "immediately usable" is a
  claim with numbers behind it.
- Then run the model over joystick input that calls for the new clip's motion,
  without touching the weights, and report the metrics.
"""
