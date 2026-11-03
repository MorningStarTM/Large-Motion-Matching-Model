"""
Resample to 60 fps and reorient z up to y up.

To implement
------------
- resample(clip, target_fps=60): resample root translation linearly and
  rotations by spherical interpolation, never by interpolating euler angles.
- zup_to_yup(clip): rotate the root transform and the global orientation, and
  leave the local joint rotations untouched.
- Report the original frame rate per subset in the pipeline log, since a wrong
  assumed rate produces motion at the wrong speed, which is easy to miss.
"""
