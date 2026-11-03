"""
Prepare LaFAN1 resolved: the main locomotion data.

60 fps, Geno skeleton, with toes and fingers. Noncommercial research licence.

To implement
------------
- Load the BVH files, validate against the Geno skeleton, and fail loudly on a
  mismatch.
- Detect foot contacts, compute Matching Database features and the joystick
  signal, and write processed clips to data/processed/.
- Report per clip statistics and flag clips with implausible root motion or
  excessive sliding, so bad data is caught before it reaches training.
"""
