"""
Build the Matching and Animation Databases from processed clips.

To implement
------------
- Take a dataset config, assemble the clip list, compute features, fit the
  normalization statistics on the training set only, and write both databases
  with their manifest.
- Assert through data/splits.py that no held out style clip entered the build.
- Report the memory footprint of each database, which is a number the thesis
  quotes directly.
"""
