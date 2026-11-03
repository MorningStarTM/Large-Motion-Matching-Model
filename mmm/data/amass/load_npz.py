"""
Read one AMASS .npz clip.

To implement
------------
- load(path) -> AmassClip with the fields the pipeline needs:
    poses[:, :66]      axis angle rotations for the 22 body joints; the hand
                       joints of SMPL+H are skipped
    trans              root translation per frame
    betas[:10]         body shape coefficients, which set bone lengths
    gender
    mocap_framerate    differs between subsets, so it must be read, not assumed
- Validate shapes and report the subset name, so the licence of every clip used
  can be traced back.
- Note the coordinate convention: AMASS is z up while Geno and raylib are y up.
  The conversion itself lives in resample.py.
"""
