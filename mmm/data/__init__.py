"""
Motion data layer: files on disk -> arrays the model and the teacher consume.

Responsibilities
----------------
- Parse and write BVH, MMM's working format (plan section 14).
- Hold the Geno skeleton definition and convert between world, character and
  local space.
- Convert rotation formats, with 6D as the network facing representation.
- Differentiable forward kinematics, so FK can sit inside a loss.
- Build the two databases of classic motion matching: the Matching Database
  (short feature vector per frame, used for search) and the Animation Database
  (full pose per frame, used for drawing).
- Label foot contacts and define the dataset splits, including the ten
  100STYLE styles held out for the editability experiment.

Everything here is NumPy first and framework agnostic, except fk.py which has a
torch path because it is used inside training losses.
"""
