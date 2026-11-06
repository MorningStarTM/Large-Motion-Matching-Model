# Third party notices

## GenoViewPython / GenoView — Daniel Holden

MIT License, Copyright (c) 2024 Daniel Holden.
https://github.com/orangeduck/GenoViewPython

The full licence text is reproduced in [`licenses/GenoViewPython-LICENSE`](licenses/GenoViewPython-LICENSE).

What this project takes from it, stated precisely, because the distinction
between *adapted* and *independently written* matters for a thesis:

| File here | Relationship |
| --- | --- |
| [`mmm/viewer/camera.py`](mmm/viewer/camera.py) | **Adapted** from `genoview.py`'s `Camera` class. Same spherical formulation — azimuth, altitude, orbit radius, look-at target. Changed: centimetre units for LaFAN1, rates instead of frame-scaled deltas so sensitivity is frame-rate independent, and a `forward_xz` accessor for camera-relative input. |
| [`tests/reference/geno_quat.py`](tests/reference/geno_quat.py) | **Vendored verbatim** (one header added) from `quat.py`, used only as a reference implementation in the test suite. Not imported by `mmm/`. |
| [`mmm/data/rotations.py`](mmm/data/rotations.py) | **Independently written**, not adapted. It works in rotation matrices with a NumPy/PyTorch dual backend, where `quat.py` works in quaternions. It is *cross-validated* against the vendored `quat.py` in [`tests/test_rotations_vs_reference.py`](tests/test_rotations_vs_reference.py) — 11 tests, agreeing to 1e-9 or better on quaternion/matrix conversion, scaled-angle-axis, composition order, vector rotation, inversion and the ZYX Euler convention. |
| [`mmm/data/bvh.py`](mmm/data/bvh.py) | **Independently written**, not adapted. `quat.py`'s companion `bvh.py` parses line-by-line with regular expressions; this one parses the hierarchy block by block and reads the motion section with a single `np.array(block.split())`, which is what makes a 15,679-frame clip load in about 2 seconds. It also handles three LaFAN1-resolved quirks the reference does not need to, documented in its module docstring. |
| [`mmm/viewer/geno/resources/`](mmm/viewer/geno/resources/) | **Redistributed verbatim.** `Geno.bin` (the skinned character), `Geno_bind.bvh` (its bind pose) and the eleven GLSL shaders are Daniel Holden's, copied unmodified under the MIT licence above. |
| [`mmm/viewer/geno/renderer.py`](mmm/viewer/geno/renderer.py) | **Adapted** from `genoview.py`'s render loop. Same deferred pipeline, same passes in the same order — shadow map, GBuffer, SSAO with a separable depth-aware blur, lighting, FXAA — and the same lighting constants, which are jointly tuned. Changed: the eleven hand-sequenced `Begin*`/`End*` pairs are restructured into context managers on a `DeferredRenderer` that owns its framebuffers, so a missed `EndTextureMode` is impossible rather than merely unlikely. |
| [`mmm/viewer/geno/model.py`](mmm/viewer/geno/model.py) | **Independently written** against his file format. `genoview.py` loads `Geno.bin` straight into raylib structures through cffi; this parses it with NumPy so the skinning maths can be unit tested without a window, validates the file as it goes, and reconciles Geno's bind pose with the LaFAN1 BVH rest pose — which `genoview.py` never has to do, because it plays BVH files that are already in the bind convention. |
| [`mmm/viewer/geno/app.py`](mmm/viewer/geno/app.py) | **Our own.** Drives our motion matching controller and draws our obstacle field; the only thing it borrows is the order of the render passes. |

The project's phase plan describes `bvh.py` and `rotations.py` as adapted from
this project. They were in fact written from scratch, because the repository
was not available at the time. Rather than restate the plan's claim, the two
modules are cross-validated against it where that is possible — which is the
stronger position, since the agreement of two independent implementations is
evidence that neither self-consistency check could provide.

## Methods

Algorithms used under their published descriptions, with no code taken:

- **Spring damper, half-life parameterisation** — Daniel Holden, *Spring-It-On:
  The Game Developer's Spring-Roll-Call* (2021). Used in
  [`mmm/teacher/trajectory.py`](mmm/teacher/trajectory.py) and
  [`mmm/teacher/inertialize.py`](mmm/teacher/inertialize.py).
- **Inertialization** — David Bollo, *Inertialization: High-Performance
  Animation Transitions in Gears of War* (GDC 2018).
- **Motion matching, and the simulation bone that owns the character's
  position and facing** — Holden et al., and `orangeduck/Motion-Matching`.
  The reason the teacher's root is driven by the spring damper rather than by
  the retrieved clip is recorded, with the measurement behind it, in
  `TrajectoryPredictor.step_root`.
- **Learned Motion Matching** — Holden, Kanoun, Perepichka, Popa,
  *Learned Motion Matching*, SIGGRAPH 2020. Background and baseline for this
  project, not its architecture.
- **6D rotation representation** — Zhou et al., *On the Continuity of Rotation
  Representations in Neural Networks*, CVPR 2019.

## Data

- **LaFAN1 resolved** — https://github.com/orangeduck/lafan1-resolved,
  a re-solve of Ubisoft's LaFAN1 dataset onto the Geno skeleton. Not
  redistributed here; `data/` is git-ignored.
