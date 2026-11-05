"""
Third person orbit camera, and the camera relative input mapping.

Adapted from GenoViewPython's `Camera` class.
GenoViewPython is MIT licensed, Copyright (c) 2024 Daniel Holden:
https://github.com/orangeduck/GenoViewPython -- itself a Python port of
GenoView. See NOTICE.md at the repository root.

Changes from the original:

- Distances are in centimetres, because LaFAN1 is. The original's 4 unit orbit
  radius becomes 400.
- Azimuth and altitude are integrated from a mouse or stick *rate* rather than
  a per frame delta scaled by `dt`, so sensitivity does not change with frame
  rate.
- `forward_xz` is exposed, which the original has no need for. It is what makes
  the movement stick camera relative, and camera relative movement is the
  reason turning with the mouse works at all: pushing the stick away from the
  camera has to mean "walk away from the camera", whichever way the camera is
  pointing.

Why camera relative input matters here
--------------------------------------
With a world space stick mapping, moving the mouse changes only the view and
the character carries on walking north; the controller is steerable but the
viewer makes it look as though it is not. The stick is therefore interpreted in
the camera's frame, and the facing request follows either the direction of
travel or, while the aim button is held, the camera's own forward -- which is
what turning on the spot and strafing need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["OrbitCamera", "camera_relative"]


def _clamp(x: float, low: float, high: float) -> float:
    return low if x < low else (high if x > high else x)


@dataclass
class OrbitCamera:
    """Spherical camera about a moving target.

    Attributes:
        azimuth: horizontal angle in radians. Measured so that
            `forward_xz` at azimuth 0 is +Z, matching `spaces.FORWARD_AXIS`.
        altitude: vertical angle in radians, clamped away from straight down
            so the ground plane never flips.
        distance: orbit radius in centimetres.
        height: how far up the target the camera looks, in centimetres. Roughly
            chest height on the Geno skeleton.
    """

    azimuth: float = 0.0
    altitude: float = 0.35
    distance: float = 400.0
    height: float = 100.0

    min_distance: float = 120.0
    max_distance: float = 1200.0
    max_altitude: float = 0.45 * np.pi
    min_altitude: float = -0.05 * np.pi

    def update(self, look: np.ndarray, zoom: float, frame_time: float) -> None:
        """Advance the orbit.

        Args:
            look: `(2,)` turn rate, radians per second. x orbits, y pitches.
            zoom: scroll ticks this frame; positive zooms in.
            frame_time: seconds per frame.
        """
        self.azimuth = float(self.azimuth - look[0] * frame_time)
        self.altitude = _clamp(
            float(self.altitude + look[1] * frame_time),
            self.min_altitude,
            self.max_altitude,
        )
        # Zoom is proportional, so one tick feels the same when far out as when
        # close in. The original scaled it by dt, which made the sensitivity
        # depend on the frame rate.
        if zoom:
            self.distance = _clamp(
                self.distance * (0.9**zoom), self.min_distance, self.max_distance
            )

    @property
    def forward_xz(self) -> np.ndarray:
        """The camera's horizontal viewing direction, `(2,)` unit XZ.

        This is the direction the character walks when the stick is pushed
        away from the viewer.
        """
        return np.array([np.sin(self.azimuth), np.cos(self.azimuth)], np.float32)

    def eye(self, target: np.ndarray) -> np.ndarray:
        """Camera position for a target, `(3,)` in world space."""
        horizontal = self.distance * np.cos(self.altitude)
        forward = self.forward_xz
        return np.array(
            [
                target[0] - forward[0] * horizontal,
                target[1] + self.distance * np.sin(self.altitude),
                target[2] - forward[1] * horizontal,
            ],
            np.float32,
        )

    def target(self, origin: np.ndarray) -> np.ndarray:
        return np.array([origin[0], origin[1] + self.height, origin[2]], np.float32)


def camera_relative(stick: np.ndarray, azimuth: float) -> np.ndarray:
    """Rotate a camera space stick into world XZ.

    `stick` is `(2,)`: x is screen right, y is away from the viewer. At azimuth
    0 the camera looks along +Z, so pushing the stick away maps to +Z and
    agrees with `spaces.FORWARD_AXIS`.

    The lateral axis has to be the camera's *screen* right, which is
    `forward x up`, not simply the forward axis turned a quarter turn in the
    direction that happens to look natural when writing it down. The world here
    is right handed with Y up, and the camera sits behind the character looking
    along +Z, so world +X lies to the viewer's **left**:

        forward = ( sin a, cos a)          in XZ
        right   = (-cos a, sin a)          = forward x up, projected

    Getting this backwards is how the controls came out mirrored -- pulling the
    stick left sent the character right -- and it survived the first round of
    tests because those only checked that the lateral axis was perpendicular to
    forward, which a negated axis satisfies just as well. `tests/test_camera.py`
    now derives the basis from the rendered camera's own eye and target instead.
    """
    stick = np.asarray(stick, np.float32).reshape(2)
    cos_a, sin_a = float(np.cos(azimuth)), float(np.sin(azimuth))
    return np.array(
        [-stick[0] * cos_a + stick[1] * sin_a, stick[0] * sin_a + stick[1] * cos_a],
        np.float32,
    )
