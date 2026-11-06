"""
Axis aligned boxes, and the queries the matching features ask of them.

An obstacle here is a box resting on the ground: a footprint in x and z, and a
top surface at some height. That is deliberately the simplest shape that covers
what LaFAN1 actually contains -- steps, platforms and vault boxes -- and it is
also the shape a user can place in the viewer by dragging a cube, so the
recovered geometry and the runtime geometry are the same type.

Two queries drive everything downstream:

- `support_height(x, z)`: how high the ground is at a point, which is what a
  foot would stand on. Zero off every box.
- `edge_ahead(origin, forward)`: the next place along a ray where the support
  height changes, and what it changes to. A step is only interesting *before*
  it is reached, so the feature that matters is the distance to its leading
  edge rather than the height underfoot.

The second one is a sampled walk along the ray, not a box intersection, and the
difference is the whole point. An intersection test answers "where is the next
box", which is the wrong question in three situations that all occur in the
data: standing on a platform, where the test fires at distance zero on the box
already underfoot; stepping *down* off a platform, where the support changes
with no box being entered at all; and crossing a gap between two boxes of the
same height, which is not a change the character needs to react to. Asking
where the support height changes answers all three with one mechanism.

Both are pure functions of geometry with no motion involved, which is why they
live apart from `recover.py` and are tested on hand built boxes.

Units follow the data: centimetres, with the ground at y = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Box", "BoxField"]


@dataclass(frozen=True)
class Box:
    """One axis aligned obstacle, resting on the ground.

    Attributes:
        x0, x1: footprint extent in x, with `x0 < x1`.
        z0, z1: footprint extent in z, with `z0 < z1`.
        top: height of the upper surface above the ground.
        base: height of the lower surface. Non zero only for a shape that
            floats, such as a bar to duck under; the recovery never produces
            one, but the geometry supports it so that a hand placed obstacle
            can.
        label: free text, carried through for diagnostics.
    """

    x0: float
    x1: float
    z0: float
    z1: float
    top: float
    base: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.z1 <= self.z0:
            raise ValueError(
                f"box footprint must be non-empty with x0<x1 and z0<z1; got "
                f"x[{self.x0}, {self.x1}] z[{self.z0}, {self.z1}]"
            )
        if self.top <= self.base:
            raise ValueError(f"box top ({self.top}) must exceed base ({self.base})")

    # -- derived ----------------------------------------------------------

    @property
    def centre(self) -> np.ndarray:
        """Centre of the footprint, `(x, z)`."""
        return np.array([(self.x0 + self.x1) / 2, (self.z0 + self.z1) / 2], np.float32)

    @property
    def size(self) -> np.ndarray:
        """Footprint extent and height, `(dx, dy, dz)`."""
        return np.array(
            [self.x1 - self.x0, self.top - self.base, self.z1 - self.z0], np.float32
        )

    @property
    def height(self) -> float:
        return float(self.top - self.base)

    def corners(self) -> np.ndarray:
        """The eight corners, `(8, 3)`. Used by the viewer to draw the cube."""
        xs, ys, zs = (self.x0, self.x1), (self.base, self.top), (self.z0, self.z1)
        return np.array([[x, y, z] for x in xs for y in ys for z in zs], np.float32)

    # -- queries ----------------------------------------------------------

    def contains_xz(self, x, z) -> np.ndarray:
        """Whether each point is inside the footprint, elementwise."""
        x, z = np.asarray(x, np.float32), np.asarray(z, np.float32)
        return (x >= self.x0) & (x <= self.x1) & (z >= self.z0) & (z <= self.z1)

    def distance_xz(self, x, z) -> np.ndarray:
        """Distance from each point to the footprint, zero inside.

        The usual axis aligned trick: clamp the per axis overshoot at zero and
        take the length. Inside the box every overshoot is negative, so the
        result is zero -- this is a distance *to* the box, not a signed
        distance field, because the features only ever ask how far until the
        obstacle is reached.
        """
        x, z = np.asarray(x, np.float32), np.asarray(z, np.float32)
        dx = np.maximum(np.maximum(self.x0 - x, x - self.x1), 0.0)
        dz = np.maximum(np.maximum(self.z0 - z, z - self.z1), 0.0)
        return np.sqrt(dx * dx + dz * dz).astype(np.float32)


class BoxField:
    """A set of boxes, queried as one surface.

    The field is what both halves of the system talk to: `recover.py` produces
    one from a clip, the viewer produces one from whatever cubes the user has
    placed, and `features.py` cannot tell the difference. That is the whole
    point of going through a field rather than passing boxes around.

    Overlapping boxes are resolved by taking the highest top, so a short box
    standing on a platform reads as one staircase rather than two conflicting
    heights.
    """

    def __init__(self, boxes=()):
        self.boxes: list[Box] = list(boxes)

    def __len__(self) -> int:
        return len(self.boxes)

    def __iter__(self):
        return iter(self.boxes)

    def __repr__(self) -> str:
        if not self.boxes:
            return "BoxField(empty)"
        tops = sorted(b.top for b in self.boxes)
        return f"BoxField({len(self.boxes)} boxes, tops {tops[0]:.0f}..{tops[-1]:.0f})"

    def add(self, box: Box) -> None:
        self.boxes.append(box)

    # -- queries ----------------------------------------------------------

    def support_height(self, x, z) -> np.ndarray:
        """Height of the walkable surface at each point.

        Vectorised over the points and looped over the boxes, because there are
        a handful of boxes and potentially tens of thousands of points: a clip
        of 15,000 frames times three trajectory horizons is 45,000 queries, and
        looping the other way round would be 45,000 Python iterations rather
        than ten.

        Only boxes whose base is on the ground can support anything, so a
        floating box is skipped here and shows up only through `edge_ahead`.
        """
        x = np.asarray(x, np.float32)
        z = np.asarray(z, np.float32)
        height = np.zeros(np.broadcast(x, z).shape, np.float32)
        for box in self.boxes:
            if box.base > 1e-6:
                continue
            inside = box.contains_xz(x, z)
            height = np.where(inside & (box.top > height), box.top, height)
        return height

    # How finely the ray is walked, in centimetres, and how big a change in
    # support height counts. 5 cm quantises the reported distance, which is far
    # below the precision the motion itself has -- a stride is 60 to 150 cm --
    # and 4 cm is below the smallest real step measured in LaFAN1 (17 cm) while
    # above the couple of centimetres that floor estimation and the recovery
    # margin can be out by.
    SAMPLE_STEP = 5.0
    HEIGHT_TOLERANCE = 4.0

    def support_profile(self, origin, forward, max_distance: float = 400.0,
                        step: float | None = None) -> tuple:
        """Support height sampled along a horizontal ray.

        Args:
            origin: `(x, z)`, or `(n, 2)` for one ray per frame sharing the
                same sample distances.
            forward: `(x, z)` or `(n, 2)`, normalised internally.
            max_distance: how far along to sample.
            step: sample spacing; `SAMPLE_STEP` by default.

        Returns:
            `(distances, heights)`. `distances` is `(n_samples,)` and `heights`
            is `(n_samples,)` for a single ray or `(n, n_samples)` for a batch.

        Batched because the per frame form is the hot path: a clip of 15,000
        frames at 80 samples each is 1.2 million support queries, which is
        fifteen vectorised calls one way round and 15,000 Python iterations the
        other.
        """
        step = self.SAMPLE_STEP if step is None else step
        origin = np.atleast_2d(np.asarray(origin, np.float32))
        forward = np.atleast_2d(np.asarray(forward, np.float32))
        single = origin.shape[0] == 1 and np.ndim(np.asarray(origin)) <= 2

        norm = np.linalg.norm(forward, axis=-1, keepdims=True)
        # A character standing perfectly still can have a degenerate facing for
        # a frame. Leaving it as a zero vector would sample the same point
        # repeatedly, which is harmless and finite; dividing by zero would not
        # be.
        forward = forward / np.maximum(norm, 1e-9)

        distances = np.arange(step, max_distance + 0.5 * step, step, dtype=np.float32)
        points = origin[:, None, :] + forward[:, None, :] * distances[None, :, None]
        heights = self.support_height(points[..., 0], points[..., 1])

        if single and heights.shape[0] == 1:
            return distances, heights[0]
        return distances, heights

    def edge_ahead(self, origin, forward, max_distance: float = 400.0,
                   step: float | None = None, tolerance: float | None = None):
        """The next change in support height along a horizontal ray.

        Args:
            origin: `(x, z)` start of the ray.
            forward: `(x, z)` direction; normalised internally, so the returned
                distance is in data units.
            max_distance: how far to look. Beyond this the answer is reported as
                `max_distance` with the height underfoot, rather than infinity:
                a feature vector has to hold a finite number, and "nothing
                within four metres" and "nothing within a kilometre" are the
                same thing to a character deciding how to step.
            step: sample spacing.
            tolerance: how big a height change counts.

        Returns:
            `(distance, top)`: how far until the support height changes, and
            what it changes to. A drop reports a `top` below the current
            support, which is correct and is why this is not clamped at zero --
            stepping down is as much a motion to retrieve as stepping up.

            With no change within range, the distance is `max_distance` and the
            top is the current support height, so the relative height the
            features store is zero. That is the same answer flat ground gives,
            which is what makes the feature continuous as an obstacle drifts out
            of range rather than jumping when it crosses the horizon.
        """
        tolerance = self.HEIGHT_TOLERANCE if tolerance is None else tolerance
        origin = np.asarray(origin, np.float32).reshape(2)
        support_now = float(self.support_height(origin[0], origin[1]))

        if not self.boxes:
            return float(max_distance), support_now

        distances, heights = self.support_profile(
            origin, forward, max_distance=max_distance, step=step
        )
        heights = np.asarray(heights).reshape(-1)
        changed = np.abs(heights - support_now) > tolerance
        if not changed.any():
            return float(max_distance), support_now

        first = int(np.argmax(changed))
        return float(distances[first]), float(heights[first])

    # -- serialisation ----------------------------------------------------

    def to_dicts(self) -> list[dict]:
        return [
            {
                "x0": b.x0, "x1": b.x1, "z0": b.z0, "z1": b.z1,
                "top": b.top, "base": b.base, "label": b.label,
            }
            for b in self.boxes
        ]

    @classmethod
    def from_dicts(cls, records) -> "BoxField":
        return cls([Box(**r) for r in records])
