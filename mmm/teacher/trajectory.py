"""
Spring damper: raw joystick input to a smooth desired path.

Parameterised by half life rather than raw stiffness, which is the form
Holden's Spring Roll Call uses and is far easier to tune: a half life of 0.2
means "the character covers half the remaining gap to the requested velocity
every 0.2 seconds", which can be reasoned about directly.

The critically damped spring below has no overshoot, which is what a character
controller wants: an overshooting velocity would make the character lurch past
the requested direction on every stick flick.

The same object serves the teacher's query and the model's joystick input, so
that at runtime both see the identical 12 numbers.

`predict` extrapolates towards the *requested* velocity and facing, not the
current smoothed ones. Using the current values as the spring goal makes the
predicted path converge back to what the character is already doing, which
zeroes the turn request in features 6:12 and leaves the search unable to see
that a turn was asked for at all. The predicted future has to describe where
the player asked to go, which is the whole point of giving the search a
trajectory rather than a velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mmm.data.features import TRAJECTORY_HORIZONS

__all__ = [
    "TrajectoryPredictor",
    "halflife_to_damping",
    "angle_to_direction",
    "direction_to_angle",
]


def halflife_to_damping(halflife: float, eps: float = 1e-5) -> float:
    """Convert a half life in seconds to the damping coefficient the spring
    integrator uses. From Holden's Spring Roll Call."""
    return 4.0 * 0.6931471805599453 / (halflife + eps)


def _wrap_angle(angle):
    """Wrap an angle to `[-pi, pi)`."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _critical_spring_angle(
    x: float, v: float, goal: float, halflife: float, dt: float
) -> tuple[float, float]:
    """One spring step on an *angle*, taking the shortest way round.

    Facing cannot be sprung as a 2D vector. Interpolating the vector towards a
    goal 180 degrees away passes through the origin, where the direction is
    undefined and normalisation either explodes or stalls; and a vector spring
    has no notion of "the short way round", so a request just past -pi can send
    the character spinning the long way. Wrapping the *offset* to `[-pi, pi)`
    fixes both: the spring always sees a difference of at most half a turn, and
    the zero crossing never arises.
    """
    damping = halflife_to_damping(halflife)
    c = damping * 0.5
    offset = _wrap_angle(x - goal)
    j = v + offset * c
    decay = float(np.exp(-c * dt))
    return _wrap_angle(decay * (offset + j * dt) + goal), decay * (v - j * c * dt)


def angle_to_direction(angle):
    """An angle measured from +Z toward +X as a unit XZ direction.

    Matches `spaces.RootFrame.yaw`, so a facing angle here and a forward
    vector there mean the same thing.
    """
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)


def direction_to_angle(direction) -> float:
    return float(np.arctan2(direction[0], direction[1]))


def _critical_spring_damper(
    x: np.ndarray,
    v: np.ndarray,
    goal: np.ndarray,
    halflife: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One step of a critically damped spring towards `goal`.

    Returns the new `(x, v)`. Uses the closed form solution rather than an
    Euler step, so the result is stable and frame rate independent even at a
    short half life, where an explicit integrator would oscillate or explode.
    """
    damping = halflife_to_damping(halflife)
    c = damping * 0.5
    offset = x - goal
    # Closed form of the critically damped case: both roots equal -c.
    j = v + offset * c
    decay = np.exp(-c * dt)
    new_x = decay * (offset + j * dt) + goal
    new_v = decay * (v - j * c * dt)
    return new_x, new_v


@dataclass
class TrajectoryPredictor:
    """Smooths joystick input and predicts the future root path.

    State is the current smoothed velocity and facing, which is why this is a
    stateful object rather than a function: the smoothing is what makes the
    desired path continuous across frames, and that continuity is the whole
    point.

    Args:
        frame_time: seconds per frame.
        velocity_halflife: how quickly the smoothed velocity chases the stick.
        facing_halflife: the same for facing. Sets how far the body lags a
            turn request, and the lag is linear in it: measured against a
            stick sweeping at 60 deg/s, a half life of 0.1 lags 8 degrees,
            0.2 lags 16 and 0.4 lags 33, while foot sliding is flat across the
            range. 0.2 is responsive without snapping; raise it for a heavier
            character.
        max_speed: units per second at full stick deflection. In LaFAN1's
            centimetres, sprinting is roughly 400 and walking roughly 120.
        horizons: lookahead in frames, matching the feature layout.
    """

    frame_time: float = 1.0 / 60.0
    velocity_halflife: float = 0.27
    facing_halflife: float = 0.2
    max_speed: float = 400.0
    horizons: tuple = TRAJECTORY_HORIZONS

    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    velocity_rate: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    facing: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0], np.float32))
    facing_angle: float = 0.0
    facing_rate: float = 0.0

    # The *requested* goals, held so `predict` can extrapolate towards where
    # the player asked to be rather than towards where the character already
    # is. Keeping them is what makes a turn request visible to the search.
    goal_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    goal_facing_angle: float = 0.0

    def reset(self, facing: np.ndarray | None = None) -> None:
        self.velocity = np.zeros(2, np.float32)
        self.velocity_rate = np.zeros(2, np.float32)
        self.goal_velocity = np.zeros(2, np.float32)
        self.facing_angle = (
            0.0 if facing is None else direction_to_angle(np.asarray(facing, np.float32))
        )
        self.goal_facing_angle = self.facing_angle
        self.facing = angle_to_direction(self.facing_angle)
        self.facing_rate = 0.0

    def update(self, stick: np.ndarray, facing_stick: np.ndarray | None = None) -> None:
        """Advance the smoothing by one frame.

        Args:
            stick: `(2,)` movement request in world XZ, magnitude in `[0, 1]`.
            facing_stick: `(2,)` facing request. Defaults to the movement
                direction, which is the usual third person behaviour; passing
                it separately is what allows strafing and turning on the spot.
        """
        stick = np.asarray(stick, dtype=np.float32).reshape(2)
        magnitude = float(np.linalg.norm(stick))
        if magnitude > 1.0:
            stick = stick / magnitude
            magnitude = 1.0

        self.goal_velocity = (stick * self.max_speed).astype(np.float32)
        self.velocity, self.velocity_rate = _critical_spring_damper(
            self.velocity, self.velocity_rate, self.goal_velocity,
            self.velocity_halflife, self.frame_time,
        )

        if facing_stick is None:
            # No separate facing request: face the way we are asked to travel.
            # A released stick leaves the goal where it was, so the character
            # holds its heading instead of snapping back to +Z.
            if magnitude > 1e-3:
                self.goal_facing_angle = direction_to_angle(stick / magnitude)
        else:
            facing_stick = np.asarray(facing_stick, dtype=np.float32).reshape(2)
            norm = float(np.linalg.norm(facing_stick))
            if norm > 1e-3:
                self.goal_facing_angle = direction_to_angle(facing_stick / norm)

        self.facing_angle, self.facing_rate = _critical_spring_angle(
            self.facing_angle, self.facing_rate, self.goal_facing_angle,
            self.facing_halflife, self.frame_time,
        )
        self.facing = angle_to_direction(self.facing_angle)

    def step_root(
        self, origin: np.ndarray, forward: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the simulated root one frame: the "simulation bone".

        This is the character's authoritative position and facing, and it is
        deliberately *not* integrated from the retrieved clip's root velocity.

        Integrating the clip instead looks right and is not. The search is only
        asked for a net heading change at 20, 40 and 60 frames, but playback
        jumps again after 10, so the character commits to the first sixth of a
        second of whatever the matched clip was doing and then re-rolls. Real
        locomotion in LaFAN1 carries a mean yaw rate near 45 deg/s, so each
        jump contributes several degrees in a direction nothing constrained:
        measured, the heading random walked to a mean error of 41.6 degrees
        against a held stick, which is why turning felt unsteerable.

        Driving the root from the spring damper instead makes facing exactly as
        controllable as the stick, and leaves the search doing what it is good
        at -- choosing a pose whose feet and trajectory suit the requested
        motion. This is how Holden's motion matching drives its simulation
        bone, with the animation matched to the simulation rather than the
        other way round.

        Args:
            origin: `(3,)` current floor origin.
            forward: `(2,)` current unit forward in XZ, unused but accepted so
                this can stand in for `spaces.integrate_root`.

        Returns:
            The new `(origin, forward)`.
        """
        new_origin = np.asarray(origin, np.float32).copy()
        new_origin[0] += float(self.velocity[0]) * self.frame_time
        new_origin[2] += float(self.velocity[1]) * self.frame_time
        new_origin[1] = 0.0
        return new_origin, self.facing.copy()

    def realised_root_velocity(self, forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The simulated root's velocity, in the character space of `forward`.

        The teacher writes this into the pose it emits so that the logged
        demonstration is self consistent: integrating the root velocity the
        teacher reports reproduces the path the teacher actually walked. Phase
        2 distils from these rollouts and the model predicts exactly this
        quantity, so a mismatch here would train the model towards a path its
        own teacher never took.
        """
        from mmm.data.spaces import _rotate_into_frame

        forward = np.asarray(forward, np.float32).reshape(2)
        local = _rotate_into_frame(self.velocity[None], forward[None])[0]
        linear = np.array([local[0], 0.0, local[1]], np.float32)
        angular = np.array([0.0, self.facing_rate, 0.0], np.float32)
        return linear, angular

    def predict(self, origin_forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The future root path, in the character's current character space.

        Integrates the smoothed velocity forward, continuing to decay it
        towards the same goal, so the prediction curves the way the character
        actually will rather than running off in a straight line.

        Args:
            origin_forward: `(2,)` the character's current forward in world XZ,
                used to express the result in character space.

        Returns:
            `positions` `(n_horizons, 2)` and `directions` `(n_horizons, 2)`,
            in character space, laid out exactly as the feature blocks
            `future_positions` and `future_directions`.
        """
        from mmm.data.spaces import _rotate_into_frame

        position = np.zeros(2, np.float32)
        velocity = self.velocity.copy()
        velocity_rate = self.velocity_rate.copy()
        facing_angle = self.facing_angle
        facing_rate = self.facing_rate

        positions = np.zeros((len(self.horizons), 2), np.float32)
        directions = np.zeros((len(self.horizons), 2), np.float32)

        longest = max(self.horizons)
        horizon_index = {h: i for i, h in enumerate(self.horizons)}

        for step in range(1, longest + 1):
            position = position + velocity * self.frame_time
            velocity, velocity_rate = _critical_spring_damper(
                velocity, velocity_rate, self.goal_velocity,
                self.velocity_halflife, self.frame_time,
            )
            facing_angle, facing_rate = _critical_spring_angle(
                facing_angle, facing_rate, self.goal_facing_angle,
                self.facing_halflife, self.frame_time,
            )
            if step in horizon_index:
                i = horizon_index[step]
                positions[i] = position
                directions[i] = angle_to_direction(facing_angle)

        forward = np.asarray(origin_forward, np.float32).reshape(2)
        return (
            _rotate_into_frame(positions, forward[None]).astype(np.float32),
            _rotate_into_frame(directions, forward[None]).astype(np.float32),
        )
