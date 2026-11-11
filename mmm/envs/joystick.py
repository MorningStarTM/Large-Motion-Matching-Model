"""
Synthetic joystick generator: turns, stops, sprints and reversals.

Used to drive the teacher headless for Phase 2 data collection and to drive
episodes in Phase 3. The distribution of inputs here decides what the model is
good at, so it is a deliberate design object rather than uniform noise: uniform
random stick directions produce a character that twitches and never holds a
heading, which is nothing like how a game is played.

Deterministic under a seed, and the seed is recorded in the rollout manifest,
so a Phase 2 dataset or a Phase 3 episode can be reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["JoystickConfig", "SyntheticJoystick", "HARD_PRIMITIVES"]

# The primitives that make transitions look worst, kept aside as an evaluation
# subset: sharp reversals and stop-starts are where motion matching has to find
# a frame it has no good match for.
HARD_PRIMITIVES = ("reverse", "stop_start")


@dataclass
class JoystickConfig:
    """Primitive mix and duration ranges.

    Attributes:
        weights: relative probability of each primitive.
        duration_range: frames each primitive lasts, as (low, high).
        idle_duration_range: frames an `idle` span lasts. Separate from
            `duration_range` because idle needs to mean something different
            from `stop`. `stop` is a transitional blip inside a hold/turn/
            sprint mix -- the character slows, pauses briefly, moves again --
            and at `duration_range`'s 30 to 150 frames (0.5 to 2.5s) that is
            right for a transition. A player who is not touching the stick at
            all holds that for seconds at a time, often much longer, so idle
            gets its own, longer range. Without this distinction the dataset
            has no real "character just standing there" state, only brief
            pauses between other things, and the model never has to learn
            that a joystick can mean nothing for a while.
        walk_magnitude: stick magnitude treated as walking.
        run_magnitude: stick magnitude treated as running.
        turn_rate_range: radians per second while turning.
        strafe_probability: chance a primitive faces somewhere other than the
            direction of travel, which is what exercises the separate facing
            input.
    """

    weights: dict = field(
        default_factory=lambda: {
            "hold": 3.0,
            "turn": 3.0,
            "stop": 1.5,
            "stop_start": 1.5,
            "sprint": 1.5,
            "reverse": 1.0,
            "circle": 1.0,
            "idle": 1.5,
        }
    )
    duration_range: tuple = (30, 150)
    idle_duration_range: tuple = (120, 400)
    walk_magnitude: float = 0.45
    run_magnitude: float = 1.0
    turn_rate_range: tuple = (0.5, 2.5)
    strafe_probability: float = 0.15


class SyntheticJoystick:
    """Generates stick input one frame at a time.

    Usage:

        joystick = SyntheticJoystick(seed=0)
        for _ in range(n):
            stick, facing = joystick.step()
            controller.step(stick, facing)

    Args:
        frame_time: seconds per frame.
        config: the primitive mix.
        seed: for reproducibility.
    """

    def __init__(self, frame_time: float = 1.0 / 60.0, config: JoystickConfig | None = None, seed: int = 0):
        self.frame_time = frame_time
        self.config = config or JoystickConfig()
        self.rng = np.random.RandomState(seed)
        self.seed = seed

        self._heading = self.rng.uniform(0.0, 2.0 * np.pi)
        self._magnitude = 0.0
        self._turn_rate = 0.0
        self._facing_offset = 0.0
        self._primitive = "hold"
        self._remaining = 0
        self._counts: dict[str, int] = {}
        self._next_primitive()

    # -- primitives -------------------------------------------------------

    def _next_primitive(self) -> None:
        names = list(self.config.weights)
        probabilities = np.array([self.config.weights[n] for n in names], dtype=np.float64)
        probabilities /= probabilities.sum()
        self._primitive = str(self.rng.choice(names, p=probabilities))
        # Idle draws its duration from its own, longer range; see
        # `JoystickConfig.idle_duration_range` for why it needs to differ from
        # a transitional `stop`.
        duration_range = (
            self.config.idle_duration_range
            if self._primitive == "idle"
            else self.config.duration_range
        )
        self._remaining = int(self.rng.randint(*duration_range))
        self._counts[self._primitive] = self._counts.get(self._primitive, 0) + self._remaining

        config = self.config
        self._turn_rate = 0.0
        self._facing_offset = 0.0

        if self._primitive == "hold":
            self._magnitude = float(self.rng.uniform(config.walk_magnitude, config.run_magnitude))
        elif self._primitive == "turn":
            self._magnitude = float(self.rng.uniform(config.walk_magnitude, config.run_magnitude))
            self._turn_rate = float(self.rng.uniform(*config.turn_rate_range)) * self.rng.choice([-1, 1])
        elif self._primitive in ("stop", "idle"):
            # Mechanically identical to a zero stick -- the distinction is
            # purely how long it is held, handled above.
            self._magnitude = 0.0
        elif self._primitive == "stop_start":
            # Half the span stopped, then moving again: the stop-start that
            # makes transitions look worst.
            self._magnitude = 0.0
            self._remaining = max(8, self._remaining // 2)
        elif self._primitive == "sprint":
            self._magnitude = 1.0
        elif self._primitive == "reverse":
            self._heading += np.pi
            self._magnitude = float(self.rng.uniform(config.walk_magnitude, config.run_magnitude))
        elif self._primitive == "circle":
            self._magnitude = float(self.rng.uniform(config.walk_magnitude, config.run_magnitude))
            self._turn_rate = float(self.rng.uniform(0.8, 1.6)) * self.rng.choice([-1, 1])

        if self.rng.rand() < config.strafe_probability:
            self._facing_offset = float(self.rng.uniform(-np.pi / 2, np.pi / 2))

    # -- the protocol -----------------------------------------------------

    def step(self) -> tuple[np.ndarray, np.ndarray | None]:
        """One frame of input.

        Returns:
            `(stick, facing_stick)` both `(2,)` in world XZ, or
            `(stick, None)` when facing should follow the direction of travel.
        """
        if self._remaining <= 0:
            previous = self._primitive
            self._next_primitive()
            # A stop_start resumes rather than picking a fresh primitive, so
            # the restart really is a restart in the same direction.
            if previous == "stop_start":
                self._primitive = "hold"
                self._magnitude = float(
                    self.rng.uniform(self.config.walk_magnitude, self.config.run_magnitude)
                )

        self._heading += self._turn_rate * self.frame_time
        self._remaining -= 1

        stick = np.array(
            [np.sin(self._heading), np.cos(self._heading)], dtype=np.float32
        ) * self._magnitude

        facing = None
        if abs(self._facing_offset) > 1e-6:
            angle = self._heading + self._facing_offset
            facing = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

        return stick, facing

    def realised_mix(self) -> dict[str, float]:
        """The fraction of frames each primitive actually produced.

        Reported rather than assumed: the configured weights are per primitive
        while the durations vary, so the realised frame mix is not the weight
        vector and the difference matters for what the model learns.
        """
        total = sum(self._counts.values()) or 1
        return {name: count / total for name, count in sorted(self._counts.items())}
