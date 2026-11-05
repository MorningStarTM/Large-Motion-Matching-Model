"""
Player input: gamepad, mouse and keyboard, read into one structure.

The left stick (or WASD) is the movement request and is interpreted in the
*camera's* frame, not the world's. The right stick, or the mouse, orbits the
camera. Turning therefore works two ways, which is what a third person game
does: swing the camera and the held stick walks the new way, or hold the aim
button and point the body independently of travel, which is what strafing and
turning on the spot need.

Raw device values go straight out of here. The smoothing lives in
`teacher/trajectory.py`'s spring damper, because the same smoothing has to
apply to the synthetic joystick in Phase 2, and duplicating it would let the
two drift apart.

`read_input` returns an `InputFrame` rather than a bare stick so that the
viewer, the synthetic joystick and any later model driven source all hand the
controller the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "InputFrame",
    "read_input",
    "read_gamepad",
    "DEAD_ZONE",
    "gamepad_available",
    "MOUSE_SENSITIVITY",
    "STICK_LOOK_RATE",
]

# Sticks rest a little off centre, and a character that creeps while the pad is
# untouched reads as a bug in the controller rather than in the hardware.
DEAD_ZONE = 0.18

# Radians per second per pixel of mouse movement. Tuned so a palm width sweep
# turns the camera through roughly a half circle at 60Hz.
MOUSE_SENSITIVITY = 0.22

# Radians per second at full right stick deflection.
STICK_LOOK_RATE = 2.8

# Pad 0 is the first connected controller.
_PAD = 0


@dataclass
class InputFrame:
    """One frame of player intent, before any smoothing.

    Attributes:
        move: `(2,)` movement request in *camera* space, magnitude `[0, 1]`.
            y is away from the viewer.
        look: `(2,)` camera turn rate in radians per second; x orbits,
            y pitches.
        zoom: scroll ticks this frame, positive to zoom in.
        aim: whether the facing request should follow the camera rather than
            the direction of travel.
    """

    move: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    look: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    zoom: float = 0.0
    aim: bool = False


def _apply_dead_zone(x: float, y: float, response: float = 1.6) -> np.ndarray:
    """Dead zone and response curve on one stick.

    The dead zone is applied radially, not per axis: per axis clamping makes
    the diagonal directions reachable at lower magnitudes than the cardinal
    ones, so the character accelerates differently depending on which way it
    faces. The remaining range is rescaled so that just past the dead zone
    means just above zero, rather than jumping to DEAD_ZONE.
    """
    vector = np.array([x, y], dtype=np.float32)
    magnitude = float(np.linalg.norm(vector))
    if magnitude < DEAD_ZONE:
        return np.zeros(2, np.float32)

    rescaled = min((magnitude - DEAD_ZONE) / (1.0 - DEAD_ZONE), 1.0)
    curved = rescaled**response
    return (vector / magnitude * curved).astype(np.float32)


def gamepad_available() -> bool:
    try:
        import pyray as rl
    except ImportError:
        return False
    return bool(rl.is_gamepad_available(_PAD))


def read_input(mouse_look: bool = True) -> InputFrame:
    """One frame of input from whatever the player is holding.

    Args:
        mouse_look: when True, mouse movement orbits the camera. Turned off so
            the cursor can be used normally.
    """
    import pyray as rl

    frame = InputFrame()

    if rl.is_gamepad_available(_PAD):
        # Y is inverted: pushing the stick away from you should walk into the
        # scene, which is +Z once the camera relative mapping is applied.
        frame.move = _apply_dead_zone(
            rl.get_gamepad_axis_movement(_PAD, rl.GamepadAxis.GAMEPAD_AXIS_LEFT_X),
            -rl.get_gamepad_axis_movement(_PAD, rl.GamepadAxis.GAMEPAD_AXIS_LEFT_Y),
        )
        look = _apply_dead_zone(
            rl.get_gamepad_axis_movement(_PAD, rl.GamepadAxis.GAMEPAD_AXIS_RIGHT_X),
            rl.get_gamepad_axis_movement(_PAD, rl.GamepadAxis.GAMEPAD_AXIS_RIGHT_Y),
        )
        frame.look = look * STICK_LOOK_RATE
        frame.aim = bool(
            rl.is_gamepad_button_down(
                _PAD, rl.GamepadButton.GAMEPAD_BUTTON_LEFT_TRIGGER_1
            )
        )
    else:
        frame.move = _keyboard_move(rl)
        frame.aim = bool(rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT))

    # The mouse is additive, so a pad and a mouse can be used together -- which
    # is also why the two have to pitch the same way. Neither axis is negated
    # here: `look[1]` lowers the camera, and lowering the camera is looking up,
    # so a negative value (mouse up, or stick up, both of which report
    # negative) means look up on either device. Negating only the mouse, as
    # this did at first, left the pad and the mouse pitching opposite ways.
    if mouse_look:
        delta = rl.get_mouse_delta()
        frame.look = frame.look + np.array(
            [delta.x * MOUSE_SENSITIVITY, delta.y * MOUSE_SENSITIVITY], np.float32
        )

    frame.zoom = float(rl.get_mouse_wheel_move())
    return frame


def _keyboard_move(rl) -> np.ndarray:
    """WASD fallback, so the viewer is usable without a pad.

    Keys are either fully pressed or not at all, so this cannot show whether
    the spring damper feels right -- which is part of what Phase 0 is judged
    on. Use a pad for that.
    """
    x = y = 0.0
    if rl.is_key_down(rl.KeyboardKey.KEY_A):
        x -= 1.0
    if rl.is_key_down(rl.KeyboardKey.KEY_D):
        x += 1.0
    if rl.is_key_down(rl.KeyboardKey.KEY_S):
        y -= 1.0
    if rl.is_key_down(rl.KeyboardKey.KEY_W):
        y += 1.0

    stick = np.array([x, y], np.float32)
    magnitude = float(np.linalg.norm(stick))
    if magnitude > 1e-6:
        stick /= magnitude
        # Shift walks instead of running, so both speeds are reachable from the
        # keyboard even though it has no analogue range.
        if rl.is_key_down(rl.KeyboardKey.KEY_LEFT_SHIFT):
            stick *= 0.45
    return stick


def read_gamepad() -> tuple[np.ndarray, np.ndarray | None]:
    """Deprecated world space reading, kept so older callers still work.

    Returns `(stick, facing)` with no camera involved, which is why the viewer
    no longer uses it: with a world space mapping, orbiting the camera does not
    change where the character walks.
    """
    frame = read_input(mouse_look=False)
    return frame.move, None
