"""
The viewer, with cubes you can place.

Subclasses `ViewerApp` and overrides `run`, reusing every `_draw_*` helper it
already has. `mmm/viewer/app.py` is untouched.

Overriding `run` rather than hooking into it is a deliberate trade. `ViewerApp.run`
has no seam for "draw extra world geometry" and no notion of placing anything,
and this needs both, so the loop is restated here. The cost is that a change to
the base loop has to be mirrored; the alternative was editing a file that works.

Controls, on top of the ones `ViewerApp` already has:

    1 - 5       place a cube ahead of the character: 20, 33, 47, 66 or 93 cm.
                Those are not round numbers -- they are five of the surface
                heights actually recovered from the mocap, so a cube placed
                with these keys is a height the database has frames for.
    6           place a tall 150 cm block, which the data has nothing for. Worth
                trying: it shows what the search does when asked for something
                the database cannot supply, which is the honest failure mode.
    backspace   remove the most recently placed cube.
    0           clear every cube.
    L           load the recovered studio layout, all 31 surfaces at once.

A cube is placed 250 cm ahead along the character's facing, far enough that the
trajectory predictor has time to see it before the character arrives.
"""

from __future__ import annotations

import numpy as np

from mmm.obstacles.geometry import Box, BoxField
from mmm.viewer.app import RAYLIB_AVAILABLE, ViewerApp

if RAYLIB_AVAILABLE:  # pragma: no cover - requires a display
    import pyray as rl

__all__ = ["ObstacleViewer", "PLACEMENT_HEIGHTS"]

# Heights the recovery actually found in LaFAN1, so a placed cube is something
# the database has motion for. See `scripts/recover_obstacles.py` for the table.
PLACEMENT_HEIGHTS = (20.9, 32.8, 47.3, 66.1, 93.5)

# And one the data has nothing for, to show the failure mode rather than hide it.
UNSUPPORTED_HEIGHT = 150.0

# How far ahead a cube lands, and how big it is. 120 cm square is close to the
# footprint of the real props (the recovered ones run 48 to 182 cm across).
PLACEMENT_DISTANCE = 250.0
PLACEMENT_SIZE = 120.0


class ObstacleViewer(ViewerApp):
    """`ViewerApp` that draws a `BoxField` and lets you add to it.

    Args:
        database: an `AugmentedDatabase`, for the skeleton and candidate poses.
        controller: an `ObstacleController`. Its `field` is what this mutates,
            so the controller sees a placed cube on its very next search
            without anything being rebuilt.
        **kwargs: passed to `ViewerApp`.
    """

    def __init__(self, database, controller, studio_field=None, **kwargs):
        super().__init__(database, controller, **kwargs)
        self.studio_field = studio_field
        self._placed: list[Box] = []

    # -- placement --------------------------------------------------------

    @property
    def field(self) -> BoxField:
        return self.controller.field

    def place_ahead(self, height: float) -> Box:
        """Drop a cube in front of the character.

        Positioned from the character's own frame rather than from the camera,
        so a cube lands where the character is heading however the camera
        happens to be pointing.
        """
        state = self.controller.state
        forward = np.asarray(state.forward, np.float64)
        forward = forward / max(float(np.linalg.norm(forward)), 1e-9)

        centre_x = float(state.origin[0]) + forward[0] * PLACEMENT_DISTANCE
        centre_z = float(state.origin[2]) + forward[1] * PLACEMENT_DISTANCE
        half = PLACEMENT_SIZE / 2.0

        box = Box(
            x0=centre_x - half,
            x1=centre_x + half,
            z0=centre_z - half,
            z1=centre_z + half,
            top=float(height),
            label=f"placed:{height:.0f}",
        )
        self.field.add(box)
        self._placed.append(box)
        return box

    def undo_placement(self) -> None:
        if not self._placed:
            return
        box = self._placed.pop()
        if box in self.field.boxes:
            self.field.boxes.remove(box)

    def clear_placements(self) -> None:
        self.controller.field = BoxField()
        self._placed = []

    def load_studio(self) -> None:
        """Swap in the recovered studio layout.

        The interesting comparison: a hand placed cube is a situation the
        database has *similar* frames for, while the studio layout is the exact
        geometry the actors negotiated. If the character handles the studio
        layout noticeably better, that gap is the generalisation cost.
        """
        if self.studio_field is None:
            return
        self.controller.field = BoxField(list(self.studio_field))
        self._placed = []

    # -- drawing ----------------------------------------------------------

    def _draw_boxes(self) -> None:  # pragma: no cover - requires a display
        """Draw each obstacle as a translucent solid with a bright wireframe.

        Translucent because a solid cube hides the character's feet at exactly
        the moment worth watching -- the step up -- and the wireframe is what
        makes the top surface readable against the grid.
        """
        for box in self.field:
            centre = rl.Vector3(
                (box.x0 + box.x1) / 2.0,
                (box.base + box.top) / 2.0,
                (box.z0 + box.z1) / 2.0,
            )
            width = box.x1 - box.x0
            height = box.top - box.base
            length = box.z1 - box.z0

            # Colour by height, so a staircase reads as a gradient rather than
            # as a wall of identical cubes.
            fraction = min(max(box.top / 120.0, 0.0), 1.0)
            fill = rl.Color(
                int(70 + 120 * fraction), int(150 - 60 * fraction), 200, 90
            )
            rl.draw_cube(centre, width, height, length, fill)
            rl.draw_cube_wires(
                centre, width, height, length, rl.Color(150, 220, 255, 220)
            )

    def _draw_support_marker(self) -> None:  # pragma: no cover - needs a display
        """Mark where the next change in support is, along the facing ray.

        This is the obstacle feature made visible. If the marker is not sitting
        on the near edge of the box the character is walking at, the query is
        wrong and no amount of watching the skeleton would reveal it.
        """
        state = self.controller.state
        distance, _ = self.controller.edge_ahead()
        forward = np.asarray(state.forward, np.float64)
        forward = forward / max(float(np.linalg.norm(forward)), 1e-9)

        point = rl.Vector3(
            float(state.origin[0]) + forward[0] * distance,
            1.0,
            float(state.origin[2]) + forward[1] * distance,
        )
        rl.draw_sphere(point, 7.0, rl.Color(255, 190, 60, 255))
        rl.draw_line_3d(
            rl.Vector3(float(state.origin[0]), 1.0, float(state.origin[2])),
            point,
            rl.Color(255, 190, 60, 160),
        )

    def _draw_obstacle_hud(self) -> None:  # pragma: no cover - needs a display
        support = self.controller.support_under_character()
        distance, top = self.controller.edge_ahead()
        lines = [
            f"boxes {len(self.field)}   placed {len(self._placed)}",
            f"support under character  {support:6.1f} cm",
            f"next change  {distance:6.1f} cm  ->  {top:6.1f} cm",
            "1-5 place recovered heights  6 place 150cm (unsupported)",
            "backspace undo   0 clear   L load studio layout",
        ]
        for i, line in enumerate(lines):
            rl.draw_text(
                line,
                12,
                self.config.height - 20 * len(lines) - 12 + i * 20,
                18,
                rl.Color(255, 210, 120, 255),
            )

    # -- the loop ---------------------------------------------------------

    def _handle_placement_keys(self) -> None:  # pragma: no cover - needs display
        keys = (
            rl.KeyboardKey.KEY_ONE,
            rl.KeyboardKey.KEY_TWO,
            rl.KeyboardKey.KEY_THREE,
            rl.KeyboardKey.KEY_FOUR,
            rl.KeyboardKey.KEY_FIVE,
        )
        for key, height in zip(keys, PLACEMENT_HEIGHTS):
            if rl.is_key_pressed(key):
                self.place_ahead(height)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_SIX):
            self.place_ahead(UNSUPPORTED_HEIGHT)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_BACKSPACE):
            self.undo_placement()
        if rl.is_key_pressed(rl.KeyboardKey.KEY_ZERO):
            self.clear_placements()
        if rl.is_key_pressed(rl.KeyboardKey.KEY_L):
            self.load_studio()

    def run(self) -> None:  # pragma: no cover - requires a display
        config = self.config
        rl.init_window(config.width, config.height, config.title)
        rl.set_target_fps(config.target_fps)

        camera = rl.Camera3D(
            rl.Vector3(
                config.camera_distance, config.camera_height, config.camera_distance
            ),
            rl.Vector3(0.0, config.camera_height, 0.0),
            rl.Vector3(0.0, 1.0, 0.0),
            45.0,
            rl.CameraProjection.CAMERA_PERSPECTIVE,
        )
        if self._mouse_look:
            rl.disable_cursor()

        try:
            while not rl.window_should_close():
                stick, facing = self._read_stick()

                if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE):
                    self._paused = not self._paused
                if rl.is_key_pressed(rl.KeyboardKey.KEY_C):
                    config.show_candidates = not config.show_candidates
                if rl.is_key_pressed(rl.KeyboardKey.KEY_T):
                    config.show_trajectory = not config.show_trajectory
                if rl.is_key_pressed(rl.KeyboardKey.KEY_F):
                    config.show_contacts = not config.show_contacts
                if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
                    self.controller.reset(row=self.controller.state.row)
                if rl.is_key_pressed(rl.KeyboardKey.KEY_M):
                    self._mouse_look = not self._mouse_look
                    if self._mouse_look:
                        rl.disable_cursor()
                    else:
                        rl.enable_cursor()

                self._handle_placement_keys()

                if not self._paused:
                    record = self.controller.step(stick, facing)
                    if self.writer is not None:
                        self.writer.append(stick, record)
                    positions = self._world_positions()
                elif self._last_record is not None:
                    record = self._last_record
                    positions = self._last_positions
                else:
                    record = self.controller.step(np.zeros(2, np.float32))
                    positions = self._world_positions()

                self._last_record = record
                self._last_positions = positions

                origin = self.controller.state.origin
                target = self.camera.target(origin)
                camera.target = self._vec(target)
                camera.position = self._vec(self.camera.eye(target))

                rl.begin_drawing()
                rl.clear_background(rl.Color(28, 30, 34, 255))
                rl.begin_mode_3d(camera)

                rl.draw_grid(config.grid_slices, config.grid_spacing)

                # Before the character, so the translucent fill blends over the
                # grid rather than over the skeleton.
                self._draw_boxes()

                if config.show_candidates and record.candidates is not None:
                    for ghost in self._candidate_positions(
                        self.database.next_frame_of(record.candidates)
                    ):
                        self._draw_skeleton(
                            ghost, rl.Color(120, 120, 140, 70), joint_radius=1.2
                        )

                self._draw_skeleton(positions, rl.Color(240, 240, 245, 255))

                if config.show_trajectory:
                    self._draw_trajectory(record)
                if config.show_contacts:
                    self._draw_contacts(positions)
                self._draw_support_marker()

                rl.end_mode_3d()
                self._draw_hud(record, stick)
                self._draw_obstacle_hud()
                rl.end_drawing()

                self._previous_positions = positions
        finally:
            if self.writer is not None and len(self.writer):
                manifest = self.writer.close()
                print(f"wrote {self.writer.path}: {manifest['frames']} frames")
            rl.close_window()
