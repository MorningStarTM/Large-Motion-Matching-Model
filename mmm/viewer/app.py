"""
The viewer application loop: raylib, driven by a gamepad.

Phase 0's "done when" is a character that walks, runs and turns smoothly on the
gamepad, which is something only the eye can confirm. So this draws the
character and, just as importantly, the overlays: the desired path, the query
points, the retrieved candidates as ghosts, and the foot contact state.
Watching the retrieved candidates is how a broken feature layout is spotted in
seconds rather than days.

The skeleton is drawn as bones and joints rather than with the Geno mesh. The
mesh needs GenoViewPython's model files; bones need nothing but the database,
so the viewer works on a fresh checkout and the mesh can be added later without
changing anything here.

Sources are switchable at runtime with a key: the KNN teacher, a plain clip
player, and later the model. Side by side comparison is how most animation bugs
are actually found.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from mmm.data import spaces
from mmm.data.contacts import CONTACT_JOINTS, ContactThresholds, detect_contacts
from mmm.data.database import MotionDatabase, pose_to_hips_height, pose_to_rotations
from mmm.data.fk import forward_kinematics
from mmm.teacher.controller import KnnController
from mmm.teacher.logger import RolloutWriter
from mmm.viewer.camera import OrbitCamera, camera_relative

__all__ = ["ViewerApp", "ViewerConfig"]

try:
    import pyray as rl
    from raylib import colors as rlcolors  # noqa: F401  (imported for availability check)

    RAYLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    rl = None
    RAYLIB_AVAILABLE = False


@dataclass
class ViewerConfig:
    width: int = 1280
    height: int = 720
    title: str = "MMM - Phase 0 KNN motion matching"
    target_fps: int = 60
    # LaFAN1 is in centimetres, so the camera and the grid are sized in
    # centimetres too. A 1 unit grid would be invisible.
    grid_spacing: float = 50.0
    grid_slices: int = 40
    camera_distance: float = 400.0
    camera_height: float = 100.0
    mouse_look: bool = True
    show_candidates: bool = True
    show_trajectory: bool = True
    show_contacts: bool = True


class ViewerApp:
    """Drives a source and draws it.

    Args:
        database: for the skeleton and for reading retrieved candidate poses.
        controller: the KNN controller to drive. Phase 0's only source.
        config: window and overlay settings.
        log_path: if given, every frame is logged through `RolloutWriter`, so
            an interactive session also produces Phase 2 training data.
        seed: recorded in the rollout manifest.
    """

    def __init__(
        self,
        database: MotionDatabase,
        controller: KnnController,
        config: ViewerConfig | None = None,
        log_path: str | None = None,
        seed: int = 0,
    ):
        if not RAYLIB_AVAILABLE:
            raise ImportError(
                "raylib is not installed, so the viewer cannot start. "
                "Install it with `pip install -r requirements.txt`, or run "
                "scripts/run_teacher.py --headless to drive the teacher with "
                "the synthetic joystick instead."
            )

        self.database = database
        self.controller = controller
        self.config = config or ViewerConfig()
        self.skeleton = database.skeleton
        self.n_joints = self.skeleton.n_joints

        self.writer = None
        if log_path:
            self.writer = RolloutWriter(
                path=log_path,
                database=database,
                config={
                    "search_interval": controller.search_interval,
                    "k": controller.k,
                    "transition_cost": controller.transition_cost,
                    "source": "gamepad",
                },
                seed=seed,
            )

        self._paused = False
        self.camera = OrbitCamera(
            distance=self.config.camera_distance, height=self.config.camera_height
        )
        self._mouse_look = self.config.mouse_look
        # Set before the loop so pausing on the very first frame -- possible if
        # the key is already down at startup -- cannot read them unassigned.
        self._last_record = None
        self._last_positions = None
        self._contact_thresholds = ContactThresholds()
        self._previous_positions: np.ndarray | None = None

    # -- input ------------------------------------------------------------

    def _read_stick(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Player input, mapped through the camera into world space.

        Returns `(stick, facing)` in world XZ, which is what the controller
        takes. The movement stick is read in camera space and rotated by the
        camera's azimuth, so swinging the camera changes where the character
        walks -- without that, moving the mouse turns the view and the
        character keeps walking north, which looks exactly like turning being
        broken.

        While the aim button is held the facing request follows the camera
        instead of the direction of travel, which is how turning on the spot
        and strafing are requested.
        """
        from mmm.viewer.gamepad import read_input

        frame = read_input(mouse_look=self._mouse_look)
        self.camera.update(frame.look, frame.zoom, self.database.frame_time)

        stick = camera_relative(frame.move, self.camera.azimuth)
        facing = self.camera.forward_xz if frame.aim else None
        return stick, facing

    # -- geometry ---------------------------------------------------------

    def _world_positions(self) -> np.ndarray:
        state = self.controller.state
        root = np.array([[0.0, state.hips_height, 0.0]], np.float32)
        local, _ = forward_kinematics(self.skeleton, state.local_rotations[None], root)
        frame = spaces.RootFrame(origin=state.origin[None], forward=state.forward[None])
        return spaces.from_character_space(local, frame)[0]

    def _candidate_positions(self, rows) -> list[np.ndarray]:
        """Retrieved candidate poses, placed at the character's own frame.

        Drawn at the character rather than wherever their source clip was, so
        the comparison that matters -- how close the candidate pose is to the
        current pose -- is the one the eye makes.
        """
        state = self.controller.state
        frame = spaces.RootFrame(origin=state.origin[None], forward=state.forward[None])
        out = []
        for row in rows:
            if row < 0:
                continue
            pose = self.database.animation.poses[row]
            rotations = pose_to_rotations(pose, self.n_joints)
            height = float(pose_to_hips_height(pose, self.n_joints))
            local, _ = forward_kinematics(
                self.skeleton, rotations[None], np.array([[0.0, height, 0.0]], np.float32)
            )
            out.append(spaces.from_character_space(local, frame)[0])
        return out

    # -- drawing ----------------------------------------------------------

    @staticmethod
    def _vec(point) -> "rl.Vector3":
        return rl.Vector3(float(point[0]), float(point[1]), float(point[2]))

    def _draw_skeleton(self, positions: np.ndarray, color, joint_radius: float = 2.5) -> None:
        for joint in range(self.n_joints):
            parent = int(self.skeleton.parents[joint])
            if parent >= 0:
                rl.draw_line_3d(self._vec(positions[parent]), self._vec(positions[joint]), color)
            rl.draw_sphere(self._vec(positions[joint]), joint_radius, color)

    def _draw_trajectory(self, record) -> None:
        state = self.controller.state
        positions, directions = self.controller.trajectory.predict(state.forward)
        frame = spaces.RootFrame(origin=state.origin[None], forward=state.forward[None])

        for i in range(len(positions)):
            point = np.array([positions[i][0], 0.0, positions[i][1]], np.float32)
            world = spaces.from_character_space(point[None, None], frame)[0, 0]
            rl.draw_sphere(self._vec(world), 6.0, rl.Color(80, 200, 255, 255))

            direction = np.array([directions[i][0], 0.0, directions[i][1]], np.float32)
            tip = world + direction * 30.0
            rl.draw_line_3d(self._vec(world), self._vec(tip), rl.Color(80, 200, 255, 255))

        rl.draw_line_3d(
            self._vec(state.origin),
            self._vec(state.origin + np.array([state.forward[0], 0.0, state.forward[1]]) * 60.0),
            rl.Color(255, 220, 60, 255),
        )

    def _draw_contacts(self, positions: np.ndarray) -> None:
        if self._previous_positions is None:
            return
        stacked = np.stack([self._previous_positions, positions])
        contacts = detect_contacts(
            stacked, self.skeleton, self.database.frame_time, self._contact_thresholds
        )[-1]
        for foot, name in enumerate(CONTACT_JOINTS):
            if not self.skeleton.has(name):
                continue
            point = positions[self.skeleton.index(name)]
            color = rl.Color(90, 230, 120, 255) if contacts[foot] else rl.Color(230, 90, 90, 160)
            rl.draw_sphere(self._vec(point), 5.0, color)

    def _draw_hud(self, record, stick: np.ndarray) -> None:
        state = self.controller.state
        clip = self.database.clip_of(state.row)
        speed = float(np.linalg.norm(state.linear_velocity[[0, 2]]))
        lines = [
            f"FPS {rl.get_fps()}   {'PAUSED' if self._paused else ''}",
            f"row {state.row}  clip {clip.name}  frame {int(self.database.matching.frame_in_clip[state.row])}",
            f"speed {speed:7.1f} u/s   stick {stick[0]:+.2f},{stick[1]:+.2f}",
            f"searched {record.searched}  jumped {record.jumped}",
        ]
        if record.distances is not None and len(record.distances):
            lines.append("distances " + "  ".join(f"{d:.2f}" for d in record.distances))
        lines.append(
            f"azimuth {np.degrees(self.camera.azimuth):+6.0f}  "
            f"dist {self.camera.distance:4.0f}  "
            f"mouse-look {'on' if self._mouse_look else 'off'}  "
            f"root {self.controller.root_source}"
        )
        lines.append(
            "move: stick/WASD (camera relative)   turn: mouse or right stick   "
            "aim: RMB/L1"
        )
        lines.append(
            "[space] pause  [c] candidates  [t] trajectory  [f] contacts  "
            "[r] reset  [m] mouse-look"
        )

        for i, line in enumerate(lines):
            rl.draw_text(line, 12, 12 + i * 20, 18, rl.Color(235, 235, 235, 255))

    # -- the loop ---------------------------------------------------------

    def run(self) -> None:
        config = self.config
        rl.init_window(config.width, config.height, config.title)
        rl.set_target_fps(config.target_fps)

        camera = rl.Camera3D(
            rl.Vector3(config.camera_distance, config.camera_height, config.camera_distance),
            rl.Vector3(0.0, config.camera_height, 0.0),
            rl.Vector3(0.0, 1.0, 0.0),
            45.0,
            rl.CameraProjection.CAMERA_PERSPECTIVE,
        )
        if self._mouse_look:
            # Hide the cursor so the mouse reads as a look control rather than
            # a pointer. `m` releases it, and Escape still closes the window.
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

                # Follow the character, so it does not walk out of frame.
                origin = self.controller.state.origin
                target = self.camera.target(origin)
                camera.target = self._vec(target)
                camera.position = self._vec(self.camera.eye(target))

                rl.begin_drawing()
                rl.clear_background(rl.Color(28, 30, 34, 255))
                rl.begin_mode_3d(camera)

                rl.draw_grid(config.grid_slices, config.grid_spacing)

                if config.show_candidates and record.candidates is not None:
                    for ghost in self._candidate_positions(
                        self.database.next_frame_of(record.candidates)
                    ):
                        self._draw_skeleton(ghost, rl.Color(120, 120, 140, 70), joint_radius=1.2)

                self._draw_skeleton(positions, rl.Color(240, 240, 245, 255))

                if config.show_trajectory:
                    self._draw_trajectory(record)
                if config.show_contacts:
                    self._draw_contacts(positions)

                rl.end_mode_3d()
                self._draw_hud(record, stick)
                rl.end_drawing()

                self._previous_positions = positions
        finally:
            if self.writer is not None and len(self.writer):
                manifest = self.writer.close()
                print(f"wrote {self.writer.path}: {manifest['frames']} frames")
            rl.close_window()
