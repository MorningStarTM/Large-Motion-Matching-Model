"""
The Geno viewer: our motion matching controller, drawn on Holden's character.

This is a second viewer, not a replacement. `mmm/viewer/app.py` and
`mmm/obstacles/viewer.py` are untouched and still work; a wireframe is better
for debugging the search -- you can see the candidate ghosts through it -- and
a skinned character is better for judging the motion. They answer different
questions, so both stay.

What the character buys
-----------------------
Foot sliding and ground penetration are the two artefacts motion matching
actually produces, and both are nearly impossible to judge on a stick figure:
there is no surface to read a foot against and no contact shadow to say where
the ground is. On a skinned character with a shadow and ambient occlusion they
are obvious. That is the argument in GenoView's README, and it is the reason
this is worth a deferred renderer rather than just a mesh.

Units
-----
Everything in this project is centimetres; `Geno.bin` and the renderer's
lighting constants are metres. The conversion happens at exactly one boundary:
poses go into `GenoCharacter.update` in centimetres and come out as metre-space
bone matrices. The obstacle boxes are converted at the point they are drawn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from mmm.data import spaces
from mmm.data.fk import forward_kinematics
from mmm.data.skeleton import LOCOMOTION_JOINTS
from mmm.obstacles.geometry import BoxField
from mmm.viewer.camera import OrbitCamera, camera_relative
from mmm.viewer.geno.model import (
    METRES_PER_UNIT,
    BoneMapping,
    GenoMesh,
    default_model_path,
    load_bind_reference,
    load_geno_mesh,
    skinning_matrices_for_bones,
)
from mmm.viewer.geno.renderer import RAYLIB_AVAILABLE, DeferredRenderer, RenderSettings

if RAYLIB_AVAILABLE:  # pragma: no cover - requires a display
    import cffi

    # Star import first; see the note in renderer.py -- the raylib package
    # exports its own `rl`, which would otherwise shadow pyray's.
    from raylib import *  # noqa: F403

    import pyray as rl  # noqa: E402
    from pyray import Color, Rectangle, Vector2, Vector3  # noqa: E402

    ffi = cffi.FFI()

__all__ = ["GenoCharacter", "GenoViewerConfig", "GenoViewerApp"]


@dataclass
class GenoViewerConfig:
    width: int = 1280
    height: int = 720
    title: str = "MMM - Geno viewer"
    target_fps: int = 60
    camera_distance: float = 400.0
    camera_height: float = 100.0
    mouse_look: bool = True
    show_skeleton: bool = False
    show_boxes: bool = True
    ground_size: float = 40.0  # metres, the plane the shadow falls on


class GenoCharacter:
    """The skinned character, posed from our joint rotations.

    Holds the CPU side (`GenoMesh`, the bind transforms, the name mapping) and
    the GPU side (a raylib `Model` whose bone matrices are rewritten each
    frame). Splitting the maths into `model.py` means everything here is
    plumbing, and the plumbing is the part that cannot be unit tested.
    """

    def __init__(self, path: str | None = None, joint_names=LOCOMOTION_JOINTS):
        self.mesh: GenoMesh = load_geno_mesh(path)
        self.mapping = BoneMapping.by_name(self.mesh, joint_names)
        self.reference = load_bind_reference(self.mesh, self.mapping,
                                             joint_names=joint_names)
        self.model = self._upload(path or default_model_path())
        self._matrices = np.zeros((self.mesh.n_bones, 4, 4), np.float32)

    def _upload(self, path):  # pragma: no cover - requires a GL context
        """Build a raylib `Model` around the mesh.

        Two cffi details worth stating, because both fail confusingly:

        - Struct sizes come from an **instance**, `ffi.sizeof(rl.Mesh())`, not
          from the type name. A bare `cffi.FFI()` has no declarations of its
          own, so `ffi.sizeof("Mesh")` raises `unknown identifier`.
        - The `void *` that `mem_alloc` returns is assigned straight to the
          struct field. cffi converts it on assignment, and casting it by name
          would hit the same unknown-identifier problem.

        raylib owns the memory once `upload_mesh` runs, so the arrays are
        copied into buffers it allocated rather than left pointing at NumPy's,
        which Python would be free to free.
        """
        mesh = self.mesh
        model = rl.Model()
        model.transform = rl.matrix_identity()

        model.materialCount = 1
        model.materials = rl.mem_alloc(ffi.sizeof(rl.Material()))
        model.materials[0] = rl.load_material_default()

        model.meshCount = 1
        model.meshMaterial = rl.mem_alloc(ffi.sizeof("int"))
        model.meshMaterial[0] = 0
        model.meshes = rl.mem_alloc(ffi.sizeof(rl.Mesh()))

        model.meshes[0].vertexCount = mesh.n_vertices
        model.meshes[0].triangleCount = mesh.n_triangles
        model.meshes[0].boneCount = mesh.n_bones
        model.boneCount = mesh.n_bones

        def upload(field, array):
            raw = array.tobytes()
            pointer = rl.mem_alloc(len(raw))
            ffi.memmove(pointer, raw, len(raw))
            setattr(model.meshes[0], field, pointer)

        upload("vertices", mesh.positions)
        upload("texcoords", mesh.texcoords)
        upload("normals", mesh.normals)
        upload("boneIds", mesh.bone_ids)
        upload("boneWeights", mesh.bone_weights)
        upload("indices", mesh.triangles)
        # raylib writes CPU-skinned results into these; they must exist and
        # start as a copy of the bind pose even though the GPU path is used.
        upload("animVertices", mesh.positions)
        upload("animNormals", mesh.normals)

        model.meshes[0].boneMatrices = rl.mem_alloc(
            mesh.n_bones * ffi.sizeof(rl.Matrix())
        )
        for bone in range(mesh.n_bones):
            model.meshes[0].boneMatrices[bone] = rl.matrix_identity()

        rl.upload_mesh(ffi.addressof(model.meshes[0]), True)
        return model

    def update(self, skeleton, local_rotations: np.ndarray, root_position_cm):
        """Pose the character from one frame of our motion.

        Our own forward kinematics does the work -- the same function the
        wireframe viewer and the loss use, already tested -- and the result is
        turned into skinning matrices against the bind reference. Nothing here
        re-derives kinematics, which is deliberate: a second implementation
        would be a second thing to keep correct.

        Args:
            skeleton: our 24 joint skeleton.
            local_rotations: `(24, 3, 3)` world-space local rotations, hips
                included, in our joint order.
            root_position_cm: `(3,)` world hips position, centimetres.

        Returns:
            `(positions, rotations)` for our 24 joints, in metres, so the
            caller can aim the shadow without redoing the work.
        """
        positions, rotations = forward_kinematics(
            skeleton,
            np.asarray(local_rotations, np.float32)[None],
            np.asarray(root_position_cm, np.float32)[None],
        )
        positions = positions[0] * METRES_PER_UNIT
        rotations = rotations[0]

        self._matrices = skinning_matrices_for_bones(
            self.reference, positions, rotations
        )
        self._write_bone_matrices()
        return positions, rotations

    def _write_bone_matrices(self):  # pragma: no cover - requires a GL context
        """Copy the skinning matrices into the mesh's GPU-bound buffer.

        Written straight through the ffi buffer rather than element by
        element: 75 matrices is 1,200 floats a frame, and going through
        pyray's property setters costs more than the skinning itself.

        **No transpose**, which is worth justifying because the opposite looks
        right. raylib's `Matrix` is described as column-major, but its struct
        declares the fields in the order `m0, m4, m8, m12` then
        `m1, m5, m9, m13` and so on, so the first four floats in memory are the
        first *row* of the mathematical matrix. A row-major NumPy view over
        that memory is therefore already `M[row][col]`, and
        `skinning_matrices` returns exactly that. Transposing here produces a
        character folded through the origin.
        """
        mesh = self.model.meshes[0]
        view = np.frombuffer(
            ffi.buffer(mesh.boneMatrices, self.mesh.n_bones * 16 * 4), np.float32
        ).reshape(self.mesh.n_bones, 4, 4)
        view[:] = self._matrices

    def unload(self):  # pragma: no cover - requires a GL context
        rl.unload_model(self.model)


class GenoViewerApp:  # pragma: no cover - requires a display
    """Drives a controller and renders the Geno character.

    Args:
        database: for the skeleton and the pose layout.
        controller: a `KnnController` or `ObstacleController`.
        field: obstacles to draw. Taken from the controller when it has one.
        config: window settings.
        settings: lighting settings.
    """

    def __init__(
        self,
        database,
        controller,
        field: BoxField | None = None,
        config: GenoViewerConfig | None = None,
        settings: RenderSettings | None = None,
        model_path: str | None = None,
    ):
        if not RAYLIB_AVAILABLE:
            raise ImportError(
                "raylib is not installed, so the Geno viewer cannot start. "
                "Install it with `pip install raylib`."
            )

        self.database = database
        self.controller = controller
        self.config = config or GenoViewerConfig()
        self.settings = settings or RenderSettings()
        self.skeleton = database.skeleton
        self.n_joints = self.skeleton.n_joints
        self.field = field if field is not None else getattr(
            controller, "field", BoxField()
        )
        self.model_path = model_path

        self.camera = OrbitCamera(
            distance=self.config.camera_distance, height=self.config.camera_height
        )
        self._mouse_look = self.config.mouse_look
        self._paused = False
        self._frame = 0

    # -- input ------------------------------------------------------------

    def _read_stick(self):
        """Player input, mapped through the camera into world space.

        The same mapping `ViewerApp` uses -- duplicated rather than shared
        because the two viewers own their own cameras, and a mapping that
        silently read the other viewer's azimuth would send the character
        somewhere unrelated to where the player is looking.
        """
        from mmm.viewer.gamepad import read_input

        frame = read_input(mouse_look=self._mouse_look)
        self.camera.update(frame.look, frame.zoom, self.database.frame_time)
        stick = camera_relative(frame.move, self.camera.azimuth)
        facing = self.camera.forward_xz if frame.aim else None
        return stick, facing

    def _world_pose(self):
        """The controller's pose, in world space, ready for the character.

        Two conversions the wireframe viewer also does, and both are easy to
        miss. The stored hips rotation is **yaw stripped** -- it lives in
        character space, so the facing has to be put back or the character
        walks sideways while its body points north. And the hips' world
        position is the root frame's origin lifted by the stored hips height,
        because the frame itself is always on the floor.
        """
        state = self.controller.state
        rotations = np.array(state.local_rotations, np.float32).copy()
        rotations[0] = spaces.apply_yaw(rotations[0], state.forward)
        position = np.array(
            [state.origin[0], state.hips_height, state.origin[2]], np.float32
        )
        return rotations, position

    # -- drawing ----------------------------------------------------------

    @staticmethod
    def _vec_m(point):
        """Our centimetres to the renderer's metres."""
        return Vector3(
            float(point[0]) * METRES_PER_UNIT,
            float(point[1]) * METRES_PER_UNIT,
            float(point[2]) * METRES_PER_UNIT,
        )

    def _draw_boxes(self, colour):
        """Obstacles, as solid cubes. Metres.

        Drawn as a scaled unit-cube **model**, not with `draw_cube`.
        `draw_cube` goes through rlgl's immediate-mode batch, which uses
        raylib's own default shader; inside the GBuffer pass that shader never
        writes the second colour attachment, so immediate-mode geometry
        contributes no normal and the lighting pass discards it. The boxes
        simply do not appear, with no error. Going through a model means they
        use the same `basic` shader as the ground and the character.
        """
        for box in self.field:
            centre = Vector3(
                float((box.x0 + box.x1) / 2) * METRES_PER_UNIT,
                float((box.base + box.top) / 2) * METRES_PER_UNIT,
                float((box.z0 + box.z1) / 2) * METRES_PER_UNIT,
            )
            scale = Vector3(
                float(box.x1 - box.x0) * METRES_PER_UNIT,
                float(box.top - box.base) * METRES_PER_UNIT,
                float(box.z1 - box.z0) * METRES_PER_UNIT,
            )
            rl.draw_model_ex(
                self._cube_model, centre, Vector3(0.0, 1.0, 0.0), 0.0, scale, colour
            )

    def _draw_scene(self, ground_model, character, shadow_pass: bool):
        """Everything that is in the world, drawn once per geometry pass.

        One method for both the shadow pass and the GBuffer pass, because the
        passes must agree about what exists -- an object drawn into one and not
        the other either casts a shadow from nothing or fails to cast one at
        all.
        """
        if shadow_pass:
            ground_model.materials[0].shader = self.renderer.shadow_shader
            character.model.materials[0].shader = self.renderer.skinned_shadow_shader
            box_colour = WHITE
            ground_colour = WHITE
            character_colour = WHITE
        else:
            ground_model.materials[0].shader = self.renderer.basic_shader
            character.model.materials[0].shader = self.renderer.skinned_basic_shader
            box_colour = Color(120, 170, 220, 255)
            ground_colour = Color(190, 190, 190, 255)
            character_colour = ORANGE

        rl.draw_model(ground_model, Vector3(0.0, -0.01, 0.0), 1.0, ground_colour)
        rl.draw_model(character.model, Vector3(0.0, 0.0, 0.0), 1.0, character_colour)

        if self.config.show_boxes and len(self.field):
            self._cube_model.materials[0].shader = (
                self.renderer.shadow_shader if shadow_pass
                else self.renderer.basic_shader
            )
            self._draw_boxes(box_colour)

    def _draw_hud(self, record):
        state = self.controller.state
        lines = [
            f"frame {self._frame}   row {state.row}   "
            f"{'PAUSED' if self._paused else 'running'}",
            f"position  {state.origin[0]:7.1f} {state.origin[2]:7.1f} cm",
        ]
        if hasattr(self.controller, "support_under_character"):
            distance, top = self.controller.edge_ahead()
            lines.append(
                f"support {self.controller.support_under_character():6.1f} cm   "
                f"next change {distance:6.1f} cm -> {top:6.1f} cm"
            )
        lines.append("WASD move   mouse turn   space pause   K skeleton   M mouse")
        for i, line in enumerate(lines):
            rl.draw_text(line, 12, 12 + i * 20, 18, Color(40, 40, 45, 255))

    # -- the loop ---------------------------------------------------------

    def run(self, max_frames=None, screenshot=None, auto_stick=None):
        """Open a window and drive the controller.

        Args:
            max_frames: stop after this many frames. Used by the self test,
                which needs the renderer to actually run but cannot sit in an
                interactive loop.
            screenshot: write a PNG here just before stopping. The only way to
                check a renderer without watching it.
            auto_stick: a fixed `(x, z)` stick to feed instead of reading
                input. Needed for a headless render to show anything: with no
                input the controller sits on row 0, and row 0 of a LaFAN1 clip
                is the calibration T-pose, so the screenshot looks like a
                broken skinning bug when nothing is wrong.
        """
        config = self.config
        rl.set_config_flags(FLAG_MSAA_4X_HINT)
        rl.init_window(config.width, config.height, config.title.encode("utf-8"))
        rl.set_target_fps(config.target_fps)

        # Render at raylib's own reported size, which is what its viewport
        # is set to. It is tempting to scale this by `get_window_scale_dpi()`
        # on a scaled display -- 1.25x is the Windows default on many laptops
        # -- but that is wrong: raylib draws into a 1280x720 viewport whatever
        # the physical surface is, and enlarging the deferred targets only
        # makes them disagree with it. The oversized screenshots that suggest
        # otherwise are an artefact of `take_screenshot`, which captures the
        # whole physical surface including the part raylib never draws to; a
        # plain clear-the-screen program shows the same margin.
        self.render_width = rl.get_render_width()
        self.render_height = rl.get_render_height()

        self.renderer = DeferredRenderer(
            self.render_width, self.render_height, self.settings
        )
        character = GenoCharacter(self.model_path)

        ground_model = rl.load_model_from_mesh(
            rl.gen_mesh_plane(config.ground_size, config.ground_size, 10, 10)
        )
        # One unit cube, scaled per obstacle. See `_draw_boxes` for why the
        # boxes are models rather than immediate-mode cubes.
        self._cube_model = rl.load_model_from_mesh(rl.gen_mesh_cube(1.0, 1.0, 1.0))

        camera = rl.Camera3D(
            Vector3(2.0, 3.0, 5.0),
            Vector3(0.0, 1.0, 0.0),
            Vector3(0.0, 1.0, 0.0),
            45.0,
            CAMERA_PERSPECTIVE,
        )
        if self._mouse_look:
            rl.disable_cursor()

        try:
            while not rl.window_should_close():
                if max_frames is not None and self._frame >= max_frames:
                    break

                if auto_stick is None:
                    stick, facing = self._read_stick()
                else:
                    stick = np.asarray(auto_stick, np.float32)
                    facing = None
                    self.camera.update(np.zeros(2, np.float32), 0.0,
                                       self.database.frame_time)

                if rl.is_key_pressed(KEY_SPACE):
                    self._paused = not self._paused
                if rl.is_key_pressed(KEY_K):
                    config.show_skeleton = not config.show_skeleton
                if rl.is_key_pressed(KEY_B):
                    config.show_boxes = not config.show_boxes
                if rl.is_key_pressed(KEY_M):
                    self._mouse_look = not self._mouse_look
                    rl.disable_cursor() if self._mouse_look else rl.enable_cursor()

                if not self._paused:
                    record = self.controller.step(stick, facing)
                else:
                    record = getattr(self, "_last_record", None)
                    if record is None:
                        record = self.controller.step(np.zeros(2, np.float32))
                self._last_record = record

                state = self.controller.state
                rotations, hips_world = self._world_pose()
                joint_positions, _ = character.update(
                    self.skeleton, rotations, hips_world
                )


                hips = joint_positions[0]
                self.renderer.light.track(hips[0], hips[2])

                target = self.camera.target(state.origin)
                camera.target = self._vec_m(target)
                camera.position = self._vec_m(self.camera.eye(target))

                rl.rl_disable_color_blend()
                rl.begin_drawing()

                with self.renderer.shadow_pass():
                    self._draw_scene(ground_model, character, shadow_pass=True)

                with self.renderer.gbuffer_pass(camera):
                    self._draw_scene(ground_model, character, shadow_pass=False)

                self.renderer.ssao_pass(camera)

                with self.renderer.lighting_pass(camera):
                    if config.show_skeleton:
                        rl.begin_mode_3d(camera)
                        self._draw_bones(joint_positions)
                        rl.end_mode_3d()

                self.renderer.present()

                rl.rl_enable_color_blend()
                self._draw_hud(record)
                rl.end_drawing()

                self._frame += 1

            if screenshot:
                # raylib writes the screenshot relative to the working
                # directory and ignores any directory in the name, so it is
                # taken under its basename and moved afterwards. Without this
                # a path like /tmp/shot.png silently lands in the repo root.
                target = os.path.abspath(screenshot)
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                basename = os.path.basename(target)
                rl.take_screenshot(basename.encode("utf-8"))
                written = os.path.join(os.getcwd(), basename)
                if os.path.abspath(written) != target and os.path.isfile(written):
                    os.replace(written, target)
                print(f"wrote {target}")
        finally:
            character.unload()
            rl.unload_model(ground_model)
            rl.unload_model(self._cube_model)
            self.renderer.unload()
            rl.close_window()

    def _draw_bones(self, positions):
        """Our 24 joints, over the lit character. Metres."""
        colour = Color(90, 90, 100, 255)
        for joint, parent in enumerate(self.skeleton.parents):
            rl.draw_sphere(Vector3(*positions[joint]), 0.012, colour)
            if parent >= 0:
                rl.draw_line_3d(
                    Vector3(*positions[joint]),
                    Vector3(*positions[int(parent)]),
                    colour,
                )
