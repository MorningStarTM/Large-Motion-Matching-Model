"""
The deferred renderer, ported from Daniel Holden's GenoView.

Adapted from GenoViewPython (MIT, Daniel Holden). The shaders in `resources/`
are his, unmodified; this file is a restructuring of his `genoview.py` main
loop into something another module can drive. The attribution is in
`NOTICE.md` and `licenses/GenoViewPython-LICENSE`.

Why a deferred renderer at all, when the wireframe viewer already works: the
point of this is to make artefacts *visible*. Foot sliding and ground
penetration are nearly impossible to judge on a stick figure, because there is
no surface and no contact shadow to read them against. A skinned character
with a shadow and ambient occlusion makes both obvious at a glance -- that is
the entire argument in GenoView's README, and it is why this is worth the
pipeline rather than just drawing a mesh.

The passes, in order:

1. **Shadow map.** The scene from the sun's point of view, depth only.
2. **GBuffer.** Colour and view-space normal and depth, in one pass with two
   colour attachments.
3. **SSAO.** Ambient occlusion plus the shadow lookup, into a half-strength
   mask, then separably blurred twice (horizontal then vertical) with a
   depth-aware kernel so the blur does not bleed across silhouettes.
4. **Lighting.** A full screen pass combining the GBuffer, the occlusion mask,
   a sun, a sky term and a ground bounce.
5. **FXAA.** Cheap edge antialiasing, because the character's silhouette
   against a flat floor is exactly where aliasing shows.

The structure here differs from his in one way that matters: everything lives
on a `DeferredRenderer` that owns its framebuffers and shaders, so the caller
writes `with renderer.frame(camera):` and draws, instead of hand-sequencing
eleven `Begin*`/`End*` pairs. That keeps the sequencing bug -- forgetting an
`EndTextureMode`, which silently renders the rest of the frame into the wrong
target -- impossible rather than merely unlikely.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from mmm.viewer.geno.model import default_resource_dir

try:  # pragma: no cover - depends on the environment
    import cffi

    # `from raylib import *` must come FIRST. The raylib package exports a name
    # `rl` of its own -- the raw cffi library object -- so importing it after
    # `import pyray as rl` silently replaces pyray's snake_case API with the
    # cffi handle, and the first call fails with "cffi library has no function
    # named set_config_flags". Star import first, alias second.
    from raylib import *  # noqa: F403  (raylib's API is a flat namespace)

    import pyray as rl  # noqa: E402  (deliberately after the star import)
    from pyray import (  # noqa: E402
        Camera3D,
        Color,
        Matrix,
        Rectangle,
        RenderTexture,
        Texture,
        Vector2,
        Vector3,
    )

    RAYLIB_AVAILABLE = True
    ffi = cffi.FFI()
except ImportError:  # pragma: no cover - depends on the environment
    rl = None
    RAYLIB_AVAILABLE = False
    ffi = None

__all__ = ["DeferredRenderer", "ShadowLight", "RenderSettings", "RAYLIB_AVAILABLE"]


@dataclass
class RenderSettings:
    """Lighting and post-processing constants.

    The defaults are Holden's, kept because they are what produces the look in
    GenoView's screenshots and because they are jointly tuned -- raising the
    exposure without dropping the sun strength washes the character out.

    Attributes:
        shadow_resolution: shadow map size, square.
        shadow_extent: half-width of the sun's orthographic frustum, in
            metres. Smaller is sharper but clips the shadow sooner; 5 m covers
            a character and the ground around it.
        light_direction: the sun, pointing *from* the light.
        specularity, glossiness: the material response in `basic.fs`.
        exposure: final tone scale.
    """

    shadow_resolution: int = 1024
    shadow_extent: float = 5.0
    shadow_distance: float = 5.0
    shadow_near: float = 0.01
    shadow_far: float = 10.0

    light_direction: tuple = (0.35, -1.0, -0.35)
    sun_colour: tuple = (253 / 255, 255 / 255, 232 / 255)
    sun_strength: float = 0.25
    sky_colour: tuple = (174 / 255, 183 / 255, 190 / 255)
    sky_strength: float = 0.15
    ground_strength: float = 0.1
    ambient_strength: float = 1.0
    exposure: float = 0.9

    specularity: float = 0.5
    glossiness: float = 10.0

    clip_near: float = 0.01
    clip_far: float = 50.0


class ShadowLight:
    """The sun, as an orthographic camera."""

    def __init__(self, settings: RenderSettings):
        direction = rl.vector3_normalize(Vector3(*settings.light_direction))
        self.direction = direction
        self.target = Vector3(0.0, 0.0, 0.0)
        self.position = rl.vector3_scale(direction, -settings.shadow_distance)
        self.up = Vector3(0.0, 1.0, 0.0)
        self.width = settings.shadow_extent
        self.height = settings.shadow_extent
        self.near = settings.shadow_near
        self.far = settings.shadow_far
        self._distance = settings.shadow_distance

    def track(self, x: float, z: float) -> None:
        """Follow the character, so the shadow map stays over it.

        A fixed shadow frustum would mean the character walks out of its own
        shadow after a few metres, which looks like the shadow switching off.
        """
        self.target = Vector3(float(x), 0.0, float(z))
        self.position = rl.vector3_add(
            self.target, rl.vector3_scale(self.direction, -self._distance)
        )


class _GBuffer:
    """Colour, normal and depth in one framebuffer.

    Two colour attachments rather than one: the lighting pass needs the
    world normal as well as the albedo, and reconstructing a normal from depth
    gives faceted results on a skinned mesh.
    """

    def __init__(self, width: int, height: int):
        self.id = rl.rl_load_framebuffer()
        if not self.id:
            raise RuntimeError("could not create the GBuffer framebuffer")

        rl.rl_enable_framebuffer(self.id)

        self.color = Texture()
        self.color.id = rl.rl_load_texture(
            ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1
        )
        self.color.width, self.color.height = width, height
        self.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
        self.color.mipmaps = 1
        rl.rl_framebuffer_attach(
            self.id, self.color.id, RL_ATTACHMENT_COLOR_CHANNEL0,
            RL_ATTACHMENT_TEXTURE2D, 0,
        )

        # 16 bit per channel for the normal: 8 bits visibly bands the ambient
        # occlusion on a smoothly curved surface like a shoulder.
        self.normal = Texture()
        self.normal.id = rl.rl_load_texture(
            ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1
        )
        self.normal.width, self.normal.height = width, height
        self.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
        self.normal.mipmaps = 1
        rl.rl_framebuffer_attach(
            self.id, self.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1,
            RL_ATTACHMENT_TEXTURE2D, 0,
        )

        self.depth = Texture()
        self.depth.id = rl.rl_load_texture_depth(width, height, False)
        self.depth.width, self.depth.height = width, height
        self.depth.format = 19  # DEPTH_COMPONENT_24BIT
        self.depth.mipmaps = 1
        rl.rl_framebuffer_attach(
            self.id, self.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0
        )

        if not rl.rl_framebuffer_complete(self.id):
            raise RuntimeError("the GBuffer framebuffer is incomplete")
        rl.rl_disable_framebuffer()

    def unload(self) -> None:
        if self.id:
            rl.rl_unload_framebuffer(self.id)
            self.id = 0


def _mult_matrix(matrix) -> None:
    """Multiply the current rlgl matrix by `matrix`.

    `matrix_to_float_v` returns a struct whose `.v` is a `float[16]`, while
    pyray's `rl_mult_matrixf` wants a `float *`; the cast bridges them. The
    intermediate is held in a local because cffi frees it as soon as nothing
    references it, and a freed pointer here would multiply by garbage.
    """
    flat = rl.matrix_to_float_v(matrix)
    rl.rl_mult_matrixf(ffi.cast("float *", flat.v))


def _float(value):
    pointer = ffi.new("float*")
    pointer[0] = float(value)
    return pointer


def _int(value):
    pointer = ffi.new("int*")
    pointer[0] = int(value)
    return pointer


class DeferredRenderer:
    """Owns the framebuffers, shaders and passes.

    Args:
        width, height: the render resolution.
        settings: lighting constants.
        resource_dir: where the shaders live.
    """

    def __init__(self, width: int, height: int, settings=None, resource_dir=None):
        if not RAYLIB_AVAILABLE:
            raise ImportError(
                "raylib is not installed, so the Geno viewer cannot start. "
                "Install it with `pip install raylib`."
            )

        self.width, self.height = width, height
        self.settings = settings or RenderSettings()
        self.resources = resource_dir or default_resource_dir()

        rl.rl_set_clip_planes(self.settings.clip_near, self.settings.clip_far)

        self.light = ShadowLight(self.settings)
        self._light_vector = Vector3(*self.settings.light_direction)
        self._light_vector = rl.vector3_normalize(self._light_vector)

        self._load_shaders()

        self.shadow_map = self._load_shadow_map(
            self.settings.shadow_resolution, self.settings.shadow_resolution
        )
        self.gbuffer = _GBuffer(width, height)
        self.lighted = rl.load_render_texture(width, height)
        self.ssao_front = rl.load_render_texture(width, height)
        self.ssao_back = rl.load_render_texture(width, height)

        # Filled in per frame by `frame()`, read by the later passes.
        self._cam_view = None
        self._cam_proj = None
        self._light_view_proj = None

    # -- setup ------------------------------------------------------------

    def _path(self, name: str) -> bytes:
        full = os.path.join(self.resources, name)
        if not os.path.isfile(full):
            raise FileNotFoundError(
                f"{full} not found. The shaders come from GenoViewPython; see "
                "NOTICE.md."
            )
        return full.encode("utf-8")

    def _load_shaders(self) -> None:
        """Load every shader and look up every uniform once.

        `get_shader_location` is a string lookup into the compiled program and
        returns -1 for a name that is not there -- including a uniform the
        compiler removed because nothing read it. Caching the locations up
        front means a typo shows up here rather than as a silently ignored
        `set_shader_value` every frame.
        """
        load = rl.load_shader

        self.shadow_shader = load(self._path("shadow.vs"), self._path("shadow.fs"))
        self.skinned_shadow_shader = load(
            self._path("skinnedShadow.vs"), self._path("shadow.fs")
        )
        self.basic_shader = load(self._path("basic.vs"), self._path("basic.fs"))
        self.skinned_basic_shader = load(
            self._path("skinnedBasic.vs"), self._path("basic.fs")
        )
        self.lighting_shader = load(self._path("post.vs"), self._path("lighting.fs"))
        self.ssao_shader = load(self._path("post.vs"), self._path("ssao.fs"))
        self.blur_shader = load(self._path("post.vs"), self._path("blur.fs"))
        self.fxaa_shader = load(self._path("post.vs"), self._path("fxaa.fs"))

        def where(shader, name):
            return rl.get_shader_location(shader, name.encode("utf-8"))

        self._loc = {}
        for shader, key, names in (
            (self.shadow_shader, "shadow", ("lightClipNear", "lightClipFar")),
            (self.skinned_shadow_shader, "skinned_shadow",
             ("lightClipNear", "lightClipFar")),
            (self.basic_shader, "basic",
             ("specularity", "glossiness", "camClipNear", "camClipFar")),
            (self.skinned_basic_shader, "skinned_basic",
             ("specularity", "glossiness", "camClipNear", "camClipFar")),
            (self.lighting_shader, "lighting",
             ("gbufferColor", "gbufferNormal", "gbufferDepth", "ssao", "camPos",
              "camInvViewProj", "lightDir", "sunColor", "sunStrength",
              "skyColor", "skyStrength", "groundStrength", "ambientStrength",
              "exposure", "camClipNear", "camClipFar")),
            (self.ssao_shader, "ssao",
             ("gbufferNormal", "gbufferDepth", "camView", "camProj",
              "camInvProj", "camInvViewProj", "lightViewProj", "shadowMap",
              "shadowInvResolution", "camClipNear", "camClipFar",
              "lightClipNear", "lightClipFar", "lightDir")),
            (self.blur_shader, "blur",
             ("gbufferNormal", "gbufferDepth", "inputTexture", "camInvProj",
              "camClipNear", "camClipFar", "invTextureResolution",
              "blurDirection")),
            (self.fxaa_shader, "fxaa", ("inputTexture", "invTextureResolution")),
        ):
            for name in names:
                self._loc[f"{key}.{name}"] = where(shader, name)

    def _load_shadow_map(self, width: int, height: int):
        target = RenderTexture()
        target.id = rl.rl_load_framebuffer()
        if not target.id:
            raise RuntimeError("could not create the shadow map framebuffer")
        target.texture.width, target.texture.height = width, height

        rl.rl_enable_framebuffer(target.id)
        target.depth.id = rl.rl_load_texture_depth(width, height, False)
        target.depth.width, target.depth.height = width, height
        target.depth.format = 19
        target.depth.mipmaps = 1
        rl.rl_framebuffer_attach(
            target.id, target.depth.id, RL_ATTACHMENT_DEPTH,
            RL_ATTACHMENT_TEXTURE2D, 0,
        )
        if not rl.rl_framebuffer_complete(target.id):
            raise RuntimeError("the shadow map framebuffer is incomplete")
        rl.rl_disable_framebuffer()
        return target

    # -- passes -----------------------------------------------------------

    @contextmanager
    def shadow_pass(self):
        """Draw the scene from the sun. Depth only.

        Inside this block the caller should draw everything that casts a
        shadow, with `shadow_material` / `skinned_shadow_material` applied.
        """
        light = self.light
        rl.begin_texture_mode(self.shadow_map)
        rl.clear_background(WHITE)
        rl.rl_draw_render_batch_active()

        rl.rl_matrix_mode(RL_PROJECTION)
        rl.rl_push_matrix()
        rl.rl_load_identity()
        rl.rl_ortho(
            -light.width / 2, light.width / 2,
            -light.height / 2, light.height / 2,
            light.near, light.far,
        )
        rl.rl_matrix_mode(RL_MODELVIEW)
        rl.rl_load_identity()
        _mult_matrix(rl.matrix_look_at(light.position, light.target, light.up))
        rl.rl_enable_depth_test()

        self._light_view_proj = rl.matrix_multiply(
            rl.rl_get_matrix_modelview(), rl.rl_get_matrix_projection()
        )
        self._light_clip_near = rl.rl_get_cull_distance_near()
        self._light_clip_far = rl.rl_get_cull_distance_far()

        near, far = _float(self._light_clip_near), _float(self._light_clip_far)
        for key, shader in (
            ("shadow", self.shadow_shader),
            ("skinned_shadow", self.skinned_shadow_shader),
        ):
            rl.set_shader_value(
                shader, self._loc[f"{key}.lightClipNear"], near, SHADER_UNIFORM_FLOAT
            )
            rl.set_shader_value(
                shader, self._loc[f"{key}.lightClipFar"], far, SHADER_UNIFORM_FLOAT
            )

        try:
            yield
        finally:
            rl.rl_draw_render_batch_active()
            rl.rl_matrix_mode(RL_PROJECTION)
            rl.rl_pop_matrix()
            rl.rl_matrix_mode(RL_MODELVIEW)
            rl.rl_load_identity()
            rl.rl_disable_depth_test()
            rl.end_texture_mode()

    @contextmanager
    def gbuffer_pass(self, camera: "Camera3D"):
        """Draw the scene into colour + normal + depth."""
        target = self.gbuffer
        rl.rl_draw_render_batch_active()
        rl.rl_enable_framebuffer(target.id)
        rl.rl_active_draw_buffers(2)
        rl.rl_viewport(0, 0, target.color.width, target.color.height)
        rl.rl_set_framebuffer_width(target.color.width)
        rl.rl_set_framebuffer_height(target.color.height)
        rl.clear_background(BLACK)

        rl.rl_matrix_mode(RL_PROJECTION)
        rl.rl_push_matrix()
        rl.rl_load_identity()

        aspect = target.color.width / target.color.height
        near = rl.rl_get_cull_distance_near()
        far = rl.rl_get_cull_distance_far()
        top = near * np.tan(camera.fovy * 0.5 * np.pi / 180.0)
        right = top * aspect
        rl.rl_frustum(-right, right, -top, top, near, far)

        rl.rl_matrix_mode(RL_MODELVIEW)
        rl.rl_load_identity()
        _mult_matrix(rl.matrix_look_at(camera.position, camera.target, camera.up))
        rl.rl_enable_depth_test()

        self._cam_view = rl.rl_get_matrix_modelview()
        self._cam_proj = rl.rl_get_matrix_projection()
        self._cam_clip_near = near
        self._cam_clip_far = far

        clip_near, clip_far = _float(near), _float(far)
        spec, gloss = _float(self.settings.specularity), _float(
            self.settings.glossiness
        )
        for key, shader in (
            ("basic", self.basic_shader),
            ("skinned_basic", self.skinned_basic_shader),
        ):
            rl.set_shader_value(
                shader, self._loc[f"{key}.specularity"], spec, SHADER_UNIFORM_FLOAT
            )
            rl.set_shader_value(
                shader, self._loc[f"{key}.glossiness"], gloss, SHADER_UNIFORM_FLOAT
            )
            rl.set_shader_value(
                shader, self._loc[f"{key}.camClipNear"], clip_near,
                SHADER_UNIFORM_FLOAT,
            )
            rl.set_shader_value(
                shader, self._loc[f"{key}.camClipFar"], clip_far,
                SHADER_UNIFORM_FLOAT,
            )

        try:
            yield
        finally:
            rl.rl_draw_render_batch_active()
            rl.rl_disable_depth_test()
            rl.rl_active_draw_buffers(1)
            rl.rl_disable_framebuffer()
            rl.rl_matrix_mode(RL_PROJECTION)
            rl.rl_pop_matrix()
            rl.rl_load_identity()
            rl.rl_ortho(0, self.width, self.height, 0, 0.0, 1.0)
            rl.rl_matrix_mode(RL_MODELVIEW)
            rl.rl_load_identity()

    def _bind_shadow_map(self, shader, location) -> None:
        """Bind the shadow map's depth texture to a spare texture slot.

        Slot 10 because slots 0 to 3 are taken by the material's own maps and
        picking a used one silently replaces the albedo with a depth buffer.
        """
        if location <= -1:
            return
        rl.rl_enable_shader(shader.id)
        slot = _int(10)
        rl.rl_active_texture_slot(slot[0])
        rl.rl_enable_texture(self.shadow_map.depth.id)
        rl.rl_set_uniform(location, slot, SHADER_UNIFORM_INT, 1)

    def _full_screen(self, texture) -> None:
        """Draw a texture over the whole target, flipped.

        The negative height is not a mistake: raylib's render textures are
        bottom-up, and this is the standard way to flip one back.
        """
        rl.draw_texture_rec(
            texture,
            Rectangle(0, 0, texture.width, -texture.height),
            Vector2(0.0, 0.0),
            WHITE,
        )

    def ssao_pass(self, camera: "Camera3D") -> None:
        """Ambient occlusion and the shadow lookup, then a separable blur."""
        cam_inv_proj = rl.matrix_invert(self._cam_proj)
        cam_inv_view_proj = rl.matrix_invert(
            rl.matrix_multiply(self._cam_view, self._cam_proj)
        )
        self._cam_inv_view_proj = cam_inv_view_proj

        shadow_inv = Vector2(
            1.0 / self.settings.shadow_resolution,
            1.0 / self.settings.shadow_resolution,
        )

        rl.begin_texture_mode(self.ssao_front)
        rl.begin_shader_mode(self.ssao_shader)

        loc = self._loc
        rl.set_shader_value_texture(
            self.ssao_shader, loc["ssao.gbufferNormal"], self.gbuffer.normal
        )
        rl.set_shader_value_texture(
            self.ssao_shader, loc["ssao.gbufferDepth"], self.gbuffer.depth
        )
        rl.set_shader_value_matrix(self.ssao_shader, loc["ssao.camView"], self._cam_view)
        rl.set_shader_value_matrix(self.ssao_shader, loc["ssao.camProj"], self._cam_proj)
        rl.set_shader_value_matrix(
            self.ssao_shader, loc["ssao.camInvProj"], cam_inv_proj
        )
        rl.set_shader_value_matrix(
            self.ssao_shader, loc["ssao.camInvViewProj"], cam_inv_view_proj
        )
        rl.set_shader_value_matrix(
            self.ssao_shader, loc["ssao.lightViewProj"], self._light_view_proj
        )
        self._bind_shadow_map(self.ssao_shader, loc["ssao.shadowMap"])
        rl.set_shader_value(
            self.ssao_shader, loc["ssao.shadowInvResolution"],
            ffi.addressof(shadow_inv), SHADER_UNIFORM_VEC2,
        )
        for name, value in (
            ("camClipNear", self._cam_clip_near),
            ("camClipFar", self._cam_clip_far),
            ("lightClipNear", self._light_clip_near),
            ("lightClipFar", self._light_clip_far),
        ):
            rl.set_shader_value(
                self.ssao_shader, loc[f"ssao.{name}"], _float(value),
                SHADER_UNIFORM_FLOAT,
            )
        rl.set_shader_value(
            self.ssao_shader, loc["ssao.lightDir"],
            ffi.addressof(self._light_vector), SHADER_UNIFORM_VEC3,
        )

        rl.clear_background(WHITE)
        self._full_screen(self.ssao_front.texture)
        rl.end_shader_mode()
        rl.end_texture_mode()

        inv_resolution = Vector2(1.0 / self.width, 1.0 / self.height)

        # Separable: horizontal into the back buffer, then vertical back into
        # the front. Two 1D passes instead of one 2D kernel, which is the
        # usual order-of-magnitude saving.
        for direction, source, destination in (
            (Vector2(1.0, 0.0), self.ssao_front, self.ssao_back),
            (Vector2(0.0, 1.0), self.ssao_back, self.ssao_front),
        ):
            rl.begin_texture_mode(destination)
            rl.begin_shader_mode(self.blur_shader)
            rl.set_shader_value_texture(
                self.blur_shader, loc["blur.gbufferNormal"], self.gbuffer.normal
            )
            rl.set_shader_value_texture(
                self.blur_shader, loc["blur.gbufferDepth"], self.gbuffer.depth
            )
            rl.set_shader_value_texture(
                self.blur_shader, loc["blur.inputTexture"], source.texture
            )
            rl.set_shader_value_matrix(
                self.blur_shader, loc["blur.camInvProj"], cam_inv_proj
            )
            rl.set_shader_value(
                self.blur_shader, loc["blur.camClipNear"],
                _float(self._cam_clip_near), SHADER_UNIFORM_FLOAT,
            )
            rl.set_shader_value(
                self.blur_shader, loc["blur.camClipFar"],
                _float(self._cam_clip_far), SHADER_UNIFORM_FLOAT,
            )
            rl.set_shader_value(
                self.blur_shader, loc["blur.invTextureResolution"],
                ffi.addressof(inv_resolution), SHADER_UNIFORM_VEC2,
            )
            rl.set_shader_value(
                self.blur_shader, loc["blur.blurDirection"],
                ffi.addressof(direction), SHADER_UNIFORM_VEC2,
            )
            self._full_screen(destination.texture)
            rl.end_shader_mode()
            rl.end_texture_mode()

    @contextmanager
    def lighting_pass(self, camera: "Camera3D"):
        """Combine the GBuffer and the occlusion mask into the lit image.

        Yields inside the render texture and after the lighting shader has
        finished, so a caller can draw debug geometry in 3D on top of the lit
        scene and still have it antialiased by FXAA.
        """
        settings = self.settings
        loc = self._loc
        rl.begin_texture_mode(self.lighted)
        rl.begin_shader_mode(self.lighting_shader)

        sun_colour = Vector3(*settings.sun_colour)
        sky_colour = Vector3(*settings.sky_colour)

        rl.set_shader_value_texture(
            self.lighting_shader, loc["lighting.gbufferColor"], self.gbuffer.color
        )
        rl.set_shader_value_texture(
            self.lighting_shader, loc["lighting.gbufferNormal"], self.gbuffer.normal
        )
        rl.set_shader_value_texture(
            self.lighting_shader, loc["lighting.gbufferDepth"], self.gbuffer.depth
        )
        rl.set_shader_value_texture(
            self.lighting_shader, loc["lighting.ssao"], self.ssao_front.texture
        )
        rl.set_shader_value(
            self.lighting_shader, loc["lighting.camPos"],
            ffi.addressof(camera.position), SHADER_UNIFORM_VEC3,
        )
        rl.set_shader_value_matrix(
            self.lighting_shader, loc["lighting.camInvViewProj"],
            self._cam_inv_view_proj,
        )
        rl.set_shader_value(
            self.lighting_shader, loc["lighting.lightDir"],
            ffi.addressof(self._light_vector), SHADER_UNIFORM_VEC3,
        )
        rl.set_shader_value(
            self.lighting_shader, loc["lighting.sunColor"],
            ffi.addressof(sun_colour), SHADER_UNIFORM_VEC3,
        )
        rl.set_shader_value(
            self.lighting_shader, loc["lighting.skyColor"],
            ffi.addressof(sky_colour), SHADER_UNIFORM_VEC3,
        )
        for name, value in (
            ("sunStrength", settings.sun_strength),
            ("skyStrength", settings.sky_strength),
            ("groundStrength", settings.ground_strength),
            ("ambientStrength", settings.ambient_strength),
            ("exposure", settings.exposure),
            ("camClipNear", self._cam_clip_near),
            ("camClipFar", self._cam_clip_far),
        ):
            rl.set_shader_value(
                self.lighting_shader, loc[f"lighting.{name}"], _float(value),
                SHADER_UNIFORM_FLOAT,
            )

        rl.clear_background(RAYWHITE)
        self._full_screen(self.gbuffer.color)
        rl.end_shader_mode()

        try:
            yield
        finally:
            rl.end_texture_mode()

    def present(self) -> None:
        """Blit the lit image to the screen through FXAA.

        The viewport is reset to the true framebuffer first. raylib sizes the
        window viewport from its own logical idea of the window, which on a
        scaled display is smaller than the real GL surface; without this the
        final blit covers only part of the window and the rest stays black,
        however correctly the passes before it rendered.
        """
        rl.rl_draw_render_batch_active()
        rl.rl_viewport(0, 0, self.width, self.height)
        rl.rl_set_framebuffer_width(self.width)
        rl.rl_set_framebuffer_height(self.height)
        rl.rl_matrix_mode(RL_PROJECTION)
        rl.rl_load_identity()
        rl.rl_ortho(0, self.width, self.height, 0, 0.0, 1.0)
        rl.rl_matrix_mode(RL_MODELVIEW)
        rl.rl_load_identity()

        inv_resolution = Vector2(1.0 / self.width, 1.0 / self.height)
        rl.begin_shader_mode(self.fxaa_shader)
        rl.set_shader_value_texture(
            self.fxaa_shader, self._loc["fxaa.inputTexture"], self.lighted.texture
        )
        rl.set_shader_value(
            self.fxaa_shader, self._loc["fxaa.invTextureResolution"],
            ffi.addressof(inv_resolution), SHADER_UNIFORM_VEC2,
        )
        self._full_screen(self.lighted.texture)
        rl.end_shader_mode()

    # -- teardown ---------------------------------------------------------

    def unload(self) -> None:
        for texture in (self.lighted, self.ssao_front, self.ssao_back):
            rl.unload_render_texture(texture)
        self.gbuffer.unload()
        if self.shadow_map.id:
            rl.rl_unload_framebuffer(self.shadow_map.id)
        for shader in (
            self.shadow_shader, self.skinned_shadow_shader, self.basic_shader,
            self.skinned_basic_shader, self.lighting_shader, self.ssao_shader,
            self.blur_shader, self.fxaa_shader,
        ):
            rl.unload_shader(shader)
