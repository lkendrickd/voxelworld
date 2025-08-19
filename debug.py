import pygame as pg
import moderngl as mgl
import glm
import numpy as np
from settings import *


class DebugInfo:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.font = pg.font.SysFont('Arial', 24, bold=True)

        # Use a dedicated texture unit for the overlay to avoid interfering
        # with world shaders that sample unit 0/1/2.
        self._overlay_tex_unit = 3

        # Shader for rendering 2D textures
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330 core
                uniform mat4 m_proj;
                uniform mat4 m_model;
                in vec2 in_vert;
                in vec2 in_uv;
                out vec2 uv;
                void main() {
                    gl_Position = m_proj * m_model * vec4(in_vert, 0.0, 1.0);
                    uv = in_uv;
                }
            ''',
            fragment_shader='''
                #version 330 core
                uniform sampler2D u_texture;
                in vec2 uv;
                out vec4 f_color;
                void main() {
                    vec4 color = texture(u_texture, uv);
                    if (color.a < 0.1) discard;
                    f_color = color;
                }
            '''
        )
        # Bind the overlay sampler to its dedicated texture unit
        self.program['u_texture'].value = self._overlay_tex_unit

        # Quad vertices for a unit square
        vertices = np.array([
            # x, y, u, v
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 0.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '2f 2f', 'in_vert', 'in_uv')],
        )
        self.texture = None

    def get_debug_info(self):
        # FPS
        fps = f"FPS: {self.app.clock.get_fps():.0f}"

        # Player position
        pos = self.app.player.position
        pos_text = f"Pos: {pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}"

        # Chunk info
        world = self.app.scene.world
        meshing_progress = "Meshing: Complete"
        if not world.mesh_building_complete:
            progress = (world.meshes_built / world.total_chunks) * 100
            meshing_progress = f"Meshing: {progress:.1f}% ({world.meshes_built}/{world.total_chunks})"

        visible_chunks = f"Visible Chunks: {getattr(world, 'visible_chunks_count', 0)}"

        return [fps, pos_text, meshing_progress, visible_chunks]

    def render(self):
        lines = self.get_debug_info()

        # Create surfaces from text lines
        surfaces = [self.font.render(line, True, (255, 255, 255)) for line in lines]

        # Determine texture size
        max_width = max(s.get_width() for s in surfaces) if surfaces else 0
        total_height = sum(s.get_height() for s in surfaces)

        if max_width == 0 or total_height == 0:
            return  # Nothing to render

        # Create a single surface for the texture
        texture_surf = pg.Surface((max_width, total_height), pg.SRCALPHA)

        current_y = 0
        for surf in surfaces:
            texture_surf.blit(surf, (0, current_y))
            current_y += surf.get_height()

        # Create moderngl texture
        texture_data = pg.image.tostring(texture_surf, 'RGBA', True)
        if self.texture:
            self.texture.release()
        self.texture = self.ctx.texture(texture_surf.get_size(), 4, texture_data)
        # Bind to the dedicated overlay unit to avoid clobbering unit 0
        self.texture.use(location=self._overlay_tex_unit)

        # Projection matrix for 2D rendering (orthographic)
        # Use 6-arg ortho (left, right, bottom, top, zNear, zFar)
        try:
            w = float(WIN_RES.x)
            h = float(WIN_RES.y)
        except AttributeError:
            w = float(WIN_RES[0])
            h = float(WIN_RES[1])
        proj = glm.ortho(0.0, w, 0.0, h, -1.0, 1.0)

        # Model matrix to position and scale the quad
        width, height = self.texture.size
        # Build model matrix without relying on glm.translate/scale (for compatibility)
        model = glm.mat4(1.0)
        # scale
        model[0][0] = float(width)
        model[1][1] = float(height)
        # translation (top-left with a small margin)
        try:
            tx = 10.0
            ty = float(WIN_RES.y) - float(height) - 10.0
        except AttributeError:
            tx = 10.0
            ty = float(WIN_RES[1]) - float(height) - 10.0
        model[3] = glm.vec4(tx, ty, 0.0, 1.0)

        # Pass matrices to the shader
        self.program['m_proj'].write(proj)
        self.program['m_model'].write(model)

        # Render the quad on top of the scene: disable depth testing and
        # depth writes to ensure the overlay doesn't interact with 3D depth.
        # Some backends don't support reading blend_func; just set it as needed
        prev_depth_mask = getattr(self.ctx, 'depth_mask', True)
        try:
            self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        except NotImplementedError:
            pass
        self.ctx.disable(mgl.DEPTH_TEST)
        # Avoid writing to the depth buffer while drawing the overlay
        try:
            self.ctx.depth_mask = False
        except Exception:
            pass
        try:
            self.vao.render(mgl.TRIANGLE_STRIP)
        finally:
            # Restore GL state for subsequent 3D renders
            self.ctx.enable(mgl.DEPTH_TEST)
            try:
                self.ctx.depth_mask = prev_depth_mask
            except Exception:
                pass
            # Leave blend function as set; do not attempt to read/restore

    def release(self):
        self.vbo.release()
        self.program.release()
        if self.texture:
            self.texture.release()
        self.vao.release()
