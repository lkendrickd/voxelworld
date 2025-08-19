from settings import *
import moderngl as mgl
import pygame as pg
import sys
import time
from shader_program import ShaderProgram
from scene import Scene
from player import Player
from textures import Textures
from typing import Tuple, cast


class VoxelEngine:
    def __init__(self):
        pg.init()

        # set GL attributes defensively (some pygame builds may not expose all constants)
        gl_set = getattr(pg.display, 'gl_set_attribute', None)
        if callable(gl_set):
            major_attr = getattr(pg, 'GL_CONTEXT_MAJOR_VERSION', None)
            minor_attr = getattr(pg, 'GL_CONTEXT_MINOR_VERSION', None)
            profile_mask_attr = getattr(pg, 'GL_CONTEXT_PROFILE_MASK', None)
            profile_core = getattr(pg, 'GL_CONTEXT_PROFILE_CORE', None)
            depth_attr = getattr(pg, 'GL_DEPTH_SIZE', None)
            ms_attr = getattr(pg, 'GL_MULTISAMPLESAMPLES', None)

            if major_attr is not None:
                gl_set(major_attr, MAJOR_VER)
            if minor_attr is not None:
                gl_set(minor_attr, MINOR_VER)
            if profile_mask_attr is not None and profile_core is not None:
                gl_set(profile_mask_attr, profile_core)
            if depth_attr is not None:
                gl_set(depth_attr, DEPTH_SIZE)
            if ms_attr is not None:
                gl_set(ms_attr, NUM_SAMPLES)

        # convert WIN_RES (glm.vec2) to plain tuple for pygame
        try:
            screen_res = (int(WIN_RES.x), int(WIN_RES.y))
        except Exception:
            screen_res = tuple(WIN_RES)

        flags = 0
        flags |= getattr(pg, 'OPENGL', 0)
        flags |= getattr(pg, 'DOUBLEBUF', 0)
        pg.display.set_mode(screen_res, flags=flags)

        # create moderngl context
        self.ctx = mgl.create_context()

        # enable GL flags defensively
        try:
            enable_fn = getattr(self.ctx, 'enable', None)
            if callable(enable_fn):
                flags = 0
                flags |= getattr(mgl, 'DEPTH_TEST', 0)
                flags |= getattr(mgl, 'CULL_FACE', 0)
                flags |= getattr(mgl, 'BLEND', 0)
                enable_fn(flags=flags)
        except Exception:
            # fallback: try calling without keyword
            try:
                if callable(getattr(self.ctx, 'enable', None)):
                    self.ctx.enable(flags)
            except Exception:
                pass

        # set gc mode if supported
        if hasattr(self.ctx, 'gc_mode'):
            try:
                self.ctx.gc_mode = 'auto'
            except Exception:
                pass

        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0

        pg.event.set_grab(True)
        pg.mouse.set_visible(False)

        self.is_running = True
        self.on_init()

    def on_init(self):
        import time
        print("Starting game initialization...")
        start_time = time.time()
        
        print("Loading textures...")
        tex_start = time.time()
        self.textures = Textures(self)
        tex_end = time.time()
        print(f"Textures loaded in {tex_end - tex_start:.2f} seconds")
        
        print("Initializing player...")
        player_start = time.time()
        self.player = Player(self)
        player_end = time.time()
        print(f"Player initialized in {player_end - player_start:.2f} seconds")
        
        print("Compiling shaders...")
        shader_start = time.time()
        self.shader_program = ShaderProgram(self)
        shader_end = time.time()
        print(f"Shaders compiled in {shader_end - shader_start:.2f} seconds")
        
        print("Building scene...")
        scene_start = time.time()
        self.scene = Scene(self)
        scene_end = time.time()
        print(f"Scene built in {scene_end - scene_start:.2f} seconds")
        
        end_time = time.time()
        print(f"Total initialization time: {end_time - start_time:.2f} seconds")
        print("Game ready!")

    def update(self):
        if self.player is not None:
            self.player.update()
        if self.shader_program is not None:
            self.shader_program.update()
        if self.scene is not None:
            self.scene.update()

        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001
        pg.display.set_caption(f'{self.clock.get_fps() :.0f}')

    def render(self):
        # convert BG_COLOR (glm.vec3) to RGBA tuple for moderngl
        try:
            clear_color = (float(BG_COLOR.x), float(BG_COLOR.y), float(BG_COLOR.z), 1.0)
        except Exception:
            # ensure we have 4 components, append alpha if needed
            tmp = tuple(BG_COLOR)
            if len(tmp) == 3:
                clear_color = (tmp[0], tmp[1], tmp[2], 1.0)
            else:
                # fallback: take first four elements and pad if shorter
                tmp2 = list(tmp)
                while len(tmp2) < 4:
                    tmp2.append(1.0)
                clear_color = tuple(tmp2[:4])
        # cast to a fixed-size 4-tuple for the type checker
        clear_color = cast(Tuple[float, float, float, float], clear_color)
        self.ctx.clear(color=clear_color)
        if self.scene is not None:
            self.scene.render()
        pg.display.flip()

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.is_running = False
            if self.player is not None:
                self.player.handle_event(event=event)

    def run(self):
        while self.is_running:
            try:
                self.handle_events()
                self.update()
                self.render()
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                # Try to continue or exit gracefully
                self.is_running = False
        pg.quit()
        sys.exit()


if __name__ == '__main__':
    app = VoxelEngine()
    app.run()
