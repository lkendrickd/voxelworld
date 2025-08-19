from settings import *
import moderngl as mgl
import pygame as pg
import sys
import time
from shader_program import ShaderProgram
from scene import Scene
from player import Player
from textures import Textures
from debug_overlay import DebugInfo
from typing import Tuple, cast, Optional


def setup_pygame_and_moderngl() -> Optional[mgl.Context]:
    """Helper function to set up Pygame and ModernGL with defensive checks."""
    pg.init()

    # Set GL attributes
    gl_set = getattr(pg.display, 'gl_set_attribute', None)
    if callable(gl_set):
        attrs = {
            'GL_CONTEXT_MAJOR_VERSION': MAJOR_VER,
            'GL_CONTEXT_MINOR_VERSION': MINOR_VER,
            'GL_CONTEXT_PROFILE_MASK': getattr(pg, 'GL_CONTEXT_PROFILE_CORE', 0),
            'GL_DEPTH_SIZE': DEPTH_SIZE,
            'GL_MULTISAMPLESAMPLES': NUM_SAMPLES
        }
        for attr_name, value in attrs.items():
            pg_attr = getattr(pg, attr_name, None)
            if pg_attr is not None:
                gl_set(pg_attr, value)

    # Set screen resolution
    try:
        screen_res = (int(WIN_RES.x), int(WIN_RES.y))
    except (AttributeError, TypeError):
        screen_res = tuple(WIN_RES)

    # Set display mode flags
    flags = getattr(pg, 'OPENGL', 0) | getattr(pg, 'DOUBLEBUF', 0)
    pg.display.set_mode(screen_res, flags=flags)

    # Create ModernGL context
    try:
        ctx = mgl.create_context()
    except Exception as e:
        print(f"Error creating ModernGL context: {e}")
        return None

    # Enable flags
    try:
        ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
    except Exception as e:
        print(f"Warning: Could not enable some GL flags: {e}")

    # Set GC mode
    if hasattr(ctx, 'gc_mode'):
        ctx.gc_mode = 'auto'

    return ctx


class VoxelEngine:
    def __init__(self):
        self.ctx = setup_pygame_and_moderngl()
        if not self.ctx:
            print("Failed to initialize rendering context. Exiting.")
            sys.exit()

        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0

        pg.event.set_grab(True)
        pg.mouse.set_visible(False)

        self.is_running = True
        self.on_init()

    def on_init(self):
        print("Starting game initialization...")
        start_time = time.time()
        
        self.textures = Textures(self)
        self.player = Player(self)
        self.shader_program = ShaderProgram(self)
        self.scene = Scene(self)
        self.debug_info = DebugInfo(self)
        
        end_time = time.time()
        print(f"Total initialization time: {end_time - start_time:.2f} seconds")
        print("Game ready!")

    def update(self):
        self.player.update()
        self.shader_program.update()
        self.scene.update()

        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001
        pg.display.set_caption(f'{self.clock.get_fps() :.0f}')

    def render(self):
        # Simplified background color handling
        try:
            # Assuming BG_COLOR is a glm.vec3 or similar
            clear_color = (float(BG_COLOR.x), float(BG_COLOR.y), float(BG_COLOR.z), 1.0)
        except AttributeError:
            # Fallback for tuples
            clear_color = (float(BG_COLOR[0]), float(BG_COLOR[1]), float(BG_COLOR[2]), 1.0)

        # ctx is guaranteed in __init__; add assert to satisfy checkers
        assert self.ctx is not None
        self.ctx.clear(color=cast(Tuple[float, float, float, float], clear_color))
        self.scene.render()
        self.debug_info.render()
        pg.display.flip()

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.is_running = False
            self.player.handle_event(event=event)

    def destroy(self):
        self.debug_info.release()
        self.shader_program.destroy()
        self.textures.destroy()

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
        self.destroy()
        pg.quit()
        sys.exit()


if __name__ == '__main__':
    app = VoxelEngine()
    app.run()
