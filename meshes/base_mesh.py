import numpy as np
from typing import Optional, Tuple


class BaseMesh:
    def __init__(self):
        # OpenGL context
        self.ctx = None
        # shader program
        self.program = None
        # vertex buffer data type format: "3f 3f"
        self.vbo_format = None
        # attribute names according to the format: ("in_position", "in_color")
        self.attrs: Optional[Tuple[str, ...]] = None
        # vertex array object
        self.vao = None

    def get_vertex_data(self) -> np.ndarray:
        """Return vertex data as a numpy ndarray. Subclasses must implement."""
        raise NotImplementedError("get_vertex_data must be implemented by subclasses")

    def get_vao(self):
        # validate prerequisites
        if self.ctx is None:
            raise RuntimeError("OpenGL context (ctx) is not set")
        if self.program is None:
            raise RuntimeError("Shader program is not set")
        if self.vbo_format is None:
            raise RuntimeError("VBO format (vbo_format) is not set")
        if not self.attrs:
            raise RuntimeError("Attribute names (attrs) are not set")

        vertex_data = self.get_vertex_data()
        # ensure vertex_data is a contiguous bytes-like object
        try:
            data = vertex_data.tobytes()
        except Exception:
            data = vertex_data

        vbo = self.ctx.buffer(data)
        vao = self.ctx.vertex_array(
            self.program, [(vbo, self.vbo_format, *self.attrs)], skip_errors=True
        )
        self.vao = vao
        return vao

    def render(self):
        if self.vao is None:
            self.get_vao()
        self.vao.render()
