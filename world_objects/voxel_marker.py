from settings import *
from meshes.cube_mesh import CubeMesh
from typing import Any, cast


class VoxelMarker:
    def __init__(self, voxel_handler):
        self.app = voxel_handler.app
        self.handler = voxel_handler
        self.position = glm.vec3(0)
        self.m_model = self.get_model_matrix()
        self.mesh = CubeMesh(self.app)

    def update(self):
        if self.handler.voxel_id:
            if self.handler.interaction_mode:
                # ensure vec3 arithmetic
                self.position = glm.vec3(self.handler.voxel_world_pos) + glm.vec3(self.handler.voxel_normal)
            else:
                self.position = glm.vec3(self.handler.voxel_world_pos)

    def set_uniform(self):
        # program expects integers/floats; set directly
        self.mesh.program['mode_id'] = int(self.handler.interaction_mode)
        self.mesh.program['m_model'].write(self.get_model_matrix())

    def get_model_matrix(self):
        trans = glm.vec3(self.position)
        translate_fn = getattr(cast(Any, glm), 'translate', None)
        if callable(translate_fn):
            return translate_fn(glm.mat4(), trans)
        m = glm.mat4()
        m[3] = glm.vec4(float(trans.x), float(trans.y), float(trans.z), 1.0)
        return m

    def render(self):
        if self.handler.voxel_id:
            self.set_uniform()
            self.mesh.render()
