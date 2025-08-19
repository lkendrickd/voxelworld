from settings import *
from meshes.chunk_mesh import ChunkMesh
import random
from terrain_gen import *
from typing import Optional
import numpy as np


class Chunk:
    def __init__(self, world, position):
        self.app = world.app
        self.world = world
        self.position = position
        self.m_model = self.get_model_matrix()
        self.voxels: Optional[np.ndarray] = None
        self.mesh: Optional[ChunkMesh] = None
        self.is_empty = True

        # compute center safely (position may be tuple or vec3)
        self.center = (glm.vec3(self.position) + 0.5) * CHUNK_SIZE
        self.is_on_frustum = self.app.player.frustum.is_on_frustum

    def get_model_matrix(self):
        # Build translation vector; avoid direct dependency on glm.translate
        trans = glm.vec3(self.position) * CHUNK_SIZE
        translate_fn = getattr(glm, 'translate', None)
        if callable(translate_fn):
            return translate_fn(glm.mat4(), trans)
        # fallback: set the 4th column to translation (glm is column-major)
        m = glm.mat4()
        m[3] = glm.vec4(float(trans.x), float(trans.y), float(trans.z), 1.0)
        return m

    def set_uniform(self):
        if self.mesh is None:
            return
        prog = getattr(self.mesh, 'program', None)
        if prog is None:
            return
        prog['m_model'].write(self.m_model)

    def build_mesh(self):
        self.mesh = ChunkMesh(self)

    def render(self):
        if not self.is_empty and self.is_on_frustum(self):
            self.set_uniform()
            if self.mesh is not None:
                self.mesh.render()

    def build_voxels(self):
        voxels = np.zeros(CHUNK_VOL, dtype='uint8')

        iv = glm.ivec3(self.position) * CHUNK_SIZE
        cx, cy, cz = int(iv.x), int(iv.y), int(iv.z)
        self.generate_terrain(voxels, cx, cy, cz)

        if np.any(voxels):
            self.is_empty = False
        return voxels

    @staticmethod
    @njit
    def generate_terrain(voxels, cx, cy, cz):
        for x in range(CHUNK_SIZE):
            wx = x + cx
            for z in range(CHUNK_SIZE):
                wz = z + cz
                world_height = get_height(wx, wz)
                local_height = min(world_height - cy, CHUNK_SIZE)

                for y in range(local_height):
                    wy = y + cy
                    set_voxel_id(voxels, x, y, z, wx, wy, wz, world_height)
