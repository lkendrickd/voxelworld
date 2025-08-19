from __future__ import annotations
from settings import *
from meshes.debris_cube_mesh import DebrisCubeMesh
import glm
import random
from dataclasses import dataclass
from typing import List


@dataclass
class DebrisPiece:
    pos: glm.vec3
    vel: glm.vec3
    scale: float
    ttl: float
    age: float = 0.0
    mode_id: int = 0  # reuse voxel_marker color index
    voxel_id: int = 1


class DebrisSystem:
    """Simple CPU-simulated debris made of tiny cubes using the voxel_marker shader.

    Minimal and safe: no new shaders needed, uses existing CubeMesh.
    """

    def __init__(self, app):
        self.app = app
        self.mesh = DebrisCubeMesh(app)
        self.pieces = []
        # Tunables
        self.gravity = glm.vec3(0, -9.8, 0)
        self.max_pieces = 3000
        self.bounce = 0.35  # energy kept on ground bounce
        self.friction = 0.8  # horizontal slow on bounce

    def spawn_at(self, voxel_world_pos, voxel_id: int = 1, count: int = 28):
        try:
            base = glm.vec3(voxel_world_pos) + 0.5
        except Exception:
            x, y, z = map(float, voxel_world_pos)
            base = glm.vec3(x + 0.5, y + 0.5, z + 0.5)

        for _ in range(count):
            # Small random offset inside the voxel
            off = glm.vec3(
                random.uniform(-0.3, 0.3),
                random.uniform(-0.3, 0.3),
                random.uniform(-0.3, 0.3),
            )
            pos = base + off

            # Velocity: slight upward burst with random spread
            vel = glm.vec3(
                random.uniform(-2.0, 2.0),
                random.uniform(2.0, 5.5),
                random.uniform(-2.0, 2.0),
            )

            # even smaller pieces per request
            scale = random.uniform(0.04, 0.09)
            ttl = random.uniform(0.8, 1.6)
            # Map voxel id to color index (0/1) to reuse shader tint
            mode_id = 0 if (voxel_id % 2 == 0) else 1
            self.pieces.append(DebrisPiece(pos=pos, vel=vel, scale=scale, ttl=ttl, mode_id=mode_id, voxel_id=int(voxel_id)))

        # Trim to max
        if len(self.pieces) > self.max_pieces:
            self.pieces = self.pieces[-self.max_pieces :]

    def update(self):
        if not self.pieces:
            return
        dt = float(self.app.delta_time) * 0.001 if getattr(self.app, 'delta_time', 0) else 0.016

        new_pieces: List[DebrisPiece] = []
        for p in self.pieces:
            # Integrate simple physics
            p.vel += self.gravity * dt
            p.pos += p.vel * dt
            p.age += dt

            # Ground plane bounce at y=0
            if p.pos.y < 0.0:
                p.pos.y = 0.0
                if p.vel.y < 0.0:
                    p.vel.y = -p.vel.y * self.bounce
                    p.vel.x *= self.friction
                    p.vel.z *= self.friction

            if p.age < p.ttl:
                new_pieces.append(p)

        self.pieces = new_pieces

    def _model_matrix(self, pos: glm.vec3, scale: float) -> glm.mat4:
        m = glm.mat4(1.0)
        # scale
        m[0][0] = float(scale)
        m[1][1] = float(scale)
        m[2][2] = float(scale)
        # translate
        m[3] = glm.vec4(float(pos.x), float(pos.y), float(pos.z), 1.0)
        return m

    def render(self):
        if not self.pieces:
            return
        prog = self.mesh.program
        # Set uniforms that change per frame
        prog['m_proj'].write(self.app.player.m_proj)
        prog['m_view'].write(self.app.player.m_view)

        # Draw each piece (simple and safe; can be optimized later)
        for p in self.pieces:
            # Select proper texture layer to match the parent voxel id
            try:
                prog['u_voxel_id'].value = int(p.voxel_id)
            except Exception:
                pass
            prog['m_model'].write(self._model_matrix(p.pos, p.scale))
            self.mesh.render()

    def clear(self):
        self.pieces.clear()
