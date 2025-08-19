from settings import *
from meshes.chunk_mesh_builder import get_chunk_index


class VoxelHandler:
    def __init__(self, world):
        self.app = world.app
        self.chunks = world.chunks

        # ray casting result
        self.chunk = None
        self.voxel_id = None
        self.voxel_index = None
        self.voxel_local_pos = None
        self.voxel_world_pos = None
        self.voxel_normal = None

        self.interaction_mode = 0  # 0: remove voxel   1: add voxel
        self.new_voxel_id = DIRT

    def add_voxel(self):
        if self.voxel_id:
            # ensure we have a valid world position and normal before adding
            if self.voxel_world_pos is None or self.voxel_normal is None:
                return

            # compute target position safely (support glm vectors or tuples/lists)
            if hasattr(self.voxel_world_pos, 'x') and hasattr(self.voxel_normal, 'x'):
                target = glm.ivec3(
                    int(self.voxel_world_pos.x + self.voxel_normal.x),
                    int(self.voxel_world_pos.y + self.voxel_normal.y),
                    int(self.voxel_world_pos.z + self.voxel_normal.z),
                )
            else:
                target = (
                    int(self.voxel_world_pos[0] + self.voxel_normal[0]),
                    int(self.voxel_world_pos[1] + self.voxel_normal[1]),
                    int(self.voxel_world_pos[2] + self.voxel_normal[2]),
                )

            # check voxel id along normal
            result = self.get_voxel_id(target)

            # is the new place empty?
            if not result[0]:
                _, voxel_index, _, chunk = result
                if chunk is not None:
                    chunk.voxels[voxel_index] = self.new_voxel_id
                    chunk.build_mesh()

                    # was it an empty chunk
                    if chunk.is_empty:
                        chunk.is_empty = False

    def rebuild_adj_chunk(self, adj_voxel_pos):
        index = get_chunk_index(adj_voxel_pos)
        if index != -1:
            chunk = self.chunks[index]
            if chunk is not None:
                chunk.build_mesh()

    def rebuild_adjacent_chunks(self):
        # ensure we have a valid voxel hit before attempting to rebuild neighbors
        if self.voxel_local_pos is None or self.voxel_world_pos is None:
            return

        lx, ly, lz = map(int, self.voxel_local_pos)
        wx, wy, wz = map(int, self.voxel_world_pos)

        if lx == 0:
            self.rebuild_adj_chunk((wx - 1, wy, wz))
        elif lx == CHUNK_SIZE - 1:
            self.rebuild_adj_chunk((wx + 1, wy, wz))

        if ly == 0:
            self.rebuild_adj_chunk((wx, wy - 1, wz))
        elif ly == CHUNK_SIZE - 1:
            self.rebuild_adj_chunk((wx, wy + 1, wz))

        if lz == 0:
            self.rebuild_adj_chunk((wx, wy, wz - 1))
        elif lz == CHUNK_SIZE - 1:
            self.rebuild_adj_chunk((wx, wy, wz + 1))

    def remove_voxel(self):
        if self.voxel_id and self.chunk is not None:
            removed_id = int(self.voxel_id)
            removed_pos = self.voxel_world_pos
            self.chunk.voxels[self.voxel_index] = 0
            self.chunk.build_mesh()
            self.rebuild_adjacent_chunks()
            # Spawn debris if scene has a debris system
            try:
                self.app.scene.debris.spawn_at(removed_pos, removed_id, count=14)
            except Exception:
                pass

    def set_voxel(self):
        if self.interaction_mode:
            self.add_voxel()
        else:
            self.remove_voxel()

    def switch_mode(self):
        self.interaction_mode = not self.interaction_mode

    def update(self):
        self.ray_cast()

    def ray_cast(self):
        # start point
        x1, y1, z1 = self.app.player.position
        # end point
        x2, y2, z2 = self.app.player.position + self.app.player.forward * MAX_RAY_DIST

        current_voxel_pos = glm.ivec3(x1, y1, z1)
        self.voxel_id = 0
        self.voxel_normal = glm.ivec3(0)
        step_dir = -1

        dx = glm.sign(x2 - x1)
        delta_x = min(dx / (x2 - x1), 10000000.0) if dx != 0 else 10000000.0
        max_x = delta_x * (1.0 - glm.fract(x1)) if dx > 0 else delta_x * glm.fract(x1)

        dy = glm.sign(y2 - y1)
        delta_y = min(dy / (y2 - y1), 10000000.0) if dy != 0 else 10000000.0
        max_y = delta_y * (1.0 - glm.fract(y1)) if dy > 0 else delta_y * glm.fract(y1)

        dz = glm.sign(z2 - z1)
        delta_z = min(dz / (z2 - z1), 10000000.0) if dz != 0 else 10000000.0
        max_z = delta_z * (1.0 - glm.fract(z1)) if dz > 0 else delta_z * glm.fract(z1)

        while not (max_x > 1.0 and max_y > 1.0 and max_z > 1.0):

            result = self.get_voxel_id(voxel_world_pos=current_voxel_pos)
            if result[0]:
                self.voxel_id, self.voxel_index, self.voxel_local_pos, self.chunk = result
                self.voxel_world_pos = current_voxel_pos

                if step_dir == 0:
                    self.voxel_normal.x = -dx
                elif step_dir == 1:
                    self.voxel_normal.y = -dy
                else:
                    self.voxel_normal.z = -dz
                return True

            if max_x < max_y:
                if max_x < max_z:
                    current_voxel_pos.x += dx
                    max_x += delta_x
                    step_dir = 0
                else:
                    current_voxel_pos.z += dz
                    max_z += delta_z
                    step_dir = 2
            else:
                if max_y < max_z:
                    current_voxel_pos.y += dy
                    max_y += delta_y
                    step_dir = 1
                else:
                    current_voxel_pos.z += dz
                    max_z += delta_z
                    step_dir = 2
        return False

    def get_voxel_id(self, voxel_world_pos):
        # accept glm vectors or (x,y,z) tuples/lists
        if hasattr(voxel_world_pos, 'x'):
            wx, wy, wz = int(voxel_world_pos.x), int(voxel_world_pos.y), int(voxel_world_pos.z)
        else:
            wx, wy, wz = map(int, voxel_world_pos)

        cx = wx // CHUNK_SIZE
        cy = wy // CHUNK_SIZE
        cz = wz // CHUNK_SIZE

        if 0 <= cx < WORLD_W and 0 <= cy < WORLD_H and 0 <= cz < WORLD_D:
            chunk_index = int(cx + WORLD_W * cz + WORLD_AREA * cy)
            chunk = self.chunks[chunk_index]

            # local voxel coordinates inside chunk
            lx = wx - cx * CHUNK_SIZE
            ly = wy - cy * CHUNK_SIZE
            lz = wz - cz * CHUNK_SIZE

            voxel_index = int(lx + CHUNK_SIZE * lz + CHUNK_AREA * ly)
            # handle missing chunk defensively
            if chunk is None:
                return 0, 0, (0, 0, 0), None

            voxel_id = int(chunk.voxels[voxel_index])

            return voxel_id, voxel_index, (lx, ly, lz), chunk
        return 0, 0, (0, 0, 0), None
