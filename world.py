from settings import *
from world_objects.chunk import Chunk
from voxel_handler import VoxelHandler
from frustum import MasterCullingSystem
import numpy as np
from typing import List, Optional


class World:
    def __init__(self, app):
        import time
        init_start = time.time()
        
        self.app = app
        self.chunks: List[Optional[Chunk]] = [None for _ in range(WORLD_VOL)]
        self.voxels = np.empty([WORLD_VOL, CHUNK_VOL], dtype='uint8')
        
        print("Building chunks...")
        chunk_start = time.time()
        self.build_chunks()
        chunk_end = time.time()
        print(f"Chunk generation took {chunk_end - chunk_start:.2f} seconds")
        
        # Progressive mesh building state
        self.chunks_to_mesh = [chunk for chunk in self.chunks if chunk is not None]
        self.meshes_built = 0
        self.total_chunks = len(self.chunks_to_mesh)
        self.meshes_per_frame = 10  # Build 10 meshes per frame
        self.mesh_building_complete = False
        
        print(f"Will build {self.total_chunks} meshes progressively...")
        
        print("Initializing voxel handler...")
        self.voxel_handler = VoxelHandler(self)

        # Master culling system (uses player's camera)
        try:
            self.culler = MasterCullingSystem(self.app.player)
        except Exception:
            self.culler = None
        self._frame_counter = 0
        self.show_cull_stats = False
        
        init_end = time.time()
        print(f"World initialization completed in {init_end - init_start:.2f} seconds")
        print("Mesh building will continue during gameplay...")

    def update(self):
        # Progressive mesh building
        if not self.mesh_building_complete and self.chunks_to_mesh:
            meshes_this_frame = 0
            while meshes_this_frame < self.meshes_per_frame and self.chunks_to_mesh:
                chunk = self.chunks_to_mesh.pop(0)
                chunk.build_mesh()
                self.meshes_built += 1
                meshes_this_frame += 1
            
            # Check if we're done
            if not self.chunks_to_mesh:
                self.mesh_building_complete = True
                print(f"Mesh building complete! Built {self.meshes_built} total meshes.")
            else:
                # Occasionally print progress (every 10th frame to avoid spam)
                if self._frame_counter % 10 == 0:
                    progress = (self.meshes_built / self.total_chunks) * 100
                    print(f"Mesh building progress: {progress:.1f}% ({self.meshes_built}/{self.total_chunks})")
        
        # Increment frame counter for both mesh building progress and culling stats
        self._frame_counter += 1
        
        self.voxel_handler.update()

    def build_chunks(self):
        for x in range(WORLD_W):
            for y in range(WORLD_H):
                for z in range(WORLD_D):
                    chunk = Chunk(self, position=(x, y, z))

                    chunk_index = x + WORLD_W * z + WORLD_AREA * y
                    self.chunks[chunk_index] = chunk

                    # put the chunk voxels in a separate array
                    self.voxels[chunk_index] = chunk.build_voxels()

                    # get pointer to voxels
                    chunk.voxels = self.voxels[chunk_index]

    def render(self):
        # If we have a master culler, use it to get visible chunks
        if self.culler is not None:
            # provide only non-empty chunks to culler
            candidates = [c for c in self.chunks if (c is not None and not c.is_empty)]
            visible_chunks, stats = self.culler.cull_all_chunks(candidates)

            # render visible chunks
            for chunk in visible_chunks:
                if chunk.mesh is not None:
                    chunk.set_uniform()
                    chunk.mesh.render()

            # optional stats printing every 60 frames
            if self.show_cull_stats:
                if self._frame_counter >= 60:
                    perf = self.culler.get_performance_stats()
                    print(f"Culling avg: {perf.get('avg_cull_time_ms', 0):.2f}ms, visible: {perf.get('visible_chunks', 0)}, efficiency: {perf.get('cull_efficiency', 0):.1f}%")
                    print(f"Cull details: frustum_culled={stats.frustum_culled}, distance_culled={stats.distance_culled}, occlusion_culled={stats.occlusion_culled}, temporal_skipped={stats.temporal_skipped}")
                    self._frame_counter = 0
        else:
            # Fallback to existing behavior
            for chunk in self.chunks:
                if chunk is not None and chunk.mesh is not None:
                    chunk.render()
