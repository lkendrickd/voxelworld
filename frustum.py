from settings import *
import glm
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from camera import Camera

# small helper to compute vector length without relying on glm.length (improves static typing)
def vec_length(v):
    # glm.dot returns a scalar; use math.sqrt for a plain float result
    return math.sqrt(glm.dot(v, v))


class Frustum:
    """Simple sphere-frustum visibility test.

    This class precomputes tangent and secant of the half FOVs and tests
    a chunk (treated as a bounding sphere) against the six frustum planes.

    Improvements:
    - Uses per-chunk radius when available (fallback to CHUNK_SPHERE_RADIUS)
    - Normalizes camera basis vectors before use
    - Clamps depth used for width/height calculation to NEAR to avoid negative/zero depths
    - Uses radius margins computed with sec(half_angle)
    """

    def __init__(self, camera):
        # use forward-referenced type to avoid importing Camera at runtime
        self.cam: 'Camera' = camera

        # compute half-angles
        half_y = V_FOV * 0.5
        half_x = H_FOV * 0.5

        # safety guard for pathological FOV values
        cos_hy = math.cos(half_y)
        cos_hx = math.cos(half_x)
        self.sec_y = 1.0 / cos_hy if abs(cos_hy) > 1e-6 else 1e6
        self.sec_x = 1.0 / cos_hx if abs(cos_hx) > 1e-6 else 1e6

        self.tan_y = math.tan(half_y)
        self.tan_x = math.tan(half_x)

    def is_on_frustum(self, chunk) -> bool:
        """Return True if chunk's bounding sphere intersects the view frustum.

        chunk must expose `center` (glm.vec3). If it has `radius` attribute that
        will be used; otherwise `CHUNK_SPHERE_RADIUS` is used.
        """
        # use per-chunk radius when available
        radius = getattr(chunk, 'radius', CHUNK_SPHERE_RADIUS)

        # vector from camera to sphere center
        sphere_vec = chunk.center - self.cam.position

        # ensure camera basis vectors are normalized
        f = glm.normalize(self.cam.forward)
        u = glm.normalize(self.cam.up)
        r = glm.normalize(self.cam.right)

        # depth along camera forward
        sz = glm.dot(sphere_vec, f)

        # early reject by near/far plane using radius
        if sz + radius < NEAR:
            return False
        if sz - radius > FAR:
            return False

        # clamp depth for projection (avoid using negative depth for half-size)
        depth = max(sz, NEAR)

        # half extents at this depth
        half_h = depth * self.tan_y
        half_w = depth * self.tan_x

        # radius margin projected into screen space
        margin_h = radius * self.sec_y
        margin_w = radius * self.sec_x

        # check vertical (top/bottom)
        sy = glm.dot(sphere_vec, u)
        if abs(sy) > (half_h + margin_h):
            return False

        # check horizontal (left/right)
        sx = glm.dot(sphere_vec, r)
        if abs(sx) > (half_w + margin_w):
            return False

        return True

import threading
from typing import List, Set, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum
import time

class CullResult(Enum):
    INSIDE = 0
    OUTSIDE = 1
    INTERSECTING = 2

@dataclass
class CullStats:
    frustum_culled: int = 0
    occlusion_culled: int = 0
    distance_culled: int = 0
    temporal_skipped: int = 0
    total_tested: int = 0
    
    def reset(self):
        self.frustum_culled = 0
        self.occlusion_culled = 0
        self.distance_culled = 0
        self.temporal_skipped = 0
        self.total_tested = 0

# Enhanced version with additional culling techniques
class AdvancedFrustum(Frustum):
    def __init__(self, camera):
        super().__init__(camera)
        
        # Precompute frustum planes for more precise culling
        self.planes = []
        self._update_frustum_planes()
        
        # Temporal coherence - track previously visible chunks
        self.prev_visible: Set[int] = set()
        self.prev_camera_pos = glm.vec3(0)
        self.prev_camera_forward = glm.vec3(0, 0, 1)
        self.frame_count = 0
        
        # Distance-based culling parameters
        self.max_render_distance = FAR * 0.8  # Use 80% of far plane
        self.lod_distances = [
            CHUNK_SIZE * 2,      # High detail
            CHUNK_SIZE * 5,      # Medium detail  
            CHUNK_SIZE * 10,     # Low detail
            CHUNK_SIZE * 15      # Very low detail
        ]
        
        # Movement thresholds for temporal coherence
        self.movement_threshold = CHUNK_SIZE * 0.5
        self.rotation_threshold = 0.1
        
        # Statistics
        self.stats = CullStats()
        
    def _update_frustum_planes(self):
        """Compute the 6 frustum planes in world space"""
        pos = self.cam.position
        forward = self.cam.forward
        up = self.cam.up
        right = self.cam.right
        
        # Calculate frustum corners
        near_center = pos + forward * NEAR
        far_center = pos + forward * FAR
        
        near_height = NEAR * self.tan_y
        near_width = NEAR * self.tan_x
        far_height = FAR * self.tan_y
        far_width = FAR * self.tan_x
        
        # Plane normals (pointing inward) and distances
        self.planes = []
        
        # Near and far planes
        self.planes.append((forward, glm.dot(-forward, near_center)))  # Near
        self.planes.append((-forward, glm.dot(forward, far_center)))   # Far
        
        # Side planes
        # Left plane
        left_normal = glm.normalize(glm.cross(up, forward + right * self.tan_x))
        self.planes.append((left_normal, glm.dot(-left_normal, pos)))
        
        # Right plane  
        right_normal = glm.normalize(glm.cross(forward - right * self.tan_x, up))
        self.planes.append((right_normal, glm.dot(-right_normal, pos)))
        
        # Top plane
        top_normal = glm.normalize(glm.cross(right, forward + up * self.tan_y))
        self.planes.append((top_normal, glm.dot(-top_normal, pos)))
        
        # Bottom plane
        bottom_normal = glm.normalize(glm.cross(forward - up * self.tan_y, right))
        self.planes.append((bottom_normal, glm.dot(-bottom_normal, pos)))
    
    def has_camera_moved_significantly(self):
        """Check if camera moved enough to require full re-cull"""
        pos_delta = vec_length(self.cam.position - self.prev_camera_pos)
        dir_dot = glm.dot(self.cam.forward, self.prev_camera_forward)
        
        return (pos_delta > self.movement_threshold or 
                dir_dot < (1.0 - self.rotation_threshold))
    
    def sphere_vs_plane_distance(self, center, radius, plane_normal, plane_distance):
        """Calculate signed distance from sphere to plane"""
        return glm.dot(center, plane_normal) + plane_distance
    
    def precise_frustum_test(self, chunk) -> CullResult:
        """More precise frustum culling using plane equations"""
        center = chunk.center
        radius = CHUNK_SPHERE_RADIUS
        
        inside_count = 0
        
        for normal, distance in self.planes:
            sphere_distance = self.sphere_vs_plane_distance(center, radius, normal, distance)
            
            if sphere_distance < -radius:
                return CullResult.OUTSIDE
            elif sphere_distance < radius:
                # Sphere intersects plane
                pass
            else:
                inside_count += 1
        
        return CullResult.INSIDE if inside_count == 6 else CullResult.INTERSECTING
    
    def get_distance_to_chunk(self, chunk):
        """Get distance from camera to chunk center"""
        return vec_length(chunk.center - self.cam.position)
    
    def should_render_at_distance(self, chunk):
        """Distance-based culling"""
        distance = self.get_distance_to_chunk(chunk)
        return distance <= self.max_render_distance
    
    def get_lod_level(self, chunk):
        """Determine appropriate LOD level for chunk"""
        distance = self.get_distance_to_chunk(chunk)
        
        for i, max_dist in enumerate(self.lod_distances):
            if distance <= max_dist:
                return i
        
        return -1  # Too far, don't render
    
    def update_temporal_data(self, visible_chunks):
        """Update temporal coherence tracking"""
        self.prev_camera_pos = glm.vec3(self.cam.position)
        self.prev_camera_forward = glm.vec3(self.cam.forward)
        self.prev_visible = {id(chunk) for chunk in visible_chunks}
        self.frame_count += 1

class DistanceCuller:
    """Simple distance-based culling helper"""
    def __init__(self, max_distance=None):
        self.max_distance = max_distance or (FAR * 0.8)
        
    def should_render(self, chunk, camera_pos):
        distance = vec_length(chunk.center - camera_pos)
        return distance <= self.max_distance
    
    def get_distance(self, chunk, camera_pos):
        return vec_length(chunk.center - camera_pos)

class OcclusionCuller:
    """Simple CPU-based occlusion culling"""
    
    def __init__(self):
        self.occluders: List = []  # Large chunks that can occlude others
        self.min_occluder_size = CHUNK_SIZE * 1.5  # Minimum size to be an occluder
        self.enabled = True
        
    def add_occluder(self, chunk):
        """Add a chunk as a potential occluder"""
        # Only large chunks or special occluder objects
        if hasattr(chunk, 'is_solid') and chunk.is_solid:
            self.occluders.append(chunk)
    
    def clear_occluders(self):
        """Clear all occluders (call when chunks change significantly)"""
        self.occluders.clear()
    
    def is_occluded(self, chunk, camera):
        """Simple ray-based occlusion test"""
        if not self.enabled or not self.occluders:
            return False
            
        # Cast ray from camera to chunk center
        ray_start = camera.position
        ray_end = chunk.center
        ray_dir = glm.normalize(ray_end - ray_start)
        ray_distance = vec_length(ray_end - ray_start)
        
        for occluder in self.occluders:
            if occluder == chunk:
                continue
                
            # Simple sphere-ray intersection
            oc = ray_start - occluder.center
            a = glm.dot(ray_dir, ray_dir)
            b = 2.0 * glm.dot(oc, ray_dir)
            c = glm.dot(oc, oc) - (CHUNK_SPHERE_RADIUS * CHUNK_SPHERE_RADIUS)
            
            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                t = (-b - math.sqrt(discriminant)) / (2 * a)
                if 0.1 < t < ray_distance - 0.1:  # Small epsilon to avoid self-occlusion
                    return True
        
        return False

class MultiThreadedCuller:
    """Multi-threaded culling for large chunk counts"""
    
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        self.chunk_threshold = 100  # Use threading only for large counts
        
    def cull_chunks_threaded(self, frustum, chunks):
        """Divide chunks among threads for parallel culling"""
        if len(chunks) < self.chunk_threshold:
            return [c for c in chunks if frustum.is_on_frustum(c)]
        
        chunk_groups = self._divide_chunks(chunks)
        results = [[] for _ in range(len(chunk_groups))]
        threads = []
        
        for i, group in enumerate(chunk_groups):
            thread = threading.Thread(
                target=self._cull_chunk_group,
                args=(frustum, group, results[i])
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads and collect results
        for thread in threads:
            thread.join()
            
        visible_chunks = []
        for result_list in results:
            visible_chunks.extend(result_list)
            
        return visible_chunks
    
    def _divide_chunks(self, chunks):
        """Divide chunks into groups for threading"""
        group_size = max(1, len(chunks) // self.num_threads)
        groups = []
        
        for i in range(self.num_threads):
            start = i * group_size
            end = start + group_size if i < self.num_threads - 1 else len(chunks)
            if start < len(chunks):
                groups.append(chunks[start:end])
            
        return groups
    
    def _cull_chunk_group(self, frustum, chunks, result_list):
        """Cull a group of chunks in a thread"""
        for chunk in chunks:
            if frustum.is_on_frustum(chunk):
                result_list.append(chunk)

class MasterCullingSystem:
    """Orchestrates all culling techniques"""
    
    def __init__(self, camera):
        self.camera = camera
        
        # Initialize all culling systems
        self.frustum_culler = AdvancedFrustum(camera)
        self.distance_culler = DistanceCuller()
        self.occlusion_culler = OcclusionCuller()
        self.threaded_culler = MultiThreadedCuller()
        
        # Feature toggles
        self.enable_distance_culling = True
        self.enable_occlusion_culling = False  # Disabled by default (expensive)
        self.enable_temporal_coherence = True
        self.enable_threading = True
        self.enable_lod = True
        
        # Performance monitoring
        self.frame_times = deque(maxlen=60)
        self.last_visible_count = 0
        
    def cull_all_chunks(self, chunks):
        """Master culling function using all enabled techniques"""
        start_time = time.time()
        
        visible_chunks = []
        self.frustum_culler.stats.reset()
        self.frustum_culler.stats.total_tested = len(chunks)
        
        # Update frustum planes for current camera position
        self.frustum_culler._update_frustum_planes()
        
        # 1. Check camera movement for temporal coherence
        camera_moved_significantly = True
        if self.enable_temporal_coherence:
            camera_moved_significantly = self.frustum_culler.has_camera_moved_significantly()
            if not camera_moved_significantly:
                # Could implement temporal coherence optimization here
                # For now, just track the stats
                self.frustum_culler.stats.temporal_skipped = len(chunks) // 10
        
        # 2. Distance culling first (cheapest test)
        if self.enable_distance_culling:
            distance_passed = []
            for chunk in chunks:
                if self.distance_culler.should_render(chunk, self.camera.position):
                    distance_passed.append(chunk)
                else:
                    self.frustum_culler.stats.distance_culled += 1
        else:
            distance_passed = chunks
        
        # 3. Frustum culling
        if self.enable_threading and len(distance_passed) > 200:
            frustum_passed = self.threaded_culler.cull_chunks_threaded(
                self.frustum_culler, distance_passed
            )
        else:
            frustum_passed = []
            for chunk in distance_passed:
                if self.frustum_culler.is_on_frustum(chunk):
                    frustum_passed.append(chunk)
                else:
                    self.frustum_culler.stats.frustum_culled += 1
        
        # 4. Occlusion culling (most expensive, only if enabled)
        if self.enable_occlusion_culling and camera_moved_significantly:
            for chunk in frustum_passed:
                if not self.occlusion_culler.is_occluded(chunk, self.camera):
                    visible_chunks.append(chunk)
                else:
                    self.frustum_culler.stats.occlusion_culled += 1
        else:
            visible_chunks = frustum_passed
        
        # 5. Update temporal coherence data
        if self.enable_temporal_coherence:
            self.frustum_culler.update_temporal_data(visible_chunks)
        
        # Record performance
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        self.last_visible_count = len(visible_chunks)
        
        return visible_chunks, self.frustum_culler.stats
    
    def get_chunk_lod_level(self, chunk):
        """Get LOD level for a specific chunk"""
        if self.enable_lod:
            return self.frustum_culler.get_lod_level(chunk)
        return 0  # Highest detail
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.frame_times:
            return {}
            
        avg_time = sum(self.frame_times) / len(self.frame_times)
        max_time = max(self.frame_times)
        min_time = min(self.frame_times)
        
        return {
            "avg_cull_time_ms": avg_time * 1000,
            "max_cull_time_ms": max_time * 1000,
            "min_cull_time_ms": min_time * 1000,
            "visible_chunks": self.last_visible_count,
            "cull_efficiency": (1.0 - self.last_visible_count / max(1, self.frustum_culler.stats.total_tested)) * 100
        }
    
    def toggle_feature(self, feature_name, enabled=None):
        """Toggle or set culling features"""
        features = {
            'distance': 'enable_distance_culling',
            'occlusion': 'enable_occlusion_culling', 
            'temporal': 'enable_temporal_coherence',
            'threading': 'enable_threading',
            'lod': 'enable_lod'
        }
        
        if feature_name in features:
            attr = features[feature_name]
            if enabled is None:
                setattr(self, attr, not getattr(self, attr))
            else:
                setattr(self, attr, enabled)
            return getattr(self, attr)
        return None

# Usage examples:
"""
# Simple upgrade from original:
# OLD: frustum = Frustum(camera)
# NEW: frustum = AdvancedFrustum(camera)

# For maximum performance:
culler = MasterCullingSystem(camera)
visible_chunks, stats = culler.cull_all_chunks(all_chunks)

# Render with LOD:
for chunk in visible_chunks:
    lod_level = culler.get_chunk_lod_level(chunk)
    render_chunk(chunk, lod_level)

# Monitor performance:
perf = culler.get_performance_stats()
print(f"Culling: {perf['avg_cull_time_ms']:.2f}ms, Visible: {perf['visible_chunks']}")

# Toggle features at runtime:
culler.toggle_feature('occlusion', True)  # Enable occlusion culling
culler.toggle_feature('threading', False)  # Disable threading
"""