"""Terrain system - obstacles and water bodies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from noise import snoise2


@dataclass
class Obstacle:
    """A solid circular obstacle beings cannot pass through."""

    x: float
    y: float
    radius: float

    def contains(self, px: float, py: float) -> bool:
        """Check if a point is inside this obstacle."""
        dx = px - self.x
        dy = py - self.y
        return dx * dx + dy * dy < self.radius * self.radius

    def distance_to_edge(self, px: float, py: float) -> float:
        """Get distance from point to obstacle edge (negative if inside)."""
        dx = px - self.x
        dy = py - self.y
        dist_to_center = math.sqrt(dx * dx + dy * dy)
        return dist_to_center - self.radius

    def get_push_vector(self, px: float, py: float) -> tuple[float, float]:
        """
        Get a vector that would push a point out of this obstacle.

        Returns (0, 0) if point is not inside.
        """
        dx = px - self.x
        dy = py - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist >= self.radius:
            return (0.0, 0.0)

        if dist < 0.001:
            # Point is at center, push in arbitrary direction
            return (self.radius, 0.0)

        # Normalize and scale to push to edge
        push_dist = self.radius - dist + 0.1  # Small buffer
        return (dx / dist * push_dist, dy / dist * push_dist)


@dataclass
class WaterBody:
    """
    An organic water area defined by polygon vertices.

    Water bodies are defined as polygons for organic shapes.
    Uses ray-casting algorithm for point-in-polygon detection.
    """

    vertices: list[tuple[float, float]]
    # Cached bounding box for quick rejection
    _bbox: tuple[float, float, float, float] = field(default=None, repr=False, init=False)  # type: ignore

    def __post_init__(self) -> None:
        """Calculate bounding box for quick rejection."""
        if not self.vertices:
            self._bbox = (0, 0, 0, 0)
            return
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        self._bbox = (min(xs), min(ys), max(xs), max(ys))

    def contains(self, px: float, py: float) -> bool:
        """
        Check if a point is inside this water body using ray-casting.

        Returns True if point is inside the polygon.
        """
        # Quick bounding box rejection
        min_x, min_y, max_x, max_y = self._bbox
        if px < min_x or px > max_x or py < min_y or py > max_y:
            return False

        # Ray-casting algorithm
        n = len(self.vertices)
        if n < 3:
            return False

        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]

            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def distance_to_edge(self, px: float, py: float) -> float:
        """
        Get the minimum distance from point to any edge of the polygon.

        Returns positive if outside, negative if inside.
        """
        min_dist = float("inf")
        n = len(self.vertices)

        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]

            # Distance from point to line segment
            dist = self._point_to_segment_distance(px, py, x1, y1, x2, y2)
            if dist < min_dist:
                min_dist = dist

        # Negate if inside
        if self.contains(px, py):
            return -min_dist
        return min_dist

    def _point_to_segment_distance(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Calculate distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq < 0.0001:
            # Degenerate segment
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        # Project point onto line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def get_nearest_edge_point(self, px: float, py: float) -> tuple[float, float]:
        """Get the nearest point on the polygon edge."""
        min_dist = float("inf")
        nearest = (px, py)
        n = len(self.vertices)

        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]

            # Find nearest point on this segment
            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx * dx + dy * dy

            if length_sq < 0.0001:
                point = (x1, y1)
            else:
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
                point = (x1 + t * dx, y1 + t * dy)

            dist = (px - point[0]) ** 2 + (py - point[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = point

        return nearest

    @property
    def center(self) -> tuple[float, float]:
        """Get the centroid of the water body."""
        if not self.vertices:
            return (0.0, 0.0)
        cx = sum(v[0] for v in self.vertices) / len(self.vertices)
        cy = sum(v[1] for v in self.vertices) / len(self.vertices)
        return (cx, cy)


class HeightmapGenerator:
    """
    Generates terrain heightmaps using simplex noise.

    Supports multiple terrain styles:
    - "island": Radial falloff creates water around edges, land in center
    - "river_valley": Gradient edges with a meandering river channel
    """

    def __init__(
        self,
        width: int,
        height: int,
        seed: int,
        style: str = "island",
        water_level: float = 0.35,
        mountain_level: float = 0.75,
        resolution: int = 4,
    ):
        """
        Initialize heightmap generator.

        Args:
            width: World width in pixels
            height: World height in pixels
            seed: Random seed for noise generation
            style: Terrain style ("island" or "river_valley")
            water_level: Height threshold below which is water (0-1)
            mountain_level: Height threshold above which is mountain (0-1)
            resolution: Pixels per heightmap cell (lower = more detail, slower)
        """
        self.width = width
        self.height = height
        self.seed = seed
        self.style = style
        self.water_level = water_level
        self.mountain_level = mountain_level
        self.resolution = resolution

        # Grid dimensions
        self.grid_width = width // resolution
        self.grid_height = height // resolution

        # Generate the heightmap
        self.heightmap = self._generate_heightmap()

    def _generate_heightmap(self) -> np.ndarray:
        """Generate the full heightmap using simplex noise with falloff."""
        # Base noise parameters
        scale = 0.008  # Noise frequency (lower = larger features)
        octaves = 4
        persistence = 0.5
        lacunarity = 2.0

        heightmap = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # Generate multi-octave simplex noise
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # World coordinates
                wx = x * self.resolution
                wy = y * self.resolution

                # Multi-octave noise
                noise_val = 0.0
                amplitude = 1.0
                frequency = scale
                max_amplitude = 0.0

                for _ in range(octaves):
                    noise_val += amplitude * snoise2(
                        wx * frequency + self.seed,
                        wy * frequency + self.seed,
                    )
                    max_amplitude += amplitude
                    amplitude *= persistence
                    frequency *= lacunarity

                # Normalize to 0-1 range
                noise_val = (noise_val / max_amplitude + 1) / 2

                heightmap[y, x] = noise_val

        # Apply falloff based on style
        if self.style == "island":
            heightmap = self._apply_island_falloff(heightmap)
        elif self.style == "river_valley":
            heightmap = self._apply_river_valley_falloff(heightmap)

        return heightmap

    def _apply_island_falloff(self, heightmap: np.ndarray) -> np.ndarray:
        """Apply radial falloff to create island shape with water at edges."""
        center_x = self.grid_width / 2
        center_y = self.grid_height / 2
        max_dist = min(center_x, center_y) * 0.85  # Leave margin for water

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Distance from center (normalized 0-1)
                dx = (x - center_x) / center_x
                dy = (y - center_y) / center_y
                dist = math.sqrt(dx * dx + dy * dy)

                # Smooth falloff curve (steeper at edges)
                # Using smoothstep-like curve for natural transition
                falloff = 1.0 - min(1.0, dist * 1.1)
                falloff = falloff * falloff * (3 - 2 * falloff)  # Smoothstep

                # Apply falloff - subtract from height
                heightmap[y, x] = heightmap[y, x] * falloff

        return heightmap

    def _apply_river_valley_falloff(self, heightmap: np.ndarray) -> np.ndarray:
        """Apply gradient falloff with river channel for valley style."""
        rng = random.Random(self.seed)

        # Create mountain ridges at top and bottom edges
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Distance from horizontal center (for valley shape)
                edge_dist_y = min(y, self.grid_height - 1 - y) / (self.grid_height / 2)

                # Smooth falloff from edges
                edge_falloff = edge_dist_y * edge_dist_y * (3 - 2 * edge_dist_y)

                # Also add slight falloff at left/right edges
                edge_dist_x = min(x, self.grid_width - 1 - x) / (self.grid_width / 4)
                edge_x_falloff = min(1.0, edge_dist_x)

                heightmap[y, x] = heightmap[y, x] * edge_falloff * edge_x_falloff

        # Carve a meandering river channel through the center
        river_y = self.grid_height // 2
        river_width = max(3, self.grid_width // 25)

        for x in range(self.grid_width):
            # Meander the river
            river_y += rng.randint(-1, 1)
            river_y = max(river_width * 2, min(self.grid_height - river_width * 2, river_y))

            # Carve the channel
            for dy in range(-river_width, river_width + 1):
                y = river_y + dy
                if 0 <= y < self.grid_height:
                    # Smoother channel (deeper in center)
                    depth_factor = 1.0 - abs(dy) / river_width
                    heightmap[y, x] *= (1.0 - 0.8 * depth_factor)

        return heightmap

    def get_height(self, x: float, y: float) -> float:
        """
        Get height at world coordinates.

        Args:
            x: World x coordinate
            y: World y coordinate

        Returns:
            Height value (0-1)
        """
        # Convert to grid coordinates
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)

        # Clamp to grid bounds
        gx = max(0, min(self.grid_width - 1, gx))
        gy = max(0, min(self.grid_height - 1, gy))

        return float(self.heightmap[gy, gx])

    def is_water(self, x: float, y: float) -> bool:
        """Check if position is in water."""
        return self.get_height(x, y) < self.water_level

    def is_mountain(self, x: float, y: float) -> bool:
        """Check if position is a mountain/obstacle."""
        return self.get_height(x, y) > self.mountain_level

    def extract_water_polygons(
        self, simplify_tolerance: float = 8.0, min_area: float = 500.0
    ) -> list[list[tuple[float, float]]]:
        """
        Extract water body polygons using marching squares algorithm.

        Args:
            simplify_tolerance: Distance tolerance for polygon simplification
            min_area: Minimum polygon area to keep (filters tiny artifacts)

        Returns:
            List of polygon vertex lists (in world coordinates)
        """
        # Find contours at water level threshold
        contours = self._marching_squares(self.water_level, invert=True)

        # Convert to world coordinates and simplify
        polygons = []
        for contour in contours:
            if len(contour) < 4:
                continue

            # Convert grid coords to world coords
            world_contour = [
                (x * self.resolution, y * self.resolution)
                for x, y in contour
            ]

            # Simplify polygon
            simplified = self._simplify_polygon(world_contour, simplify_tolerance)

            if len(simplified) >= 3:
                # Filter by area to remove tiny artifacts
                area = self._polygon_area(simplified)
                if area >= min_area:
                    polygons.append(simplified)

        return polygons

    def _polygon_area(self, vertices: list[tuple[float, float]]) -> float:
        """Calculate the area of a polygon using the shoelace formula."""
        n = len(vertices)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        return abs(area) / 2.0

    def extract_mountain_polygons(
        self, simplify_tolerance: float = 8.0, min_area: float = 400.0
    ) -> list[list[tuple[float, float]]]:
        """
        Extract mountain/obstacle polygons using marching squares.

        Args:
            simplify_tolerance: Distance tolerance for polygon simplification
            min_area: Minimum polygon area to keep

        Returns:
            List of polygon vertex lists (in world coordinates)
        """
        contours = self._marching_squares(self.mountain_level, invert=False)

        polygons = []
        for contour in contours:
            if len(contour) < 4:
                continue

            world_contour = [
                (x * self.resolution, y * self.resolution)
                for x, y in contour
            ]

            simplified = self._simplify_polygon(world_contour, simplify_tolerance)

            if len(simplified) >= 3:
                area = self._polygon_area(simplified)
                if area >= min_area:
                    polygons.append(simplified)

        return polygons

    def _marching_squares(self, threshold: float, invert: bool = False) -> list[list[tuple[float, float]]]:
        """
        Marching squares algorithm to extract contours at threshold.

        Args:
            threshold: Height threshold for contour
            invert: If True, find regions below threshold; if False, above

        Returns:
            List of contour vertex lists (in grid coordinates)
        """
        # Create binary grid
        if invert:
            binary = (self.heightmap < threshold).astype(np.uint8)
        else:
            binary = (self.heightmap > threshold).astype(np.uint8)

        # Edge lookup table for marching squares (16 cases)
        # Each entry is a list of edge pairs to connect
        # Edges: 0=top, 1=right, 2=bottom, 3=left
        edge_table = [
            [],              # 0: no edges
            [(3, 2)],        # 1: bottom-left
            [(2, 1)],        # 2: bottom-right
            [(3, 1)],        # 3: bottom
            [(1, 0)],        # 4: top-right
            [(3, 2), (1, 0)],  # 5: bottom-left, top-right (ambiguous)
            [(2, 0)],        # 6: right
            [(3, 0)],        # 7: top, left, bottom
            [(0, 3)],        # 8: top-left
            [(0, 2)],        # 9: left
            [(0, 3), (2, 1)],  # 10: top-left, bottom-right (ambiguous)
            [(0, 1)],        # 11: top, right, bottom
            [(1, 3)],        # 12: top
            [(1, 2)],        # 13: top, left, right
            [(2, 3)],        # 14: top, bottom, right
            [],              # 15: all corners (no edges)
        ]

        # Track visited edges to build contours
        visited = set()
        contours = []

        # Process each cell
        for y in range(self.grid_height - 1):
            for x in range(self.grid_width - 1):
                # Get cell configuration (4-bit index)
                config = (
                    (binary[y, x] << 3) |
                    (binary[y, x + 1] << 2) |
                    (binary[y + 1, x + 1] << 1) |
                    binary[y + 1, x]
                )

                edges = edge_table[config]
                if not edges:
                    continue

                # For each edge pair, try to trace a contour
                for edge_start, edge_end in edges:
                    edge_key = (x, y, edge_start, edge_end)
                    if edge_key in visited:
                        continue

                    contour = self._trace_contour(
                        binary, edge_table, x, y, edge_start, visited
                    )
                    if contour and len(contour) >= 3:
                        contours.append(contour)

        return contours

    def _trace_contour(
        self,
        binary: np.ndarray,
        edge_table: list,
        start_x: int,
        start_y: int,
        start_edge: int,
        visited: set,
    ) -> list[tuple[float, float]]:
        """Trace a single contour starting from given cell and edge."""
        contour = []
        x, y = start_x, start_y
        current_edge = start_edge
        max_steps = self.grid_width * self.grid_height * 2

        for _ in range(max_steps):
            # Get edge midpoint in grid coordinates
            point = self._get_edge_point(x, y, current_edge)
            contour.append(point)

            # Mark as visited
            visited.add((x, y, current_edge, (current_edge + 2) % 4))

            # Move to next cell based on current edge
            # Edge 0 (top) -> move up
            # Edge 1 (right) -> move right
            # Edge 2 (bottom) -> move down
            # Edge 3 (left) -> move left
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][current_edge]
            next_x, next_y = x + dx, y + dy

            # Check bounds
            if not (0 <= next_x < self.grid_width - 1 and 0 <= next_y < self.grid_height - 1):
                break

            # Get next cell config
            config = (
                (binary[next_y, next_x] << 3) |
                (binary[next_y, next_x + 1] << 2) |
                (binary[next_y + 1, next_x + 1] << 1) |
                binary[next_y + 1, next_x]
            )

            edges = edge_table[config]
            if not edges:
                break

            # Find the continuing edge (entering from opposite side)
            entering_edge = (current_edge + 2) % 4
            found = False

            for edge_start, edge_end in edges:
                if edge_start == entering_edge:
                    current_edge = edge_end
                    x, y = next_x, next_y
                    found = True
                    break
                elif edge_end == entering_edge:
                    current_edge = edge_start
                    x, y = next_x, next_y
                    found = True
                    break

            if not found:
                break

            # Check if we've returned to start
            if x == start_x and y == start_y and current_edge == start_edge:
                break

        return contour

    def _get_edge_point(self, x: int, y: int, edge: int) -> tuple[float, float]:
        """Get the midpoint of an edge in grid coordinates."""
        # Edge 0: top (x+0.5, y)
        # Edge 1: right (x+1, y+0.5)
        # Edge 2: bottom (x+0.5, y+1)
        # Edge 3: left (x, y+0.5)
        offsets = [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]
        ox, oy = offsets[edge]
        return (x + ox, y + oy)

    def _simplify_polygon(
        self, vertices: list[tuple[float, float]], tolerance: float
    ) -> list[tuple[float, float]]:
        """
        Simplify polygon using Ramer-Douglas-Peucker algorithm.

        Args:
            vertices: List of (x, y) vertices
            tolerance: Maximum distance for point removal

        Returns:
            Simplified vertex list
        """
        if len(vertices) < 3:
            return vertices

        # Find point furthest from line between first and last
        start = vertices[0]
        end = vertices[-1]
        max_dist = 0.0
        max_idx = 0

        for i in range(1, len(vertices) - 1):
            dist = self._point_line_distance(vertices[i], start, end)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        # If max distance exceeds tolerance, recursively simplify
        if max_dist > tolerance:
            left = self._simplify_polygon(vertices[: max_idx + 1], tolerance)
            right = self._simplify_polygon(vertices[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [start, end]

    def _point_line_distance(
        self, point: tuple[float, float], line_start: tuple[float, float], line_end: tuple[float, float]
    ) -> float:
        """Calculate perpendicular distance from point to line segment."""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq < 0.0001:
            return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

        # Project point onto line
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)


class MountainObstacle:
    """
    A polygon-based mountain obstacle that beings cannot pass through.

    Similar to WaterBody but represents elevated terrain.
    """

    def __init__(self, vertices: list[tuple[float, float]]):
        """Initialize mountain obstacle with polygon vertices."""
        self.vertices = vertices
        self._bbox: tuple[float, float, float, float]

        if not vertices:
            self._bbox = (0, 0, 0, 0)
        else:
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            self._bbox = (min(xs), min(ys), max(xs), max(ys))

        # Calculate center and approximate radius for compatibility
        if vertices:
            self.x = sum(v[0] for v in vertices) / len(vertices)
            self.y = sum(v[1] for v in vertices) / len(vertices)
            self.radius = max(
                math.sqrt((v[0] - self.x) ** 2 + (v[1] - self.y) ** 2)
                for v in vertices
            )
        else:
            self.x, self.y, self.radius = 0.0, 0.0, 0.0

    def contains(self, px: float, py: float) -> bool:
        """Check if point is inside the mountain polygon."""
        min_x, min_y, max_x, max_y = self._bbox
        if px < min_x or px > max_x or py < min_y or py > max_y:
            return False

        n = len(self.vertices)
        if n < 3:
            return False

        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]

            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def get_push_vector(self, px: float, py: float) -> tuple[float, float]:
        """Get vector to push point out of mountain."""
        if not self.contains(px, py):
            return (0.0, 0.0)

        # Find nearest edge and push perpendicular to it
        min_dist = float("inf")
        push_vec = (0.0, 0.0)
        n = len(self.vertices)

        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]

            # Find nearest point on edge
            dx, dy = x2 - x1, y2 - y1
            length_sq = dx * dx + dy * dy

            if length_sq < 0.0001:
                continue

            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
            nearest_x = x1 + t * dx
            nearest_y = y1 + t * dy

            dist = math.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)

            if dist < min_dist:
                min_dist = dist
                # Push away from edge
                if dist > 0.001:
                    push_vec = (
                        (px - nearest_x) / dist * (dist + 1),
                        (py - nearest_y) / dist * (dist + 1),
                    )
                else:
                    # Point is on edge, push perpendicular
                    length = math.sqrt(length_sq)
                    push_vec = (-dy / length * 2, dx / length * 2)

        return push_vec


class Terrain:
    """
    Manages terrain features: obstacles and water bodies.

    Provides collision detection and water queries.
    """

    def __init__(self, width: float, height: float):
        """
        Initialize terrain.

        Args:
            width: World width
            height: World height
        """
        self.width = width
        self.height = height
        self.obstacles: list[Obstacle] = []
        self.water_bodies: list[WaterBody] = []
        self.mountain_obstacles: list[MountainObstacle] = []
        self.heightmap: HeightmapGenerator | None = None

    def add_obstacle(self, x: float, y: float, radius: float) -> Obstacle:
        """Add a circular obstacle."""
        obstacle = Obstacle(x=x, y=y, radius=radius)
        self.obstacles.append(obstacle)
        return obstacle

    def add_water_body(self, vertices: list[tuple[float, float]]) -> WaterBody:
        """Add a water body with given polygon vertices."""
        water = WaterBody(vertices=vertices)
        self.water_bodies.append(water)
        return water

    def is_blocked(self, x: float, y: float) -> bool:
        """Check if a position is blocked by an obstacle or mountain."""
        # Use heightmap for fast, accurate check if available
        if self.heightmap is not None:
            if self.heightmap.is_mountain(x, y):
                return True

        for obstacle in self.obstacles:
            if obstacle.contains(x, y):
                return True
        for mountain in self.mountain_obstacles:
            if mountain.contains(x, y):
                return True
        return False

    def is_in_water(self, x: float, y: float) -> bool:
        """Check if a position is inside any water body."""
        # Use heightmap for fast, accurate check if available
        if self.heightmap is not None:
            return self.heightmap.is_water(x, y)

        # Fallback to polygon-based check
        for water in self.water_bodies:
            if water.contains(x, y):
                return True
        return False

    def get_water_body_at(self, x: float, y: float) -> WaterBody | None:
        """Get the water body at a position, if any."""
        for water in self.water_bodies:
            if water.contains(x, y):
                return water
        return None

    def distance_to_water(self, x: float, y: float) -> float:
        """
        Get distance to nearest water body edge.

        Returns negative if inside water, positive if outside.
        Returns infinity if no water bodies exist.
        """
        if not self.water_bodies:
            return float("inf")

        min_dist = float("inf")
        for water in self.water_bodies:
            dist = water.distance_to_edge(x, y)
            if abs(dist) < abs(min_dist):
                min_dist = dist

        return min_dist

    def get_push_out_of_obstacles(self, x: float, y: float) -> tuple[float, float]:
        """
        Get a vector to push a point out of any obstacles it's inside.

        Returns (0, 0) if not inside any obstacle.
        """
        total_dx, total_dy = 0.0, 0.0

        for obstacle in self.obstacles:
            dx, dy = obstacle.get_push_vector(x, y)
            total_dx += dx
            total_dy += dy

        for mountain in self.mountain_obstacles:
            dx, dy = mountain.get_push_vector(x, y)
            total_dx += dx
            total_dy += dy

        return (total_dx, total_dy)

    def validate_move(
        self, from_x: float, from_y: float, to_x: float, to_y: float
    ) -> tuple[float, float]:
        """
        Validate a movement and return the actual destination.

        If the destination is blocked, slides along the obstacle or returns
        the original position.
        """
        # Check if destination is blocked
        if not self.is_blocked(to_x, to_y):
            return (to_x, to_y)

        # Try to slide along obstacles
        push_x, push_y = self.get_push_out_of_obstacles(to_x, to_y)
        new_x = to_x + push_x
        new_y = to_y + push_y

        # If still blocked, return original position
        if self.is_blocked(new_x, new_y):
            return (from_x, from_y)

        return (new_x, new_y)

    def generate_from_heightmap(
        self,
        seed: int,
        style: str = "island",
        water_level: float = 0.35,
        mountain_level: float = 0.75,
    ) -> None:
        """
        Generate terrain features from a heightmap.

        Uses simplex noise to create a coherent heightmap, then extracts
        water bodies and mountain obstacles from elevation thresholds.

        Args:
            seed: Random seed for noise generation
            style: Terrain style ("island" or "river_valley")
            water_level: Height below which is water (0-1)
            mountain_level: Height above which is mountain (0-1)
        """
        # Clear existing terrain
        self.obstacles.clear()
        self.water_bodies.clear()
        self.mountain_obstacles.clear()

        # Generate heightmap
        self.heightmap = HeightmapGenerator(
            width=int(self.width),
            height=int(self.height),
            seed=seed,
            style=style,
            water_level=water_level,
            mountain_level=mountain_level,
        )

        # Extract water polygons (use larger min_area to avoid too many small polygons)
        water_polygons = self.heightmap.extract_water_polygons(
            simplify_tolerance=6.0, min_area=2000.0
        )
        for vertices in water_polygons:
            if len(vertices) >= 3:
                self.add_water_body(vertices)

        # Extract mountain polygons
        mountain_polygons = self.heightmap.extract_mountain_polygons(
            simplify_tolerance=8.0, min_area=800.0
        )
        for vertices in mountain_polygons:
            if len(vertices) >= 3:
                self.mountain_obstacles.append(MountainObstacle(vertices))

    def get_height(self, x: float, y: float) -> float:
        """
        Get terrain height at position.

        Returns:
            Height value (0-1), or 0.5 if no heightmap is generated.
        """
        if self.heightmap is not None:
            return self.heightmap.get_height(x, y)
        return 0.5

    def generate_obstacles(
        self, rng: random.Random, count: int, min_radius: float = 15.0, max_radius: float = 40.0
    ) -> None:
        """
        Generate random circular obstacles (legacy method).

        Args:
            rng: Random number generator
            count: Number of obstacles to generate
            min_radius: Minimum obstacle radius
            max_radius: Maximum obstacle radius
        """
        margin = max_radius * 2

        for _ in range(count):
            x = rng.uniform(margin, self.width - margin)
            y = rng.uniform(margin, self.height - margin)
            radius = rng.uniform(min_radius, max_radius)
            self.add_obstacle(x, y, radius)

    def generate_water_bodies(
        self,
        rng: random.Random,
        count: int,
        min_size: float = 40.0,
        max_size: float = 100.0,
    ) -> None:
        """
        Generate organic water bodies (lakes/ponds).

        Uses a blob generation technique with randomized vertices.

        Args:
            rng: Random number generator
            count: Number of water bodies to generate
            min_size: Minimum water body size
            max_size: Maximum water body size
        """
        margin = max_size

        for _ in range(count):
            center_x = rng.uniform(margin, self.width - margin)
            center_y = rng.uniform(margin, self.height - margin)
            size = rng.uniform(min_size, max_size)

            vertices = self._generate_blob_vertices(rng, center_x, center_y, size)
            self.add_water_body(vertices)

    def _generate_blob_vertices(
        self,
        rng: random.Random,
        center_x: float,
        center_y: float,
        base_radius: float,
        num_points: int = 12,
    ) -> list[tuple[float, float]]:
        """
        Generate organic blob-shaped polygon vertices.

        Uses varying radius at each angle to create organic shapes.
        """
        vertices: list[tuple[float, float]] = []
        angle_step = 2 * math.pi / num_points

        for i in range(num_points):
            angle = i * angle_step

            # Vary radius for organic shape (0.5 to 1.2 of base)
            radius_variation = rng.uniform(0.5, 1.2)
            radius = base_radius * radius_variation

            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            vertices.append((x, y))

        return vertices

    def generate_river(
        self,
        rng: random.Random,
        start: tuple[float, float] | None = None,
        direction: str = "horizontal",
        width: float = 30.0,
        segments: int = 8,
    ) -> None:
        """
        Generate a river that flows across the world.

        Args:
            rng: Random number generator
            start: Starting point (None for random edge)
            direction: 'horizontal' or 'vertical'
            width: River width
            segments: Number of segments for river path
        """
        if direction == "horizontal":
            if start is None:
                start = (0, rng.uniform(self.height * 0.3, self.height * 0.7))

            # Generate river path points
            path: list[tuple[float, float]] = [start]
            step_x = self.width / segments

            for i in range(1, segments + 1):
                x = step_x * i
                # Meander the river
                prev_y = path[-1][1]
                y = prev_y + rng.uniform(-50, 50)
                y = max(width, min(self.height - width, y))
                path.append((x, y))

        else:  # vertical
            if start is None:
                start = (rng.uniform(self.width * 0.3, self.width * 0.7), 0)

            path = [start]
            step_y = self.height / segments

            for i in range(1, segments + 1):
                y = step_y * i
                prev_x = path[-1][0]
                x = prev_x + rng.uniform(-50, 50)
                x = max(width, min(self.width - width, x))
                path.append((x, y))

        # Convert path to polygon (outline both sides of river)
        half_width = width / 2
        left_side: list[tuple[float, float]] = []
        right_side: list[tuple[float, float]] = []

        for i, (px, py) in enumerate(path):
            # Get perpendicular direction
            if i == 0:
                dx, dy = path[1][0] - px, path[1][1] - py
            elif i == len(path) - 1:
                dx, dy = px - path[-2][0], py - path[-2][1]
            else:
                dx, dy = path[i + 1][0] - path[i - 1][0], path[i + 1][1] - path[i - 1][1]

            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                dx, dy = dx / length, dy / length

            # Perpendicular vector
            perp_x, perp_y = -dy, dx

            left_side.append((px + perp_x * half_width, py + perp_y * half_width))
            right_side.append((px - perp_x * half_width, py - perp_y * half_width))

        # Combine to form polygon (left side forward, right side backward)
        vertices = left_side + right_side[::-1]
        self.add_water_body(vertices)

