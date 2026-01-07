"""World simulation - manages all entities and their interactions."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

from ..config import BeingConfig, WorldConfig
from .being import Action, Being
from .resource import Resource, ResourceType
from .spatial import SpatialGrid
from .terrain import Terrain


@dataclass
class WorldStats:
    """Statistics about the current world state."""

    tick: int = 0
    beings_alive: int = 0
    beings_died_this_tick: int = 0
    total_deaths: int = 0
    food_count: int = 0
    water_count: int = 0
    food_consumed_this_tick: float = 0.0
    water_consumed_this_tick: float = 0.0
    # Averages across all beings
    avg_energy: float = 0.0
    avg_food: float = 0.0
    avg_water: float = 0.0


class StatsHistory:
    """Tracks statistics over time for charting."""

    def __init__(self, max_length: int = 300):
        """
        Initialize stats history.

        Args:
            max_length: Maximum number of ticks to keep in history
        """
        self.max_length = max_length
        self.population: deque[int] = deque(maxlen=max_length)
        self.avg_energy: deque[float] = deque(maxlen=max_length)
        self.avg_food: deque[float] = deque(maxlen=max_length)
        self.avg_water: deque[float] = deque(maxlen=max_length)
        self.food_count: deque[int] = deque(maxlen=max_length)
        self.water_count: deque[int] = deque(maxlen=max_length)
        self.deaths_per_tick: deque[int] = deque(maxlen=max_length)

    def record(self, stats: WorldStats) -> None:
        """Record current stats to history."""
        self.population.append(stats.beings_alive)
        self.avg_energy.append(stats.avg_energy)
        self.avg_food.append(stats.avg_food)
        self.avg_water.append(stats.avg_water)
        self.food_count.append(stats.food_count)
        self.water_count.append(stats.water_count)
        self.deaths_per_tick.append(stats.beings_died_this_tick)


class World:
    """
    The simulation world containing all entities.

    Manages:
    - Beings and their lifecycle
    - Resources (food, water) and spawning
    - Spatial indexing for efficient queries
    - Simulation stepping
    """

    def __init__(self, world_config: WorldConfig, being_config: BeingConfig):
        """
        Initialize the world.

        Args:
            world_config: Configuration for world parameters
            being_config: Configuration for being attributes
        """
        self.config = world_config
        self.being_config = being_config
        self.width = world_config.width
        self.height = world_config.height

        # Seeded random number generator for reproducibility
        if world_config.seed is not None:
            self.seed = world_config.seed
        else:
            self.seed = random.randint(0, 2**31 - 1)
        self.rng = random.Random(self.seed)

        # Entity storage
        self.beings: list[Being] = []
        self.resources: list[Resource] = []

        # Spatial grids for efficient lookups
        self.being_grid: SpatialGrid[Being] = SpatialGrid(
            self.width, self.height, cell_size=64.0
        )
        self.resource_grid: SpatialGrid[Resource] = SpatialGrid(
            self.width, self.height, cell_size=64.0
        )

        # Terrain (obstacles and water bodies)
        self.terrain = Terrain(self.width, self.height)

        # Statistics
        self.stats = WorldStats()
        self.stats_history = StatsHistory()

        # Resource spawn accumulators
        self._food_spawn_accumulator = 0.0
        self._water_spawn_accumulator = 0.0

    def spawn_being(self, x: float | None = None, y: float | None = None) -> Being:
        """
        Spawn a new being at the given position (or random if not specified).

        Avoids spawning inside obstacles or water.

        Returns the spawned being.
        """
        max_attempts = 50

        if x is None or y is None:
            for _ in range(max_attempts):
                test_x = self.rng.uniform(0, self.width) if x is None else x
                test_y = self.rng.uniform(0, self.height) if y is None else y

                # Check if position is valid (not in obstacle or water)
                if not self.terrain.is_blocked(test_x, test_y) and not self.terrain.is_in_water(test_x, test_y):
                    x, y = test_x, test_y
                    break
            else:
                # Fallback to random position if can't find valid spot
                x = self.rng.uniform(0, self.width) if x is None else x
                y = self.rng.uniform(0, self.height) if y is None else y

        # Generate a deterministic sub-seed for this being
        being_seed = self.rng.randint(0, 2**31 - 1)

        being = Being(
            x=x,
            y=y,
            energy_loss_idle=self.being_config.energy_loss_idle,
            energy_loss_moving=self.being_config.energy_loss_moving,
            food_to_energy=self.being_config.food_to_energy,
            water_to_energy=self.being_config.water_to_energy,
            base_speed=self.being_config.base_speed,
            turn_speed=self.being_config.turn_speed,
            vision_range=self.being_config.vision_range,
            vision_angle=self.being_config.vision_angle,
            hunger_threshold=self.being_config.hunger_threshold,
            thirst_threshold=self.being_config.thirst_threshold,
            rng_seed=being_seed,
        )

        self.beings.append(being)
        self.being_grid.insert(being)
        return being

    def spawn_resource(
        self,
        resource_type: ResourceType,
        x: float | None = None,
        y: float | None = None,
        use_clustering: bool = True,
    ) -> Resource:
        """
        Spawn a new resource at the given position (or random/clustered if not specified).

        Args:
            resource_type: Type of resource to spawn
            x: X position (None for auto-placement)
            y: Y position (None for auto-placement)
            use_clustering: Whether to apply clustering logic when x/y are None

        Returns the spawned resource.
        """
        if x is None and y is None:
            # Decide whether to cluster near existing resource
            if (use_clustering and
                self.config.resource_clustering > 0 and
                self.rng.random() < self.config.resource_clustering):
                # Find existing resources of same type to cluster near
                same_type = [r for r in self.resources if r.type == resource_type]
                if same_type:
                    # Pick a random existing resource to cluster near
                    parent = self.rng.choice(same_type)
                    angle = self.rng.uniform(0, 2 * 3.14159)
                    distance = self.rng.uniform(5, self.config.cluster_radius)
                    x = parent.x + distance * math.cos(angle)
                    y = parent.y + distance * math.sin(angle)
                    # Wrap around world boundaries
                    x = x % self.width
                    y = y % self.height

        # Fall back to random if still None, avoiding obstacles/water
        if x is None or y is None:
            max_attempts = 30
            for _ in range(max_attempts):
                test_x = self.rng.uniform(0, self.width) if x is None else x
                test_y = self.rng.uniform(0, self.height) if y is None else y

                if not self.terrain.is_blocked(test_x, test_y) and not self.terrain.is_in_water(test_x, test_y):
                    x, y = test_x, test_y
                    break
            else:
                # Fallback if can't find valid position
                x = self.rng.uniform(0, self.width) if x is None else x
                y = self.rng.uniform(0, self.height) if y is None else y

        resource = Resource(x=x, y=y, type=resource_type)
        self.resources.append(resource)
        self.resource_grid.insert(resource)
        return resource

    def initialize(self) -> None:
        """Initialize the world with starting population, resources, and terrain."""
        # Generate terrain using heightmap-based generation
        self.terrain.generate_from_heightmap(
            seed=self.seed,
            style=self.config.terrain_style,
            water_level=self.config.water_level,
            mountain_level=self.config.mountain_level,
        )

        # Spawn initial beings (avoiding obstacles)
        for _ in range(self.config.initial_population):
            self.spawn_being()

        # Spawn initial resources (food only - water is from water bodies now)
        for _ in range(self.config.initial_food):
            self.spawn_resource(ResourceType.FOOD)

        self.stats.beings_alive = len(self.beings)
        self.stats.food_count = sum(1 for r in self.resources if r.type == ResourceType.FOOD)
        self.stats.water_count = sum(1 for r in self.resources if r.type == ResourceType.WATER)

        # Compute initial averages
        if self.beings:
            self.stats.avg_energy = sum(b.energy for b in self.beings) / len(self.beings)
            self.stats.avg_food = sum(b.food for b in self.beings) / len(self.beings)
            self.stats.avg_water = sum(b.water for b in self.beings) / len(self.beings)

        # Record initial state
        self.stats_history.record(self.stats)

    def get_visible_resources(
        self, being: Being, resource_type: ResourceType
    ) -> Iterator[Resource]:
        """Get resources of a given type visible to a being."""
        for resource in self.resource_grid.get_nearby(
            being.x, being.y, being.vision_range
        ):
            if resource.type == resource_type and being.can_see(resource.x, resource.y):
                yield resource

    def get_nearby_resources(
        self, being: Being, radius: float = 10.0
    ) -> Iterator[Resource]:
        """Get resources within consumption range of a being."""
        for resource in self.resource_grid.get_nearby(being.x, being.y, radius):
            yield resource

    def can_drink_from_water_body(self, being: Being) -> tuple[bool, float]:
        """
        Check if a being can drink from a water body.

        Returns:
            (can_drink, drink_rate_multiplier)
            - can_drink: True if being is near or in water
            - drink_rate_multiplier: 2.0 if inside water, 1.0 if at edge
        """
        edge_distance = 15.0  # Distance from edge to drink

        # Check if inside water (faster drinking)
        if self.terrain.is_in_water(being.x, being.y):
            return (True, 2.0)

        # Check if near water edge
        dist_to_water = self.terrain.distance_to_water(being.x, being.y)
        if dist_to_water <= edge_distance:
            return (True, 1.0)

        return (False, 0.0)

    def get_nearest_water_body_edge(self, being: Being) -> tuple[float, float] | None:
        """Get the nearest point on any water body edge visible to the being."""
        best_point: tuple[float, float] | None = None
        best_dist = float("inf")

        for water in self.terrain.water_bodies:
            edge_point = water.get_nearest_edge_point(being.x, being.y)
            dist = being.get_distance_to(edge_point[0], edge_point[1])

            if dist < best_dist and dist <= being.vision_range:
                if being.can_see(edge_point[0], edge_point[1]):
                    best_dist = dist
                    best_point = edge_point

        return best_point

    def step(self) -> None:
        """
        Advance the simulation by one tick.

        This:
        1. Updates each being (chooses action, moves, consumes energy)
        2. Handles resource consumption
        3. Removes dead beings and depleted resources
        4. Spawns new resources
        """
        self.stats.tick += 1
        self.stats.beings_died_this_tick = 0
        self.stats.food_consumed_this_tick = 0.0
        self.stats.water_consumed_this_tick = 0.0

        # Update each being
        dead_beings: list[Being] = []

        for being in self.beings:
            if not being.is_alive:
                dead_beings.append(being)
                continue

            # Decide action using simple AI (will be expanded)
            action = self._decide_action(being)

            # Handle consumption if trying to eat/drink
            if action == Action.EAT or action == Action.DRINK:
                consumed = self._try_consume(being, action)
                if consumed == 0:
                    # No resource nearby, move instead
                    action = Action.MOVE_FORWARD

            # Store old position for collision checking
            old_x, old_y = being.x, being.y

            # Update being state
            being.update(action, self.width, self.height)

            # Check terrain collision and adjust position if needed
            if being.speed > 0:
                # Check if new position is valid (not in obstacle)
                if self.terrain.is_blocked(being.x, being.y):
                    # Use terrain to find valid position
                    new_x, new_y = self.terrain.validate_move(
                        old_x, old_y, being.x, being.y
                    )
                    being.x, being.y = new_x, new_y

                # Update spatial grid
                self.being_grid.update(being)

            # Check if died this tick
            if not being.is_alive:
                dead_beings.append(being)

        # Remove dead beings
        for being in dead_beings:
            self.beings.remove(being)
            self.being_grid.remove(being)
            self.stats.beings_died_this_tick += 1
            self.stats.total_deaths += 1

        # Remove depleted resources
        depleted: list[Resource] = []
        for resource in self.resources:
            if resource.is_depleted:
                depleted.append(resource)

        for resource in depleted:
            self.resources.remove(resource)
            self.resource_grid.remove(resource)

        # Spawn new resources
        self._spawn_resources()

        # Update stats
        self.stats.beings_alive = len(self.beings)
        self.stats.food_count = sum(1 for r in self.resources if r.type == ResourceType.FOOD)
        self.stats.water_count = sum(1 for r in self.resources if r.type == ResourceType.WATER)

        # Compute averages across beings
        if self.beings:
            self.stats.avg_energy = sum(b.energy for b in self.beings) / len(self.beings)
            self.stats.avg_food = sum(b.food for b in self.beings) / len(self.beings)
            self.stats.avg_water = sum(b.water for b in self.beings) / len(self.beings)
        else:
            self.stats.avg_energy = 0.0
            self.stats.avg_food = 0.0
            self.stats.avg_water = 0.0

        # Record to history
        self.stats_history.record(self.stats)

    def _decide_action(self, being: Being) -> Action:
        """
        Decide what action a being should take.

        Simple rule-based AI:
        - If hungry and food visible: move toward food
        - If thirsty and near water body: drink
        - If thirsty and water body visible: move toward water
        - Otherwise: wander randomly
        """
        # Check for nearby food to consume
        consume_radius = 8.0

        for resource in self.get_nearby_resources(being, consume_radius):
            if resource.type == ResourceType.FOOD and being.is_hungry:
                return Action.EAT

        # Check if can drink from water body
        can_drink, _ = self.can_drink_from_water_body(being)
        if can_drink and being.is_thirsty:
            return Action.DRINK

        # Look for targets to move toward
        target_pos: tuple[float, float] | None = None
        target_distance = float("inf")

        # Prioritize based on most urgent need
        if being.is_thirsty and being.water < being.food:
            # Look for water body first
            water_edge = self.get_nearest_water_body_edge(being)
            if water_edge is not None:
                dist = being.get_distance_to(water_edge[0], water_edge[1])
                if dist < target_distance:
                    target_pos = water_edge
                    target_distance = dist

        if target_pos is None and being.is_hungry:
            # Look for food
            for resource in self.get_visible_resources(being, ResourceType.FOOD):
                dist = being.get_distance_to(resource.x, resource.y)
                if dist < target_distance:
                    target_pos = (resource.x, resource.y)
                    target_distance = dist

        if target_pos is None and being.is_thirsty:
            # Fallback: look for water if food not found
            water_edge = self.get_nearest_water_body_edge(being)
            if water_edge is not None:
                dist = being.get_distance_to(water_edge[0], water_edge[1])
                if dist < target_distance:
                    target_pos = water_edge
                    target_distance = dist

        # Move toward target if found
        if target_pos is not None:
            target_angle = being.get_angle_to(target_pos[0], target_pos[1])
            angle_diff = being.angle_diff(target_angle)

            # Turn toward target
            if abs(angle_diff) > being.turn_speed:
                return Action.TURN_LEFT if angle_diff > 0 else Action.TURN_RIGHT
            else:
                return Action.MOVE_FORWARD

        # No target: wander randomly using being's own RNG
        rand = being.rng.random()
        if rand < 0.7:
            return Action.MOVE_FORWARD
        elif rand < 0.85:
            return Action.TURN_LEFT
        else:
            return Action.TURN_RIGHT

    def _try_consume(self, being: Being, action: Action) -> float:
        """Try to consume a nearby resource or water body. Returns amount consumed."""
        consume_radius = 8.0

        if action == Action.EAT:
            # Consume food from resources
            for resource in self.get_nearby_resources(being, consume_radius):
                if resource.type == ResourceType.FOOD:
                    consumed = being.consume(resource)
                    self.stats.food_consumed_this_tick += consumed
                    return consumed
        elif action == Action.DRINK:
            # Drink from water body
            can_drink, rate_multiplier = self.can_drink_from_water_body(being)
            if can_drink:
                # Water bodies have infinite water - just gain water based on rate
                base_amount = 0.15  # Base drink amount per tick
                amount = base_amount * rate_multiplier
                being.water = min(1.0, being.water + amount)
                self.stats.water_consumed_this_tick += amount
                return amount

        return 0.0

    def _spawn_resources(self) -> None:
        """Spawn new resources based on spawn rates."""
        # Accumulate spawn amounts
        self._food_spawn_accumulator += self.config.food_spawn_rate
        self._water_spawn_accumulator += self.config.water_spawn_rate

        # Spawn when accumulated >= 1
        while self._food_spawn_accumulator >= 1.0:
            self.spawn_resource(ResourceType.FOOD)
            self._food_spawn_accumulator -= 1.0

        while self._water_spawn_accumulator >= 1.0:
            self.spawn_resource(ResourceType.WATER)
            self._water_spawn_accumulator -= 1.0

