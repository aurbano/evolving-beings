"""Being entity - a living creature in the simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .resource import Resource


class Action(Enum):
    """Possible actions a being can take."""

    IDLE = auto()
    MOVE_FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    EAT = auto()
    DRINK = auto()


@dataclass
class BeingState:
    """Observable state of a being for decision making."""

    energy: float
    food: float
    water: float
    nearby_food: list[tuple[float, float]]  # (distance, angle) pairs
    nearby_water: list[tuple[float, float]]
    nearby_beings: list[tuple[float, float]]


@dataclass
class Being:
    """
    A living being in the simulation.

    Beings have:
    - Position and orientation in 2D space
    - Internal resources (food, water) that convert to energy
    - Energy that depletes over time (death when empty)
    - Vision to detect nearby entities
    - Simple behavior to seek resources and survive
    """

    # Position and movement
    x: float
    y: float
    angle: float = field(default=-1.0)  # Will be set in __post_init__ if not provided
    speed: float = 0.0

    # Internal state
    energy: float = 1.0
    food: float = 0.5
    water: float = 0.5

    # Configuration (from BeingConfig)
    energy_loss_idle: float = 0.001
    energy_loss_moving: float = 0.003
    food_to_energy: float = 0.3
    water_to_energy: float = 0.3
    base_speed: float = 2.0
    turn_speed: float = 0.15
    vision_range: float = 80.0
    vision_angle: float = 120.0  # degrees

    # Behavior thresholds
    hunger_threshold: float = 0.4
    thirst_threshold: float = 0.4

    # Random seed for deterministic behavior
    rng_seed: int | None = None

    # Unique identifier
    _id_counter: int = field(default=0, repr=False, init=False)
    id: int = field(default=-1, init=False)
    rng: random.Random = field(default=None, repr=False, init=False)  # type: ignore

    def __post_init__(self) -> None:
        """Assign unique ID and initialize RNG after initialization."""
        Being._id_counter += 1
        self.id = Being._id_counter

        # Initialize RNG for deterministic behavior
        if self.rng_seed is not None:
            self.rng = random.Random(self.rng_seed)
        else:
            self.rng = random.Random()

        # Set random initial angle if not provided
        if self.angle < 0:
            self.angle = self.rng.uniform(0, 2 * math.pi)

    def __hash__(self) -> int:
        """Hash based on unique ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on unique ID."""
        if not isinstance(other, Being):
            return NotImplemented
        return self.id == other.id

    @property
    def is_alive(self) -> bool:
        """Check if the being is still alive."""
        return self.energy > 0

    @property
    def is_hungry(self) -> bool:
        """Check if the being needs food."""
        return self.food < self.hunger_threshold

    @property
    def is_thirsty(self) -> bool:
        """Check if the being needs water."""
        return self.water < self.thirst_threshold

    @property
    def direction(self) -> tuple[float, float]:
        """Get the unit direction vector."""
        return (math.cos(self.angle), math.sin(self.angle))

    def consume(self, resource: Resource) -> float:
        """
        Consume a resource and gain its value.

        Returns the amount consumed.
        """
        from .resource import ResourceType

        amount = resource.consume(0.2)  # consume up to 0.2 per tick
        if resource.type == ResourceType.FOOD:
            self.food = min(1.0, self.food + amount)
        else:
            self.water = min(1.0, self.water + amount)
        return amount

    def update(self, action: Action, world_width: float, world_height: float) -> None:
        """
        Update the being's state based on the chosen action.

        Args:
            action: The action to perform
            world_width: Width of the world (for boundary wrapping)
            world_height: Height of the world (for boundary wrapping)
        """
        # Process action
        if action == Action.TURN_LEFT:
            self.angle += self.turn_speed
        elif action == Action.TURN_RIGHT:
            self.angle -= self.turn_speed
        elif action == Action.MOVE_FORWARD:
            self.speed = self.base_speed
        elif action == Action.IDLE:
            self.speed = 0.0
        # EAT and DRINK are handled externally when near resources

        # Normalize angle to [0, 2π]
        self.angle = self.angle % (2 * math.pi)

        # Move
        if self.speed > 0:
            dx, dy = self.direction
            self.x += dx * self.speed
            self.y += dy * self.speed

            # Wrap around world boundaries
            self.x = self.x % world_width
            self.y = self.y % world_height

        # Energy consumption
        energy_loss = self.energy_loss_moving if self.speed > 0 else self.energy_loss_idle
        self.energy -= energy_loss

        # Convert internal resources to energy when low
        if self.energy < 0.8:
            if self.food > 0.1:
                transfer = min(0.01, self.food)
                self.food -= transfer
                self.energy += transfer * self.food_to_energy

            if self.water > 0.1:
                transfer = min(0.01, self.water)
                self.water -= transfer
                self.energy += transfer * self.water_to_energy

        # Clamp energy
        self.energy = max(0.0, min(1.0, self.energy))

    def get_angle_to(self, target_x: float, target_y: float) -> float:
        """Get the angle from this being to a target position."""
        dx = target_x - self.x
        dy = target_y - self.y
        return math.atan2(dy, dx)

    def get_distance_to(self, target_x: float, target_y: float) -> float:
        """Get the distance from this being to a target position."""
        dx = target_x - self.x
        dy = target_y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def angle_diff(self, target_angle: float) -> float:
        """
        Get the signed angular difference to a target angle.

        Positive = target is to the left, negative = target is to the right.
        """
        diff = target_angle - self.angle
        # Normalize to [-π, π]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def can_see(self, target_x: float, target_y: float) -> bool:
        """Check if a position is within this being's field of view."""
        distance = self.get_distance_to(target_x, target_y)
        if distance > self.vision_range:
            return False

        target_angle = self.get_angle_to(target_x, target_y)
        angle_diff = abs(self.angle_diff(target_angle))
        half_fov = math.radians(self.vision_angle / 2)

        return angle_diff <= half_fov

