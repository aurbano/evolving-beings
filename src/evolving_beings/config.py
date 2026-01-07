"""Centralized configuration for the simulation."""

from dataclasses import dataclass


@dataclass
class WorldConfig:
    """Configuration for the world simulation."""

    width: int = 800
    height: int = 800
    initial_population: int = 100
    initial_food: int = 200
    initial_water: int = 200
    food_spawn_rate: float = 0.5  # resources per tick
    water_spawn_rate: float = 0.5
    # Clustering: probability (0-1) that new resources spawn near existing ones
    resource_clustering: float = 0.7
    # Radius within which clustered resources spawn
    cluster_radius: float = 50.0
    # Random seed for reproducibility (None = random seed)
    seed: int | None = None
    # Terrain generation (heightmap-based)
    terrain_style: str = "island"  # "island" or "river_valley"
    water_level: float = 0.35  # Height threshold below which is water
    mountain_level: float = 0.65  # Height threshold above which is mountain/obstacle
    # Legacy terrain options (deprecated, kept for compatibility)
    num_obstacles: int = 5
    num_water_bodies: int = 3
    generate_river: bool = True


@dataclass
class BeingConfig:
    """Configuration for being attributes."""

    # Energy dynamics
    energy_loss_idle: float = 0.001
    energy_loss_moving: float = 0.003
    food_to_energy: float = 0.3
    water_to_energy: float = 0.3

    # Movement
    base_speed: float = 2.0
    turn_speed: float = 0.15  # radians per tick

    # Vision
    vision_range: float = 80.0
    vision_angle: float = 120.0  # degrees

    # Needs thresholds
    hunger_threshold: float = 0.4  # below this, being is "hungry"
    thirst_threshold: float = 0.4


@dataclass
class RendererConfig:
    """Configuration for the Pygame renderer."""

    window_width: int = 1080
    window_height: int = 800
    sidebar_width: int = 280
    target_fps: int = 60
    being_radius: int = 4
    resource_radius: int = 5
    show_vision: bool = False  # debug: show vision cones


@dataclass
class Config:
    """Main configuration container."""

    world: WorldConfig
    being: BeingConfig
    renderer: RendererConfig

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls(
            world=WorldConfig(),
            being=BeingConfig(),
            renderer=RendererConfig(),
        )

