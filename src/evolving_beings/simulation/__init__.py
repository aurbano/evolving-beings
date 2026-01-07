"""Simulation module - pure logic, no rendering."""

from .being import Being
from .resource import Resource, ResourceType
from .spatial import SpatialGrid
from .terrain import Obstacle, Terrain, WaterBody
from .world import StatsHistory, World, WorldStats

__all__ = [
    "Being",
    "Obstacle",
    "Resource",
    "ResourceType",
    "SpatialGrid",
    "StatsHistory",
    "Terrain",
    "WaterBody",
    "World",
    "WorldStats",
]

