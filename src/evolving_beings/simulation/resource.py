"""Resource entities - food and water in the world."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ResourceType(Enum):
    """Types of resources available in the world."""

    FOOD = auto()
    WATER = auto()


@dataclass
class Resource:
    """
    A consumable resource in the world.

    Resources have a position and an amount that decreases when consumed.
    When amount reaches 0, the resource is depleted and should be removed.
    """

    x: float
    y: float
    type: ResourceType
    amount: float = 1.0
    max_amount: float = 1.0

    # Unique identifier
    _id_counter: int = field(default=0, repr=False, init=False)
    id: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        """Assign unique ID after initialization."""
        Resource._id_counter += 1
        self.id = Resource._id_counter

    def __hash__(self) -> int:
        """Hash based on unique ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on unique ID."""
        if not isinstance(other, Resource):
            return NotImplemented
        return self.id == other.id

    @property
    def is_depleted(self) -> bool:
        """Check if the resource has been fully consumed."""
        return self.amount <= 0

    @property
    def fullness(self) -> float:
        """Get the resource fullness as a ratio (0-1)."""
        return self.amount / self.max_amount if self.max_amount > 0 else 0

    def consume(self, amount: float) -> float:
        """
        Consume some of this resource.

        Args:
            amount: Maximum amount to consume

        Returns:
            The actual amount consumed (may be less if resource is nearly depleted)
        """
        consumed = min(amount, self.amount)
        self.amount -= consumed
        return consumed

    def replenish(self, amount: float) -> None:
        """Add to this resource (e.g., for regenerating resources)."""
        self.amount = min(self.max_amount, self.amount + amount)

