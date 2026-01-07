"""Spatial hash grid for efficient neighbor queries."""

from collections import defaultdict
from typing import TYPE_CHECKING, Iterator, Protocol, TypeVar

if TYPE_CHECKING:
    pass


class HasPosition(Protocol):
    """Protocol for entities that have a position."""

    x: float
    y: float


T = TypeVar("T", bound=HasPosition)


class SpatialGrid[T]:
    """
    Spatial hash grid for O(1) neighbor lookups.

    Divides the world into cells and tracks which entities are in each cell.
    Finding neighbors only requires checking nearby cells rather than all entities.
    """

    def __init__(self, width: float, height: float, cell_size: float = 32.0):
        """
        Initialize the spatial grid.

        Args:
            width: World width in units
            height: World height in units
            cell_size: Size of each grid cell (larger = fewer cells, more entities per cell)
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(width / cell_size) + 1
        self.rows = int(height / cell_size) + 1
        self.cells: dict[tuple[int, int], set[T]] = defaultdict(set)
        self._entity_cells: dict[int, tuple[int, int]] = {}  # entity id -> cell

    def _get_cell(self, x: float, y: float) -> tuple[int, int]:
        """Get the cell coordinates for a position."""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        # Clamp to valid range
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return (col, row)

    def insert(self, entity: T) -> None:
        """Add an entity to the grid."""
        cell = self._get_cell(entity.x, entity.y)
        self.cells[cell].add(entity)
        self._entity_cells[id(entity)] = cell

    def remove(self, entity: T) -> None:
        """Remove an entity from the grid."""
        entity_id = id(entity)
        if entity_id in self._entity_cells:
            cell = self._entity_cells[entity_id]
            self.cells[cell].discard(entity)
            del self._entity_cells[entity_id]

    def update(self, entity: T) -> None:
        """Update an entity's position in the grid."""
        entity_id = id(entity)
        new_cell = self._get_cell(entity.x, entity.y)

        if entity_id in self._entity_cells:
            old_cell = self._entity_cells[entity_id]
            if old_cell != new_cell:
                self.cells[old_cell].discard(entity)
                self.cells[new_cell].add(entity)
                self._entity_cells[entity_id] = new_cell
        else:
            # Entity wasn't in grid, insert it
            self.insert(entity)

    def get_nearby(self, x: float, y: float, radius: float) -> Iterator[T]:
        """
        Get all entities within radius of the given position.

        Args:
            x: Center x coordinate
            y: Center y coordinate
            radius: Search radius

        Yields:
            Entities within the radius
        """
        # Calculate which cells could contain entities within radius
        min_col = int((x - radius) / self.cell_size)
        max_col = int((x + radius) / self.cell_size)
        min_row = int((y - radius) / self.cell_size)
        max_row = int((y + radius) / self.cell_size)

        # Clamp to valid range
        min_col = max(0, min_col)
        max_col = min(self.cols - 1, max_col)
        min_row = max(0, min_row)
        max_row = min(self.rows - 1, max_row)

        radius_sq = radius * radius

        # Check all potentially relevant cells
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                for entity in self.cells[(col, row)]:
                    # Actual distance check
                    dx = entity.x - x
                    dy = entity.y - y
                    if dx * dx + dy * dy <= radius_sq:
                        yield entity

    def get_at(self, x: float, y: float, radius: float = 1.0) -> T | None:
        """Get the first entity at or very near a position."""
        for entity in self.get_nearby(x, y, radius):
            return entity
        return None

    def clear(self) -> None:
        """Remove all entities from the grid."""
        self.cells.clear()
        self._entity_cells.clear()

    def __len__(self) -> int:
        """Return the total number of entities in the grid."""
        return len(self._entity_cells)

