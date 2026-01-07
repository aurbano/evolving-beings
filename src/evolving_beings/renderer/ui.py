"""UI widgets for the simulation renderer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import pygame


class SimulationMode(Enum):
    """Simulation state modes."""

    SETUP = auto()
    RUNNING = auto()
    PAUSED = auto()


class BrushType(Enum):
    """Brush types for painting in setup mode."""

    NONE = auto()
    FOOD = auto()
    BEING = auto()
    ERASE = auto()


@dataclass
class UIColors:
    """Color scheme for UI elements."""

    # Background
    bg: tuple[int, int, int] = (30, 32, 40)
    bg_hover: tuple[int, int, int] = (45, 48, 58)
    bg_active: tuple[int, int, int] = (55, 58, 70)

    # Accents
    accent: tuple[int, int, int] = (100, 180, 255)
    accent_dim: tuple[int, int, int] = (60, 100, 140)

    # Text
    text: tuple[int, int, int] = (220, 225, 235)
    text_dim: tuple[int, int, int] = (140, 145, 155)

    # Slider
    slider_track: tuple[int, int, int] = (50, 52, 62)
    slider_fill: tuple[int, int, int] = (80, 160, 220)
    slider_knob: tuple[int, int, int] = (240, 245, 255)

    # Chart colors
    chart_population: tuple[int, int, int] = (80, 200, 220)
    chart_energy: tuple[int, int, int] = (255, 200, 80)
    chart_food: tuple[int, int, int] = (120, 200, 100)
    chart_water: tuple[int, int, int] = (100, 160, 255)
    chart_deaths: tuple[int, int, int] = (255, 100, 100)
    chart_bg: tuple[int, int, int] = (25, 27, 35)


UI_COLORS = UIColors()


class Sparkline:
    """A mini line chart for displaying time-series data."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int] = UI_COLORS.chart_population,
        bg_color: tuple[int, int, int] = UI_COLORS.chart_bg,
    ):
        """
        Initialize a sparkline chart.

        Args:
            x: X position
            y: Y position
            width: Chart width
            height: Chart height
            color: Line color
            bg_color: Background color
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.bg_color = bg_color

    def render(
        self,
        surface: pygame.Surface,
        data: Sequence[float | int],
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        """
        Render the sparkline chart.

        Args:
            surface: Surface to render on
            data: Sequence of values to plot
            min_val: Optional minimum value for scaling (auto if None)
            max_val: Optional maximum value for scaling (auto if None)
        """
        # Draw background
        pygame.draw.rect(surface, self.bg_color, self.rect, border_radius=3)

        if len(data) < 2:
            return

        # Calculate value range
        data_min = min(data) if min_val is None else min_val
        data_max = max(data) if max_val is None else max_val
        value_range = data_max - data_min if data_max > data_min else 1.0

        # Calculate points
        padding = 2
        chart_width = self.rect.width - padding * 2
        chart_height = self.rect.height - padding * 2

        points: list[tuple[int, int]] = []
        for i, value in enumerate(data):
            x = self.rect.x + padding + int(i * chart_width / (len(data) - 1))
            normalized = (value - data_min) / value_range
            y = self.rect.y + padding + int((1 - normalized) * chart_height)
            points.append((x, y))

        # Draw line
        if len(points) >= 2:
            pygame.draw.lines(surface, self.color, False, points, 2)


class Slider:
    """An interactive slider control."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        min_value: float,
        max_value: float,
        value: float,
        label: str = "",
        on_change: Callable[[float], None] | None = None,
        show_int: bool = False,
    ):
        """
        Initialize a slider.

        Args:
            x: X position
            y: Y position
            width: Slider width
            height: Slider height
            min_value: Minimum value
            max_value: Maximum value
            value: Initial value
            label: Label text
            on_change: Callback when value changes
            show_int: Display value as integer
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.min_value = min_value
        self.max_value = max_value
        self._value = value
        self.label = label
        self.on_change = on_change
        self.show_int = show_int
        self.dragging = False
        self.hovered = False
        self.enabled = True

        # Visual properties
        self.track_height = 4
        self.knob_radius = 6

    @property
    def value(self) -> float:
        """Get current value."""
        return self._value

    @value.setter
    def value(self, val: float) -> None:
        """Set value and trigger callback."""
        old_value = self._value
        self._value = max(self.min_value, min(self.max_value, val))
        if old_value != self._value and self.on_change:
            self.on_change(self._value)

    def _get_knob_x(self) -> int:
        """Get the X position of the knob."""
        ratio = (self._value - self.min_value) / (self.max_value - self.min_value)
        return int(self.rect.x + self.knob_radius + ratio * (self.rect.width - self.knob_radius * 2))

    def _value_from_x(self, x: int) -> float:
        """Convert X position to value."""
        ratio = (x - self.rect.x - self.knob_radius) / (self.rect.width - self.knob_radius * 2)
        ratio = max(0, min(1, ratio))
        return self.min_value + ratio * (self.max_value - self.min_value)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame events.

        Returns:
            True if event was consumed
        """
        if not self.enabled:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.value = self._value_from_x(event.pos[0])
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
            if self.dragging:
                self.value = self._value_from_x(event.pos[0])
                return True

        return False

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Render the slider."""
        # Determine colors based on enabled state
        if self.enabled:
            track_color = UI_COLORS.slider_track
            fill_color = UI_COLORS.slider_fill
            knob_color = UI_COLORS.slider_knob if self.hovered or self.dragging else UI_COLORS.accent
            label_color = UI_COLORS.text_dim
            value_color = UI_COLORS.text
        else:
            track_color = (40, 42, 50)
            fill_color = (60, 70, 85)
            knob_color = (80, 85, 95)
            label_color = (80, 85, 95)
            value_color = (100, 105, 115)

        # Track
        track_y = self.rect.y + self.rect.height // 2 - self.track_height // 2
        track_rect = pygame.Rect(
            self.rect.x + self.knob_radius,
            track_y,
            self.rect.width - self.knob_radius * 2,
            self.track_height,
        )
        pygame.draw.rect(surface, track_color, track_rect, border_radius=2)

        # Filled portion
        knob_x = self._get_knob_x()
        fill_rect = pygame.Rect(
            track_rect.x,
            track_rect.y,
            knob_x - track_rect.x,
            self.track_height,
        )
        pygame.draw.rect(surface, fill_color, fill_rect, border_radius=2)

        # Knob
        pygame.draw.circle(
            surface,
            knob_color,
            (knob_x, self.rect.y + self.rect.height // 2),
            self.knob_radius,
        )

        # Label and value on the same line above slider
        if self.label:
            label_surface = font.render(self.label, True, label_color)
            surface.blit(label_surface, (self.rect.x, self.rect.y - 15))

        if self.show_int:
            value_text = f"{int(self._value)}"
        else:
            value_text = f"{self._value:.2f}"
        value_surface = font.render(value_text, True, value_color)
        surface.blit(
            value_surface,
            (self.rect.right - value_surface.get_width(), self.rect.y - 15),
        )


class Button:
    """A clickable button."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        on_click: Callable[[], None] | None = None,
        toggle: bool = False,
        active: bool = False,
    ):
        """
        Initialize a button.

        Args:
            x: X position
            y: Y position
            width: Button width
            height: Button height
            text: Button text
            on_click: Callback when clicked
            toggle: Whether this is a toggle button
            active: Initial active state (for toggle buttons)
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.on_click = on_click
        self.toggle = toggle
        self.active = active
        self.hovered = False
        self.pressed = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame events.

        Returns:
            True if event was consumed
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed and self.rect.collidepoint(event.pos):
                if self.toggle:
                    self.active = not self.active
                if self.on_click:
                    self.on_click()
                self.pressed = False
                return True
            self.pressed = False

        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)

        return False

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Render the button."""
        # Determine background color
        if self.active:
            bg_color = UI_COLORS.accent
        elif self.pressed:
            bg_color = UI_COLORS.bg_active
        elif self.hovered:
            bg_color = UI_COLORS.bg_hover
        else:
            bg_color = UI_COLORS.bg

        # Draw background
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=4)

        # Draw border
        border_color = UI_COLORS.accent if self.active else UI_COLORS.accent_dim
        pygame.draw.rect(surface, border_color, self.rect, width=1, border_radius=4)

        # Draw text
        text_color = UI_COLORS.bg if self.active else UI_COLORS.text
        text_surface = font.render(self.text, True, text_color)
        text_x = self.rect.x + (self.rect.width - text_surface.get_width()) // 2
        text_y = self.rect.y + (self.rect.height - text_surface.get_height()) // 2
        surface.blit(text_surface, (text_x, text_y))


class SpeedSelector:
    """A button group for selecting simulation speed."""

    SPEEDS = [0.1, 0.5, 1.0, 2.0, 4.0]
    LABELS = ["0.1x", "0.5x", "1x", "2x", "4x"]

    def __init__(
        self,
        x: int,
        y: int,
        button_width: int = 40,
        button_height: int = 24,
        on_change: Callable[[float], None] | None = None,
    ):
        """
        Initialize speed selector.

        Args:
            x: X position
            y: Y position
            button_width: Width of each button
            button_height: Height of each button
            on_change: Callback when speed changes
        """
        self.buttons: list[Button] = []
        self.on_change = on_change
        self._speed = 1.0

        for i, (speed, label) in enumerate(zip(self.SPEEDS, self.LABELS)):
            btn = Button(
                x + i * (button_width + 4),
                y,
                button_width,
                button_height,
                label,
                on_click=lambda s=speed: self._set_speed(s),
                toggle=True,
                active=(speed == 1.0),
            )
            self.buttons.append(btn)

    @property
    def speed(self) -> float:
        """Get current speed multiplier."""
        return self._speed

    def _set_speed(self, speed: float) -> None:
        """Set speed and update button states."""
        self._speed = speed
        for btn, spd in zip(self.buttons, self.SPEEDS):
            btn.active = (spd == speed)
        if self.on_change:
            self.on_change(speed)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events."""
        for btn in self.buttons:
            if btn.handle_event(event):
                return True
        return False

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Render speed selector."""
        for btn in self.buttons:
            btn.render(surface, font)

