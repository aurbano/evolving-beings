"""Pygame-CE renderer for visualizing the simulation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from ..config import RendererConfig
from ..simulation.resource import ResourceType
from . import colors
from .ui import (
    UI_COLORS,
    BrushType,
    Button,
    SimulationMode,
    Slider,
    Sparkline,
    SpeedSelector,
)

if TYPE_CHECKING:
    from ..simulation.world import World


class PygameRenderer:
    """
    Pygame-based renderer for the evolving beings simulation.

    Renders:
    - World background
    - Beings (colored by health state)
    - Resources (food and water)
    - Sidebar with statistics, charts, and controls
    """

    def __init__(self, config: RendererConfig):
        """
        Initialize the renderer.

        Args:
            config: Renderer configuration
        """
        self.config = config
        self.window_width = config.window_width
        self.window_height = config.window_height
        self.sidebar_width = config.sidebar_width
        self.world_width = config.window_width - config.sidebar_width
        self.world_height = config.window_height

        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Evolving Beings")

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)

        # Pre-render some surfaces
        self._world_surface = pygame.Surface((self.world_width, self.world_height))
        self._sidebar_surface = pygame.Surface((self.sidebar_width, self.window_height))

        # FPS tracking
        self._fps_history: list[float] = []

        # Mode and state
        self.mode = SimulationMode.SETUP
        self.speed_multiplier = 1.0
        self.brush_type = BrushType.NONE
        self.brush_size = 1

        # Step control for manual stepping
        self._pending_steps = 0

        # UI Elements - will be initialized in _init_ui
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI elements."""
        padding = 15
        chart_width = self.sidebar_width - padding * 2 - 80  # Leave room for label
        chart_height = 30

        # Mode buttons
        btn_width = 55
        btn_height = 26
        btn_y = 10
        self.btn_setup = Button(
            padding, btn_y, btn_width, btn_height, "Setup",
            on_click=lambda: self._set_mode(SimulationMode.SETUP),
            toggle=True, active=True
        )
        self.btn_run = Button(
            padding + btn_width + 4, btn_y, btn_width, btn_height, "Run",
            on_click=lambda: self._set_mode(SimulationMode.RUNNING),
            toggle=True, active=False
        )
        self.btn_pause = Button(
            padding + (btn_width + 4) * 2, btn_y, btn_width, btn_height, "Pause",
            on_click=lambda: self._set_mode(SimulationMode.PAUSED),
            toggle=True, active=False
        )
        self.btn_step = Button(
            padding + (btn_width + 4) * 3, btn_y, btn_width, btn_height, "Step",
            on_click=self._on_step_click,
            toggle=False, active=False
        )
        self.mode_buttons = [self.btn_setup, self.btn_run, self.btn_pause, self.btn_step]

        # Sparkline charts - positions will be set during render based on mode
        self.chart_population = Sparkline(
            padding, 0, chart_width, chart_height,
            color=UI_COLORS.chart_population
        )
        self.chart_energy = Sparkline(
            padding, 0, chart_width, chart_height,
            color=UI_COLORS.chart_energy
        )
        self.chart_food = Sparkline(
            padding, 0, chart_width, chart_height,
            color=UI_COLORS.chart_food
        )
        self.chart_water = Sparkline(
            padding, 0, chart_width, chart_height,
            color=UI_COLORS.chart_water
        )

        # Control sliders
        slider_width = self.sidebar_width - padding * 2
        self.slider_food_rate = Slider(
            padding, 0, slider_width, 20,
            min_value=0.0, max_value=3.0, value=0.5,
            label="Food Rate"
        )

        # Clustering slider
        self.slider_clustering = Slider(
            padding, 0, slider_width, 20,
            min_value=0.0, max_value=1.0, value=0.7,
            label="Clustering"
        )

        # Speed selector
        self.speed_selector = SpeedSelector(
            padding, 0, button_width=50, button_height=24,
            on_change=self._on_speed_change
        )

        # Brush buttons (for setup mode)
        self.btn_food_brush = Button(
            padding, 0, 60, 26, "Food",
            on_click=lambda: self._set_brush(BrushType.FOOD),
            toggle=True
        )
        self.btn_being_brush = Button(
            padding + 65, 0, 60, 26, "Being",
            on_click=lambda: self._set_brush(BrushType.BEING),
            toggle=True
        )
        self.btn_erase_brush = Button(
            padding + 130, 0, 60, 26, "Erase",
            on_click=lambda: self._set_brush(BrushType.ERASE),
            toggle=True
        )
        self.brush_buttons = [self.btn_food_brush, self.btn_being_brush, self.btn_erase_brush]

        # Brush size slider
        self.slider_brush_size = Slider(
            padding, 0, slider_width, 20,
            min_value=1, max_value=10, value=3,
            label="Brush Size", show_int=True
        )

        # Setup mode sliders - callbacks will be set when world is available
        self.slider_initial_pop = Slider(
            padding, 0, slider_width, 20,
            min_value=10, max_value=500, value=100,
            label="Population", show_int=True
        )
        self.slider_initial_food = Slider(
            padding, 0, slider_width, 20,
            min_value=0, max_value=500, value=200,
            label="Food", show_int=True
        )

        # Reference to world for slider callbacks (set by main.py)
        self._world: World | None = None

    def set_world(self, world: World) -> None:
        """Set the world reference for slider callbacks."""
        self._world = world

    def _reinitialize_world(self) -> None:
        """Reinitialize world based on current slider values."""
        if self._world is None or self.mode != SimulationMode.SETUP:
            return

        from ..simulation.resource import ResourceType

        target_pop = int(self.slider_initial_pop.value)
        target_food = int(self.slider_initial_food.value)

        # Adjust beings
        while len(self._world.beings) > target_pop:
            being = self._world.beings.pop()
            self._world.being_grid.remove(being)
        while len(self._world.beings) < target_pop:
            self._world.spawn_being()

        # Adjust food
        food_resources = [r for r in self._world.resources if r.type == ResourceType.FOOD]
        while len(food_resources) > target_food:
            resource = food_resources.pop()
            self._world.resources.remove(resource)
            self._world.resource_grid.remove(resource)
        while len(food_resources) < target_food:
            self._world.spawn_resource(ResourceType.FOOD)
            food_resources.append(self._world.resources[-1])

        # Update stats
        self._world.stats.beings_alive = len(self._world.beings)
        self._world.stats.food_count = target_food

    def _set_mode(self, mode: SimulationMode) -> None:
        """Set simulation mode and update button states."""
        self.mode = mode
        self.btn_setup.active = (mode == SimulationMode.SETUP)
        self.btn_run.active = (mode == SimulationMode.RUNNING)
        self.btn_pause.active = (mode == SimulationMode.PAUSED)

        # Clear brush when leaving setup mode
        if mode != SimulationMode.SETUP:
            self._set_brush(BrushType.NONE)

    def _set_brush(self, brush: BrushType) -> None:
        """Set brush type and update button states."""
        # Toggle off if same brush selected
        if self.brush_type == brush:
            self.brush_type = BrushType.NONE
        else:
            self.brush_type = brush

        self.btn_food_brush.active = (self.brush_type == BrushType.FOOD)
        self.btn_being_brush.active = (self.brush_type == BrushType.BEING)
        self.btn_erase_brush.active = (self.brush_type == BrushType.ERASE)

    def _on_speed_change(self, speed: float) -> None:
        """Handle speed change."""
        self.speed_multiplier = speed

    def _on_step_click(self) -> None:
        """Handle step button click - advance one tick."""
        if self.mode == SimulationMode.PAUSED:
            self._pending_steps += 1

    def handle_events(self, world: World | None = None) -> bool:
        """
        Handle Pygame events.

        Args:
            world: The world to modify (for brush tools)

        Returns:
            False if the window should close, True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    # Toggle pause/run
                    if self.mode == SimulationMode.RUNNING:
                        self._set_mode(SimulationMode.PAUSED)
                    elif self.mode == SimulationMode.PAUSED:
                        self._set_mode(SimulationMode.RUNNING)

            # Handle UI events
            consumed = False
            for btn in self.mode_buttons:
                if btn.handle_event(event):
                    consumed = True
                    break

            if not consumed:
                if self.slider_food_rate.handle_event(event):
                    if world:
                        world.config.food_spawn_rate = self.slider_food_rate.value
                    consumed = True
                elif self.slider_clustering.handle_event(event):
                    if world:
                        world.config.resource_clustering = self.slider_clustering.value
                    consumed = True
                elif self.speed_selector.handle_event(event):
                    consumed = True

            # Setup mode controls
            if not consumed and self.mode == SimulationMode.SETUP:
                for btn in self.brush_buttons:
                    if btn.handle_event(event):
                        consumed = True
                        break
                if not consumed:
                    if self.slider_brush_size.handle_event(event):
                        self.brush_size = int(self.slider_brush_size.value)
                        consumed = True
                    elif self.slider_initial_pop.handle_event(event):
                        self._reinitialize_world()
                        consumed = True
                    elif self.slider_initial_food.handle_event(event):
                        self._reinitialize_world()
                        consumed = True

            # Handle brush painting on world
            if not consumed and world and self.mode == SimulationMode.SETUP:
                if self.brush_type != BrushType.NONE:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._apply_brush(world, event.pos)
                    elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                        self._apply_brush(world, event.pos)

        return True

    def _apply_brush(self, world: World, screen_pos: tuple[int, int]) -> None:
        """Apply brush at screen position."""
        # Check if click is in world area (right of sidebar)
        if screen_pos[0] < self.sidebar_width:
            return

        # Convert screen to world coordinates
        scale_x = world.width / self.world_width
        scale_y = world.height / self.world_height
        world_x = (screen_pos[0] - self.sidebar_width) * scale_x
        world_y = screen_pos[1] * scale_y

        if self.brush_type == BrushType.FOOD:
            for _ in range(self.brush_size):
                offset_x = (hash(str(world_x) + str(_)) % 20 - 10)
                offset_y = (hash(str(world_y) + str(_) + "y") % 20 - 10)
                x = max(0, min(world.width, world_x + offset_x))
                y = max(0, min(world.height, world_y + offset_y))
                world.spawn_resource(ResourceType.FOOD, x, y, use_clustering=False)
        elif self.brush_type == BrushType.BEING:
            # Spawn beings at click location (fewer than brush_size since beings are more impactful)
            for _ in range(max(1, self.brush_size // 2)):
                offset_x = (hash(str(world_x) + str(_)) % 30 - 15)
                offset_y = (hash(str(world_y) + str(_) + "y") % 30 - 15)
                x = max(0, min(world.width, world_x + offset_x))
                y = max(0, min(world.height, world_y + offset_y))
                # Only spawn if not in obstacle or water
                if not world.terrain.is_blocked(x, y) and not world.terrain.is_in_water(x, y):
                    world.spawn_being(x, y)
        elif self.brush_type == BrushType.ERASE:
            # Remove resources near click
            erase_radius = 15 * scale_x * self.brush_size
            to_remove = []
            for resource in world.resources:
                dx = resource.x - world_x
                dy = resource.y - world_y
                if dx * dx + dy * dy < erase_radius * erase_radius:
                    to_remove.append(resource)
            for resource in to_remove[:self.brush_size * 2]:  # Limit erasure per click
                world.resources.remove(resource)
                world.resource_grid.remove(resource)

    def should_step(self) -> bool:
        """Check if simulation should step this frame."""
        if self.mode == SimulationMode.RUNNING:
            return True
        # Check for manual stepping when paused
        if self._pending_steps > 0:
            self._pending_steps -= 1
            return True
        return False

    def get_steps_per_frame(self) -> int:
        """Get number of simulation steps to run per frame."""
        if self.speed_multiplier <= 1.0:
            return 1
        return int(self.speed_multiplier)

    def render(self, world: World) -> None:
        """
        Render the current state of the world.

        Args:
            world: The simulation world to render
        """
        # Clear screen
        self.screen.fill(colors.BG_DARK)

        # Render world area
        self._render_world(world)

        # Render sidebar
        self._render_sidebar(world)

        # Blit surfaces to screen
        self.screen.blit(self._world_surface, (self.sidebar_width, 0))
        self.screen.blit(self._sidebar_surface, (0, 0))

        # Render brush cursor if in setup mode with brush selected (after blitting surfaces)
        if self.mode == SimulationMode.SETUP and self.brush_type != BrushType.NONE:
            self._render_brush_cursor()

        # Update display
        pygame.display.flip()

        # Track FPS
        self._fps_history.append(self.clock.get_fps())
        if len(self._fps_history) > 60:
            self._fps_history.pop(0)

    def _render_brush_cursor(self) -> None:
        """Render brush cursor overlay on world surface."""
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos[0] < self.sidebar_width:
            return

        # Determine brush color
        if self.brush_type == BrushType.FOOD:
            color = UI_COLORS.chart_food
            fill_color = (*UI_COLORS.chart_food, 40)  # Semi-transparent
        elif self.brush_type == BrushType.BEING:
            color = UI_COLORS.chart_population
            fill_color = (*UI_COLORS.chart_population, 40)
        else:
            color = (255, 100, 100)
            fill_color = (255, 100, 100, 40)

        # Calculate radius based on brush size (match actual brush effect)
        radius = 10 + self.brush_size * 4

        # Create a transparent surface for the fill
        cursor_surface = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)

        # Draw filled circle with transparency
        pygame.draw.circle(cursor_surface, fill_color, (radius + 2, radius + 2), radius)

        # Draw outline
        pygame.draw.circle(cursor_surface, color, (radius + 2, radius + 2), radius, 2)

        # Draw crosshair in center
        center = radius + 2
        pygame.draw.line(cursor_surface, color, (center - 5, center), (center + 5, center), 1)
        pygame.draw.line(cursor_surface, color, (center, center - 5), (center, center + 5), 1)

        # Blit to screen
        self.screen.blit(cursor_surface, (mouse_pos[0] - radius - 2, mouse_pos[1] - radius - 2))

    def _render_world(self, world: World) -> None:
        """Render the world background, terrain, resources, and beings."""
        # Scale factors for world -> screen coordinates
        scale_x = self.world_width / world.width
        scale_y = self.world_height / world.height

        # Render elevation-based ground if heightmap is available
        if world.terrain.heightmap is not None:
            self._render_elevation_ground(world, scale_x, scale_y)
        else:
            # Fallback to flat background
            self._world_surface.fill(colors.WORLD_BG)

            # Draw a subtle grid pattern
            grid_size = 64
            for x in range(0, self.world_width, int(grid_size * scale_x)):
                pygame.draw.line(
                    self._world_surface,
                    colors.WORLD_BG_ACCENT,
                    (x, 0),
                    (x, self.world_height),
                    1,
                )
            for y in range(0, self.world_height, int(grid_size * scale_y)):
                pygame.draw.line(
                    self._world_surface,
                    colors.WORLD_BG_ACCENT,
                    (0, y),
                    (self.world_width, y),
                    1,
                )

        # Draw terrain (water bodies and obstacles)
        self._render_terrain(world, scale_x, scale_y)

        # Draw resources
        for resource in world.resources:
            screen_x = int(resource.x * scale_x)
            screen_y = int(resource.y * scale_y)
            color = colors.get_resource_color(resource.type.name, resource.fullness)
            radius = max(2, int(self.config.resource_radius * resource.fullness))
            pygame.draw.circle(self._world_surface, color, (screen_x, screen_y), radius)

        # Draw beings
        for being in world.beings:
            screen_x = int(being.x * scale_x)
            screen_y = int(being.y * scale_y)
            color = colors.get_being_color(being.energy, being.food, being.water)

            # Draw being body
            pygame.draw.circle(
                self._world_surface, color, (screen_x, screen_y), self.config.being_radius
            )

            # Draw direction indicator
            dir_length = self.config.being_radius + 3
            end_x = screen_x + int(math.cos(being.angle) * dir_length)
            end_y = screen_y + int(math.sin(being.angle) * dir_length)
            pygame.draw.line(self._world_surface, color, (screen_x, screen_y), (end_x, end_y), 2)

            # Debug: draw vision cone
            if self.config.show_vision:
                self._draw_vision_cone(being, screen_x, screen_y, scale_x)

    def _render_elevation_ground(self, world: World, scale_x: float, scale_y: float) -> None:
        """Render ground with elevation-based color shading."""
        heightmap = world.terrain.heightmap
        if heightmap is None:
            return

        # Render at heightmap resolution for performance
        resolution = heightmap.resolution
        water_level = heightmap.water_level
        mountain_level = heightmap.mountain_level

        for gy in range(heightmap.grid_height):
            for gx in range(heightmap.grid_width):
                height = heightmap.heightmap[gy, gx]

                # Skip water areas (will be rendered by water polygons)
                if height < water_level:
                    continue

                # Skip mountain areas (will be rendered by mountain polygons)
                if height > mountain_level:
                    continue

                # Get elevation-based color
                color = colors.get_ground_color(height, water_level, mountain_level)

                # Calculate screen rectangle
                screen_x = int(gx * resolution * scale_x)
                screen_y = int(gy * resolution * scale_y)
                rect_w = max(1, int(resolution * scale_x) + 1)
                rect_h = max(1, int(resolution * scale_y) + 1)

                pygame.draw.rect(
                    self._world_surface,
                    color,
                    (screen_x, screen_y, rect_w, rect_h),
                )

    def _render_terrain(self, world: World, scale_x: float, scale_y: float) -> None:
        """Render terrain features: water bodies, mountains, and obstacles."""
        # Draw water bodies first (they go under everything else)
        for water in world.terrain.water_bodies:
            if len(water.vertices) < 3:
                continue

            # Convert world coordinates to screen coordinates
            screen_vertices = [
                (int(v[0] * scale_x), int(v[1] * scale_y))
                for v in water.vertices
            ]

            # Create a surface with alpha for water transparency
            water_surface = pygame.Surface(
                (self.world_width, self.world_height), pygame.SRCALPHA
            )

            # Draw filled polygon
            pygame.draw.polygon(
                water_surface,
                (*colors.WATER_BODY_COLOR, colors.WATER_BODY_ALPHA),
                screen_vertices,
            )

            # Draw edge with lighter color
            pygame.draw.polygon(
                water_surface,
                (*colors.WATER_BODY_EDGE, 255),
                screen_vertices,
                width=2,
            )

            self._world_surface.blit(water_surface, (0, 0))

        # Draw mountain obstacles (polygon-based from heightmap)
        for mountain in world.terrain.mountain_obstacles:
            if len(mountain.vertices) < 3:
                continue

            screen_vertices = [
                (int(v[0] * scale_x), int(v[1] * scale_y))
                for v in mountain.vertices
            ]

            # Draw filled polygon
            pygame.draw.polygon(
                self._world_surface,
                colors.MOUNTAIN_COLOR,
                screen_vertices,
            )

            # Draw highlight edge (simulate 3D)
            pygame.draw.polygon(
                self._world_surface,
                colors.MOUNTAIN_HIGHLIGHT,
                screen_vertices,
                width=2,
            )

        # Draw legacy circular obstacles (if any)
        for obstacle in world.terrain.obstacles:
            screen_x = int(obstacle.x * scale_x)
            screen_y = int(obstacle.y * scale_y)
            screen_radius = int(obstacle.radius * min(scale_x, scale_y))

            # Draw filled circle with highlight effect
            pygame.draw.circle(
                self._world_surface,
                colors.OBSTACLE_COLOR,
                (screen_x, screen_y),
                screen_radius,
            )

            # Draw highlight on top-left edge
            highlight_offset = screen_radius // 4
            pygame.draw.circle(
                self._world_surface,
                colors.OBSTACLE_HIGHLIGHT,
                (screen_x - highlight_offset, screen_y - highlight_offset),
                screen_radius // 3,
            )

            # Draw dark outline
            pygame.draw.circle(
                self._world_surface,
                (30, 25, 20),
                (screen_x, screen_y),
                screen_radius,
                width=2,
            )

    def _draw_vision_cone(
        self, being, screen_x: int, screen_y: int, scale: float
    ) -> None:
        """Draw a debug vision cone for a being."""
        vision_range = int(being.vision_range * scale)
        half_angle = math.radians(being.vision_angle / 2)

        # Create points for the cone
        left_angle = being.angle + half_angle
        right_angle = being.angle - half_angle

        left_x = screen_x + int(math.cos(left_angle) * vision_range)
        left_y = screen_y + int(math.sin(left_angle) * vision_range)
        right_x = screen_x + int(math.cos(right_angle) * vision_range)
        right_y = screen_y + int(math.sin(right_angle) * vision_range)

        # Draw cone edges
        pygame.draw.line(
            self._world_surface, (100, 100, 100), (screen_x, screen_y), (left_x, left_y), 1
        )
        pygame.draw.line(
            self._world_surface, (100, 100, 100), (screen_x, screen_y), (right_x, right_y), 1
        )

    def _render_sidebar(self, world: World) -> None:
        """Render the sidebar with statistics and controls."""
        self._sidebar_surface.fill(colors.BG_SIDEBAR)

        # Draw divider line on right edge
        pygame.draw.line(
            self._sidebar_surface,
            colors.DIVIDER,
            (self.sidebar_width - 1, 0),
            (self.sidebar_width - 1, self.window_height),
            2,
        )

        padding = 12
        y = 10

        # Mode buttons
        for btn in self.mode_buttons:
            btn.render(self._sidebar_surface, self.font_small)
        y += 36

        # Status line: Tick and FPS
        avg_fps = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0
        status_text = f"Tick: {world.stats.tick:,}   FPS: {avg_fps:.0f}"
        status_surface = self.font_small.render(status_text, True, colors.TEXT_SECONDARY)
        self._sidebar_surface.blit(status_surface, (padding, y))
        y += 18

        # Seed display
        seed_text = f"Seed: {world.seed}"
        seed_surface = self.font_small.render(seed_text, True, colors.TEXT_SECONDARY)
        self._sidebar_surface.blit(seed_surface, (padding, y))
        y += 18

        # Thin divider
        pygame.draw.line(
            self._sidebar_surface, colors.DIVIDER,
            (padding, y), (self.sidebar_width - padding, y)
        )
        y += 8

        # === INITIAL CONFIG SECTION ===
        y = self._render_section_header("INITIAL", y, padding)

        # Enable/disable based on mode
        is_setup = self.mode == SimulationMode.SETUP
        self.slider_initial_pop.enabled = is_setup
        self.slider_initial_food.enabled = is_setup

        self.slider_initial_pop.rect.y = y
        self.slider_initial_pop.render(self._sidebar_surface, self.font_small)
        y += 35

        self.slider_initial_food.rect.y = y
        self.slider_initial_food.render(self._sidebar_surface, self.font_small)
        y += 30

        # Thin divider
        pygame.draw.line(
            self._sidebar_surface, colors.DIVIDER,
            (padding, y), (self.sidebar_width - padding, y)
        )
        y += 8

        # === SPAWN RATES SECTION ===
        y = self._render_section_header("SPAWN RATES", y, padding)

        self.slider_food_rate.rect.y = y
        self.slider_food_rate.render(self._sidebar_surface, self.font_small)
        y += 35

        self.slider_clustering.rect.y = y
        self.slider_clustering.render(self._sidebar_surface, self.font_small)
        y += 30

        # Thin divider
        pygame.draw.line(
            self._sidebar_surface, colors.DIVIDER,
            (padding, y), (self.sidebar_width - padding, y)
        )
        y += 8

        # === STATS SECTION (only in run/paused mode) ===
        if self.mode != SimulationMode.SETUP:
            y = self._render_stats_section(world, y, padding)
        else:
            y = self._render_brush_section(world, y, padding)

        # === SPEED SECTION ===
        pygame.draw.line(
            self._sidebar_surface, colors.DIVIDER,
            (padding, y), (self.sidebar_width - padding, y)
        )
        y += 8

        y = self._render_section_header("SPEED", y, padding)
        for btn in self.speed_selector.buttons:
            btn.rect.y = y
        self.speed_selector.render(self._sidebar_surface, self.font_small)
        y += 35

        # === HELP ===
        pygame.draw.line(
            self._sidebar_surface, colors.DIVIDER,
            (padding, y), (self.sidebar_width - padding, y)
        )
        y += 10

        hints = ["SPACE pause/resume", "ESC quit"]
        for hint in hints:
            hint_surface = self.font_small.render(hint, True, colors.TEXT_SECONDARY)
            self._sidebar_surface.blit(hint_surface, (padding, y))
            y += 16

    def _render_section_header(self, title: str, y: int, padding: int) -> int:
        """Render a section header and return new y position."""
        header_surface = self.font_small.render(title, True, UI_COLORS.accent)
        self._sidebar_surface.blit(header_surface, (padding, y))
        return y + 32  # Header height + gap + space for slider labels (15px above slider)

    def _render_stats_section(self, world: World, y: int, padding: int) -> int:
        """Render the stats section with charts."""
        chart_width = 90
        chart_height = 24

        y = self._render_section_header("LIVE STATS", y, padding)

        # Population row
        pop_label = self.font_small.render(f"Pop: {world.stats.beings_alive}", True, colors.TEXT_PRIMARY)
        self._sidebar_surface.blit(pop_label, (padding, y + 4))
        self.chart_population.rect.x = self.sidebar_width - padding - chart_width
        self.chart_population.rect.y = y
        self.chart_population.rect.width = chart_width
        self.chart_population.rect.height = chart_height
        self.chart_population.render(self._sidebar_surface, list(world.stats_history.population), min_val=0)
        y += 30

        # Energy row
        energy_label = self.font_small.render(f"Energy: {world.stats.avg_energy:.2f}", True, colors.TEXT_PRIMARY)
        self._sidebar_surface.blit(energy_label, (padding, y + 4))
        self.chart_energy.rect.x = self.sidebar_width - padding - chart_width
        self.chart_energy.rect.y = y
        self.chart_energy.rect.width = chart_width
        self.chart_energy.rect.height = chart_height
        self.chart_energy.render(self._sidebar_surface, list(world.stats_history.avg_energy), min_val=0, max_val=1)
        y += 30

        # Resources row
        food_water = f"F:{world.stats.food_count} W:{world.stats.water_count}"
        res_label = self.font_small.render(food_water, True, colors.TEXT_PRIMARY)
        self._sidebar_surface.blit(res_label, (padding, y + 4))
        self.chart_food.rect.x = self.sidebar_width - padding - chart_width
        self.chart_food.rect.y = y
        self.chart_food.rect.width = chart_width
        self.chart_food.rect.height = chart_height
        self.chart_food.render(self._sidebar_surface, list(world.stats_history.food_count), min_val=0)
        y += 30

        # Deaths
        deaths_label = self.font_small.render(f"Deaths: {world.stats.total_deaths}", True, colors.TEXT_SECONDARY)
        self._sidebar_surface.blit(deaths_label, (padding, y))
        y += 22

        return y

    def _render_brush_section(self, world: World, y: int, padding: int) -> int:
        """Render brush tools section for setup mode."""
        y = self._render_section_header("BRUSH TOOLS", y, padding)

        # Brush buttons
        for btn in self.brush_buttons:
            btn.rect.y = y
        for btn in self.brush_buttons:
            btn.render(self._sidebar_surface, self.font_small)
        y += 45  # Extra space for slider label below buttons

        # Brush size slider
        self.slider_brush_size.rect.y = y
        self.slider_brush_size.render(self._sidebar_surface, self.font_small)
        y += 35

        # Current counts
        from ..simulation.resource import ResourceType
        food_count = sum(1 for r in world.resources if r.type == ResourceType.FOOD)
        water_bodies = len(world.terrain.water_bodies)
        counts = f"Beings: {len(world.beings)}, Food: {food_count}"
        counts_surface = self.font_small.render(counts, True, colors.TEXT_SECONDARY)
        self._sidebar_surface.blit(counts_surface, (padding, y))
        y += 18

        terrain_info = f"Water bodies: {water_bodies}, Obstacles: {len(world.terrain.obstacles)}"
        terrain_surface = self.font_small.render(terrain_info, True, colors.TEXT_SECONDARY)
        self._sidebar_surface.blit(terrain_surface, (padding, y))
        y += 22

        return y

    def tick(self) -> float:
        """
        Advance the renderer clock and return delta time.

        Returns:
            Time elapsed since last tick in seconds.
        """
        # Adjust FPS based on speed
        target_fps = self.config.target_fps
        if self.speed_multiplier < 1.0:
            target_fps = int(self.config.target_fps * self.speed_multiplier)

        return self.clock.tick(target_fps) / 1000.0

    def cleanup(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()
