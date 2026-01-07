# Evolving Beings - Architecture Guide

## Overview

Evolving Beings is a living being simulator exploring emergent behavior in 2D worlds. Beings have simple needs (food, water, energy) and basic perception (vision), with the goal of discovering what minimal primitives can yield complex, interesting behavior.

## Architecture

The codebase follows a strict separation between **simulation** and **rendering**:

```
src/evolving_beings/
├── config.py              # Centralized configuration (dataclasses)
├── main.py                # Application entry point
├── simulation/            # Pure logic, no rendering
│   ├── being.py           # Being entity and behavior
│   ├── resource.py        # Food/water resources
│   ├── spatial.py         # Spatial hash grid for O(1) lookups
│   └── world.py           # World manager, orchestrates simulation
└── renderer/              # Visualization layer
    ├── colors.py          # Color definitions and utilities
    └── pygame_renderer.py # Pygame-CE based renderer
```

### Key Design Decisions

1. **Simulation/Renderer Separation**: The simulation module has zero knowledge of rendering. This enables:
   - Headless training runs (no GUI overhead)
   - Easy renderer swapping (could add web-based renderer)
   - Clear testing boundaries

2. **Spatial Hash Grid**: Replaces O(n²) neighbor searches with O(1) cell lookups. Critical for scaling to thousands of beings.

3. **Float Positions**: Beings use continuous (x, y) coordinates rather than discrete grid cells, allowing smooth movement and more natural behavior.

4. **Configuration via Dataclasses**: All parameters are centralized in `config.py` with sensible defaults, making experimentation easy.

5. **Rule-Based AI (for now)**: Beings use simple heuristics (seek food when hungry, etc.) rather than learned behavior. This provides a working baseline for future ML integration.

## Core Concepts

### Being
A living entity with:
- **Position/Orientation**: (x, y) and angle in radians
- **Internal Resources**: food, water (0-1 range)
- **Energy**: Depletes over time, replenished from food/water. Death at 0.
- **Vision**: Configurable range and field-of-view angle
- **Actions**: IDLE, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, EAT, DRINK

### Resource
Static world entities (food or water):
- Have position and amount (depletes when consumed)
- Spawn continuously based on configured rates

### World
Orchestrates the simulation:
- Manages entity lifecycles (spawn, update, death)
- Maintains spatial grids for efficient queries
- Tracks statistics per tick

### Renderer
Queries world state and draws to screen:
- Beings colored by health state
- Resources colored by type and fullness
- Sidebar shows live statistics

## Running

```bash
# Install dependencies
pip install -e .

# Run simulation
python run.py
```

## Configuration

Edit `src/evolving_beings/config.py` to adjust:
- World size and initial population
- Resource spawn rates
- Being energy dynamics, vision parameters
- Renderer settings (FPS, entity sizes)

## Extending

### Adding New Behavior
Modify `World._decide_action()` in `world.py`. This is where the "brain" logic lives. Currently rule-based, but designed to be replaced with neural networks or other learning systems.

### Adding New Entity Types
1. Create a new class in `simulation/`
2. Add a spatial grid for it in `World.__init__`
3. Add spawn/update logic to `World`
4. Add rendering logic to `PygameRenderer._render_world()`

### Adding a New Renderer
Implement a class with:
- `handle_events() -> bool`
- `render(world: World) -> None`
- `tick() -> float`
- `cleanup() -> None`

