# Evolving Beings

A living being simulator exploring emergent behavior in 2D worlds.

Beings have simple needs (food, water, energy) and basic perception (vision). The goal is to discover what minimal primitives can yield complex, interesting behavior—perhaps even communication and collaboration.

![Main UI](screenshots/main.png?raw=true "Evolving Beings Simulation")

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the simulation
python run.py
```

## Features

- **Living Beings**: Entities with energy, hunger, thirst, and vision
- **Resource System**: Food and water spawn in the world, beings must find and consume them
- **Simple AI**: Rule-based survival behavior (seek food when hungry, water when thirsty)
- **Efficient Simulation**: Spatial hash grid for O(1) neighbor lookups
- **Clean Architecture**: Simulation and rendering are fully separated

## Architecture

The codebase separates simulation logic from rendering:

```
src/evolving_beings/
├── simulation/          # Pure logic, no rendering
│   ├── being.py         # Being entity
│   ├── resource.py      # Food/water
│   ├── spatial.py       # Spatial hash grid
│   └── world.py         # World manager
└── renderer/            # Visualization
    ├── colors.py        # Color schemes
    └── pygame_renderer.py
```

This enables headless training runs and easy renderer swapping.

## Controls

- **ESC**: Quit simulation

## Configuration

Edit `src/evolving_beings/config.py` to adjust:
- World size and initial population
- Resource spawn rates
- Being energy dynamics, vision parameters
- Renderer settings (FPS, entity sizes)

## Roadmap

See [IDEAS.md](IDEAS.md) for planned future enhancements including:
- Genetic algorithm for evolving being parameters
- Neural network brains with reinforcement learning
- Being communication and social dynamics
- Save/load and reproducible simulations

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src/

# Install notebook dependencies
pip install -e ".[notebook]"
```

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation and extension guides.
