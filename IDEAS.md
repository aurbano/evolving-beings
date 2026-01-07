# Future Enhancement Ideas

This document captures ideas for future improvements to the Evolving Beings simulation. These are intentionally deferred to keep the initial implementation focused.

---

## Infrastructure

### Event System
Decouple components using an event bus pattern:
- `BeingSpawned`, `BeingDied`, `ResourceConsumed`, `ResourceSpawned`
- Enables analytics, achievements, training signals without coupling
- Renderers/trainers subscribe to relevant events

```python
# Example API
class EventBus:
    def emit(self, event: Event) -> None: ...
    def subscribe(self, event_type: type, handler: Callable) -> None: ...
```

### Fixed Timestep Simulation
Separate simulation tick rate from render FPS:
- Ensures reproducible behavior regardless of hardware
- Critical for training and replays
- Accumulate time, step simulation at fixed intervals

```python
accumulator = 0.0
FIXED_DT = 1/60

while running:
    accumulator += delta_time
    while accumulator >= FIXED_DT:
        world.step(FIXED_DT)
        accumulator -= FIXED_DT
    renderer.render(world)
```

### Seeded Randomness
Enable reproducible simulations:
- World accepts a seed, creates a `random.Random` instance
- Beings get deterministic sub-seeds based on their ID
- Useful for debugging, training, and sharing interesting scenarios

### Save/Load System
Serialize world state:
- Save snapshots at interesting moments
- Resume simulations from saved state
- Export/import trained beings
- Consider using `pickle` for simplicity or JSON for portability

### Metrics Collector
Track statistics over time:
- Population, average energy, deaths per tick
- Resource consumption rates
- Being behavior distribution (% eating, moving, etc.)
- Export to CSV for analysis, or feed to live charts

---

## Gameplay / Simulation

### Genetic Algorithm
Evolve being parameters over generations:
- Vision range, metabolism rates, speed
- Beings that survive longer reproduce
- Offspring inherit mutated parameters
- Track lineages and evolutionary fitness

### Neural Network Brains
Replace rule-based behavior with learned policies:
- Small neural network per being (or shared species brain)
- Inputs: vision, internal state
- Outputs: action probabilities
- Train via genetic algorithms or reinforcement learning

### Being Communication
Enable beings to signal each other:
- Change their "display color" as an action
- Other beings can see this color in their vision
- Could evolve cooperation or deception

### Predator/Prey Dynamics
Add conflict between beings:
- Beings can attack each other
- Dead beings become food resources
- Creates natural selection pressure
- Different "species" with different strategies

### Terrain/Obstacles
Add structure to the world:
- Walls, obstacles beings must navigate
- Terrain types (grass, desert) with different resource spawn rates
- Water bodies that beings must avoid or cross

### Day/Night Cycle
Add temporal variation:
- Resources spawn differently at different times
- Being vision reduced at "night"
- Creates need for different strategies

---

## Visualization

### Web-Based Renderer
Alternative to Pygame for browser deployment:
- Python backend with WebSocket
- React/Canvas or WebGL frontend
- Share simulations via URL

### Heatmaps
Visualize aggregate behavior:
- Where beings spend most time
- Resource consumption hotspots
- Death locations

### Lineage Viewer
Visualize evolutionary history:
- Family trees of beings
- Trait inheritance over generations
- Identify successful lineages

### Time Controls
Add playback controls:
- Pause/resume
- Speed up (skip rendering)
- Slow motion for observation
- Step single tick

---

## Performance

### NumPy Vectorization
Batch operations for better performance:
- Update all being positions in one vectorized operation
- Compute all vision queries using array operations

### Entity Component System
More flexible than class inheritance:
- Components: Position, Vision, Metabolism, Brain
- Systems: MovementSystem, VisionSystem, EnergySystem
- Easier to add new capabilities, better cache locality

### GPU Acceleration
For very large populations:
- CUDA/OpenCL for parallel simulation
- Compute shaders for vision queries
- Could support millions of beings

---

## Training

### Headless Mode
Run simulation without renderer:
- Command-line interface
- Maximum speed for training
- Output statistics to file

### Distributed Training
Train across multiple machines:
- Each machine runs independent simulations
- Share best-performing genomes/weights
- Scale to massive experiments

### Curriculum Learning
Gradually increase difficulty:
- Start with abundant resources
- Reduce spawn rates over time
- Add predators after basic survival is learned

