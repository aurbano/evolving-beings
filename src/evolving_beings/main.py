"""Main entry point for the Evolving Beings simulation."""

from .config import Config
from .renderer import PygameRenderer, SimulationMode
from .simulation import World


def main() -> None:
    """Run the evolving beings simulation."""
    # Load configuration
    config = Config.default()

    # Create world
    world = World(config.world, config.being)
    world.initialize()

    # Create renderer
    renderer = PygameRenderer(config.renderer)
    renderer.set_world(world)

    # Sync initial slider values with config
    renderer.slider_food_rate.value = config.world.food_spawn_rate
    renderer.slider_clustering.value = config.world.resource_clustering
    renderer.slider_initial_pop.value = config.world.initial_population
    renderer.slider_initial_food.value = config.world.initial_food

    print("Starting Evolving Beings simulation...")
    print(f"  Seed: {world.seed}")
    print(f"  Population: {config.world.initial_population}")
    print(f"  Food: {config.world.initial_food}")
    print(f"  Terrain style: {config.world.terrain_style}")
    print(f"  Terrain: {len(world.terrain.water_bodies)} water bodies, {len(world.terrain.mountain_obstacles)} mountains")
    print(f"  World size: {config.world.width}x{config.world.height}")
    print()
    print("Controls:")
    print("  - Click 'Run' or press SPACE to start simulation")
    print("  - Click 'Step' when paused to advance one tick")
    print("  - Use sliders to adjust spawn rates")
    print("  - In Setup mode, use brushes to paint resources")
    print("  - ESC to quit")
    print()

    # Main loop
    running = True
    while running:
        # Handle input
        running = renderer.handle_events(world)

        # Update simulation if running
        if renderer.should_step():
            steps = renderer.get_steps_per_frame()
            for _ in range(steps):
                world.step()

                # Check if all beings are dead
                if world.stats.beings_alive == 0:
                    print(f"All beings have died after {world.stats.tick} ticks.")
                    print(f"Total deaths: {world.stats.total_deaths}")
                    # Pause simulation
                    renderer._set_mode(SimulationMode.PAUSED)
                    break

        # Render
        renderer.render(world)

        # Tick
        renderer.tick()

    # Cleanup
    renderer.cleanup()
    print("Simulation ended.")


if __name__ == "__main__":
    main()
