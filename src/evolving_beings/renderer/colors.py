"""Color definitions for the renderer."""

# Background
BG_DARK = (28, 28, 32)
BG_SIDEBAR = (38, 38, 45)

# World background - earthy tones
WORLD_BG = (62, 54, 45)
WORLD_BG_ACCENT = (72, 64, 52)

# Beings - color based on health
BEING_HEALTHY = (64, 224, 208)  # Turquoise
BEING_HUNGRY = (255, 165, 0)  # Orange
BEING_DYING = (220, 20, 60)  # Crimson

# Resources
FOOD_COLOR = (124, 252, 0)  # Lawn green
FOOD_DEPLETED = (60, 120, 0)  # Darker green
WATER_COLOR = (30, 144, 255)  # Dodger blue
WATER_DEPLETED = (20, 80, 140)  # Darker blue

# Terrain
OBSTACLE_COLOR = (55, 45, 40)  # Dark brown/rock
OBSTACLE_HIGHLIGHT = (75, 65, 55)  # Lighter edge
WATER_BODY_COLOR = (25, 105, 180)  # Deep water blue
WATER_BODY_EDGE = (60, 160, 220)  # Lighter water edge
WATER_BODY_ALPHA = 180  # Transparency for water

# Mountain obstacles
MOUNTAIN_COLOR = (75, 65, 55)  # Gray-brown rock
MOUNTAIN_HIGHLIGHT = (95, 85, 75)  # Lighter highlights
MOUNTAIN_SHADOW = (45, 38, 32)  # Darker shadows

# Elevation-based ground colors (low to high)
GROUND_LOW = (52, 48, 38)  # Dark lowlands (near water)
GROUND_MID = (65, 58, 45)  # Mid elevation
GROUND_HIGH = (80, 72, 58)  # Higher ground (approaching mountains)

# UI
TEXT_PRIMARY = (240, 240, 245)
TEXT_SECONDARY = (160, 160, 170)
TEXT_ACCENT = (100, 200, 255)
DIVIDER = (60, 60, 70)

# Vision cone (debug)
VISION_CONE = (255, 255, 255, 30)  # Semi-transparent white


def lerp_color(
    color1: tuple[int, int, int],
    color2: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Linearly interpolate between two colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t),
    )


def get_being_color(energy: float, food: float, water: float) -> tuple[int, int, int]:
    """Get the color for a being based on its state."""
    # Blend based on overall health
    health = min(energy, (food + water) / 2)

    if health > 0.6:
        return BEING_HEALTHY
    elif health > 0.3:
        return lerp_color(BEING_HUNGRY, BEING_HEALTHY, (health - 0.3) / 0.3)
    else:
        return lerp_color(BEING_DYING, BEING_HUNGRY, health / 0.3)


def get_resource_color(resource_type: str, fullness: float) -> tuple[int, int, int]:
    """Get the color for a resource based on type and fullness."""
    if resource_type == "FOOD":
        return lerp_color(FOOD_DEPLETED, FOOD_COLOR, fullness)
    else:
        return lerp_color(WATER_DEPLETED, WATER_COLOR, fullness)


def get_ground_color(height: float, water_level: float, mountain_level: float) -> tuple[int, int, int]:
    """
    Get ground color based on elevation.

    Args:
        height: Terrain height (0-1)
        water_level: Height below which is water
        mountain_level: Height above which is mountain

    Returns:
        RGB color tuple for the ground at this elevation
    """
    if height <= water_level:
        # Below water - shouldn't normally be rendered, but handle it
        return GROUND_LOW

    # Normalize height to land range (water_level to mountain_level)
    land_range = mountain_level - water_level
    if land_range <= 0:
        return GROUND_MID

    # t goes from 0 (at water level) to 1 (at mountain level)
    t = (height - water_level) / land_range
    t = max(0.0, min(1.0, t))

    # Blend from low to mid to high
    if t < 0.5:
        return lerp_color(GROUND_LOW, GROUND_MID, t * 2)
    else:
        return lerp_color(GROUND_MID, GROUND_HIGH, (t - 0.5) * 2)

