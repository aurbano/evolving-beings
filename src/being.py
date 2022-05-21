import numpy as np
from math import cos, sin, degrees, atan2, hypot
import random

# Hyper-parameters
ENERGY_LOSS_GENERAL = 0.001
ENERGY_LOSS_ACTIONS = 0.005

FOOD_TO_ENERGY = 0.001
WATER_TO_ENERGY = 0.001

# Rotation vectors for movement
rotation_angle = 45
theta = np.deg2rad(rotation_angle)
rot_right = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
rot_left = np.array([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])


class Being:
    """
    Living being representation. If energy reaches 0 it will be dead

    Energy goes down when performing actions, low energy leads to lower happiness.
    """
    def __init__(self, sprite_index):
        # Subjective states (things the "brain" feels)
        self.happiness = 1.
        self.hunger = 0.
        self.thirst = 0.

        # Objective states (hidden from the "brain")
        self.food = 1.
        self.water = 1.
        self.energy = 1.

        # State
        self.angle = 0.
        self.direction = [1, 0]  # vector from (0,0) (the being) to the direction its facing
        self.speed = 0.
        self.action_space = ['NOOP', 'TURN_LEFT', 'TURN_RIGHT', 'MOVE', 'STOP', 'EAT', 'DRINK']

        # Vision
        self.vision_angle = 45      # degrees of vision, it can see forward, and vision_angle/2 to each side
        self.vision_pixels = 5      # resolution of vision, size of the 2-d array it can "see"
        self.vision_distance = 10   # distance it can see objects at

        # Don't change these
        self.vision_chunk_size = self.vision_angle // self.vision_pixels
        self.sprite_index = sprite_index

    def choose_action(self, vision):
        """
        Choose an action based on the current state and the vision
        :param vision:
        :return:
        """
        state = [self.happiness, self.hunger, self.thirst, self.energy, *vision]

        # TODO: Use the state to learn somehow

        action = random.choice(self.action_space)

        if action == 'TURN_LEFT' or action == 'TURN_RIGHT':
            rot = rot_left if action == 'TURN_LEFT' else rot_right
            self.direction = np.round(np.dot(rot, self.direction), 0).astype(int)
            self.angle += theta if action == 'TURN_LEFT' else -theta

        return action

    def step(self, location, being_locations):
        self.energy = max(0, self.energy - ENERGY_LOSS_GENERAL)

        if self.energy < 1:
            if self.food > 0:
                self.food = max(0, self.food - FOOD_TO_ENERGY)
                self.energy = min(1, self.energy + FOOD_TO_ENERGY)

            if self.water > 0:
                self.water = max(0, self.water - WATER_TO_ENERGY)
                self.energy = min(1, self.energy + WATER_TO_ENERGY)

        if self.speed > 0:
            self.energy = max(0, self.energy - ENERGY_LOSS_ACTIONS)

        vision = self.vision(location, being_locations)

        return self.choose_action(vision)

    def vision(self, location, locations):
        """
        Calculate the vision array
        :param locations:
        :return:
        """
        direction_angle = degrees(atan2(self.direction[1], self.direction[0]))

        min_angle = direction_angle - self.vision_angle / 2
        max_angle = direction_angle + self.vision_angle / 2

        min_angle %= 360
        max_angle %= 360

        vision = [0] * self.vision_pixels

        # calculate beings in its field of view
        for coords in locations:
            if coords == location:
                # skip itself
                continue

            y = coords[1] - location[1]
            x = coords[0] - location[0]

            angle = degrees(atan2(y, x))
            angle %= 360

            if min_angle <= angle <= max_angle:
                dist = hypot(x, y)
                if dist <= self.vision_distance:
                    # visible
                    vision_chunk = int((angle - min_angle) // self.vision_chunk_size)
                    vision[vision_chunk] += 1

        return vision

    def is_alive(self):
        return self.energy > 0