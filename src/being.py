import numpy as np
from math import cos, sin
import random

# Hyper-parameters
ENERGY_LOSS_GENERAL = 0.001
ENERGY_LOSS_ACTIONS = 0.005

FOOD_TO_ENERGY = 0.1
WATER_TO_ENERGY = 0.1

# Rotation vectors for movement
theta = np.deg2rad(45)
rot_right = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
rot_left = np.array([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])


class Being:
    """
    Living being representation. If energy reaches 0 it will be dead

    Energy goes down when performing actions, low energy leads to lower happiness.
    """
    def __init__(self, sprite_index):
        # Subjective states (things the "brain" feels)
        self.happiness = 1
        self.hunger = 0
        self.thirst = 0

        # Objective states (hidden from the "brain")
        self.food = 1
        self.water = 1
        self.energy = 1

        self.angle = 0
        self.direction = [1, 0]  # vector from (0,0) (the being) to the direction its facing
        self.speed = 0
        self.action_space = ['NOOP', 'TURN_LEFT', 'TURN_RIGHT', 'MOVE', 'STOP', 'EAT', 'DRINK']

        self.sprite_index = sprite_index

    def choose_action(self):
        action = random.choice(self.action_space)

        if action == 'TURN_LEFT' or action == 'TURN_RIGHT':
            rot = rot_left if action == 'TURN_LEFT' else rot_right
            self.direction = np.round(np.dot(rot, self.direction), 0).astype(int)
            self.angle += theta if action == 'TURN_LEFT' else -theta

        return action

    def step(self):
        self.energy -= ENERGY_LOSS_GENERAL

        if self.food > 0:
            self.food -= FOOD_TO_ENERGY
            self.energy += FOOD_TO_ENERGY

        if self.water > 0:
            self.water -= WATER_TO_ENERGY
            self.energy += WATER_TO_ENERGY

        if self.speed > 0:
            self.energy -= ENERGY_LOSS_ACTIONS

    def is_alive(self):
        return self.energy > 0