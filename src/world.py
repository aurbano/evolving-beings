import numpy as np
import random

from src.being import Being
from src.cell import Cell


class World:
    def __init__(self, w=128, h=128):
        self.w = w
        self.h = h
        self.state = np.empty((w, h), dtype=object)
        for i in range(w):
            for j in range(h):
                self.state[i, j] = Cell(i, j)

    def step(self):
        state = np.copy(self.state)

        for i, row in enumerate(self.state):
            for j, cell in enumerate(row):
                if cell.type != 'BEING':
                    continue

                cell.content.step()
                action = cell.content.choose_action()

                if action == 'MOVE':
                    # lets see if the desired cell is empty
                    direction = cell.content.direction
                    next_loc = [
                        max(0, min(self.w - 1, cell.x + direction[0])),
                        max(0, min(self.h - 1, cell.y + direction[1]))
                    ]

                    if state[next_loc[0], next_loc[1]].type == 'NONE':
                        # cell is empty, lets move!
                        state[next_loc[0], next_loc[1]].update('BEING', cell.content)
                        state[i, j].update('NONE')

        self.state = state

    def spawn(self, number):
        for _ in range(number):
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)

            while self.state[x, y].type != 'NONE':
                x = random.randint(0, self.w - 1)
                y = random.randint(0, self.h - 1)
                # TODO: this could be infinite

            being = Being()
            self.state[x, y].update('BEING', being)

    def render(self):
        state = np.zeros((self.w, self.h))
        for i, row in enumerate(self.state):
            for j, cell in enumerate(row):
                state[i, j] = cell.color()

        return state
