import random
import operator

from src.being import Being


class World:
    def __init__(self, w=128, h=128):
        self.w = w
        self.h = h
        self.alive = 0

        self.locations = dict()

    def step(self):
        self.alive = 0
        new_locations = dict()
        dead_sprites = []
        for location, being in self.locations.items():
            if not being.is_alive():
                dead_sprites.append(being.sprite_index)
                continue

            self.alive += 1

            being.step()
            action = being.choose_action()

            if action == 'STOP':
                being.speed = 0

            if action == 'MOVE':
                being.speed = 1

            if being.speed > 0:
                new_location = (
                    max(0, min(self.w, location[0] + being.direction[0])),
                    max(0, min(self.h, location[1] + being.direction[1])),
                )

                if new_location not in self.locations and new_location not in new_locations:
                    new_locations[new_location] = being
                    continue

            new_locations[location] = being

        self.locations = new_locations

        return dead_sprites

    def spawn(self, sprite_index):
        x = random.randint(0, self.w - 1)
        y = random.randint(0, self.h - 1)
        location = (x, y)

        while location in self.locations:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            location = (x, y)

        self.alive += 1
        being = Being(sprite_index)
        self.locations[location] = being

        return location, being
