import random
import numpy as np
from typing import Optional, List

import arcade
from arcade.gui import UIManager
from arcade.gui.widgets import UITextArea

from src.being_sprite import BeingSprite
from src.world import World

WORLD_SIZE = 800
SIDEBAR_WIDTH = 200

SCREEN_WIDTH = WORLD_SIZE + SIDEBAR_WIDTH
SCREEN_HEIGHT = WORLD_SIZE
SCREEN_TITLE = "Evolving Beings"
DEBUG_PADDING = 10
FPS_LIMIT = 120
INITIAL_POPULATION = 1000

FOOD_SCALE = 0.25
TILE_SCALE = 1

TILE_SIZE = 64


class EvolvingBeings(arcade.Window):
    def __init__(self):
        super().__init__(
            width=SCREEN_WIDTH,
            height=SCREEN_HEIGHT,
            title=SCREEN_TITLE,
            update_rate=1/FPS_LIMIT,
        )

        self.background_color = arcade.color.DARK_BROWN

        self.manager: Optional[UIManager] = None
        self.debug_text: Optional[UITextArea] = None
        self.world: Optional[World] = None
        self.grid_sprite_list: Optional[arcade.SpriteList] = None
        self.food_sprite_list: Optional[arcade.SpriteList] = None
        self.bg_sprite_list: Optional[arcade.SpriteList] = None
        self.fps: List[float] = []
        self.available_sprites: List[int] = []

    def setup(self):
        """Sets up the world for the current simulation"""
        self.world = World(WORLD_SIZE, WORLD_SIZE)

        self.manager = UIManager()
        self.manager.enable()

        self.debug_text = UITextArea(
            x=DEBUG_PADDING,
            y=-DEBUG_PADDING,
            width=SCREEN_WIDTH - SCREEN_HEIGHT - DEBUG_PADDING,
            height=SCREEN_HEIGHT - DEBUG_PADDING,
            text='',
            text_color=(255, 255, 255, 255),
        )
        self.manager.add(self.debug_text)

        # cell rendering
        self.grid_sprite_list = arcade.SpriteList()
        self.food_sprite_list = arcade.SpriteList()

        # create beings
        for _ in range(INITIAL_POPULATION):
            idx = len(self.grid_sprite_list)
            location, _ = self.world.spawn(idx)

            being_sprite = BeingSprite()
            being_sprite.center_x = location[0] + SIDEBAR_WIDTH
            being_sprite.center_y = location[1]

            self.grid_sprite_list.append(being_sprite)

        self.bg_sprite_list = arcade.SpriteList(use_spatial_hash=True)
        for x in range(SIDEBAR_WIDTH, SCREEN_WIDTH, TILE_SIZE):
            for y in range(0, SCREEN_HEIGHT, TILE_SIZE):
                bg_tile_idx = random.randint(1, 2)
                bg = arcade.Sprite(f":resources:images/topdown_tanks/tileSand{bg_tile_idx}.png", TILE_SCALE)
                bg.center_x = x + TILE_SIZE / 2
                bg.center_y = y + TILE_SIZE / 2
                self.bg_sprite_list.append(bg)

    def on_update(self, delta_time: float):
        """Updates the position of all game objects

        Arguments:
            delta_time {float} -- How much time since the last call
        """
        self.fps.append(1 / delta_time)
        if len(self.fps) > 100:
            self.fps.pop(0)

        self.debug_text.text = f'FPS: {round(np.mean(self.fps))}/{FPS_LIMIT}\n' \
                               f'Beings alive: {self.world.alive}\n' \
                               f'Sprite buffer: {len(self.available_sprites)}\n' \

        idx = 0
        for location, being in self.world.locations.items():
            self.grid_sprite_list[idx].angle = being.angle
            if being.speed > 0:
                self.grid_sprite_list[idx].center_x = location[0] + SIDEBAR_WIDTH
                self.grid_sprite_list[idx].center_y = location[1]

            idx += 1

        dead_sprites = self.world.step()
        for sprite_index in dead_sprites:
            self.grid_sprite_list[sprite_index].visible = False
            self.available_sprites.append(sprite_index)

    def on_draw(self):
        self.clear()
        self.manager.draw()

        self.bg_sprite_list.draw()
        # self.food_sprite_list.draw()
        self.grid_sprite_list.draw()


if __name__ == "__main__":
    print('Init Evolving Beings')

    window = EvolvingBeings()
    window.setup()

    arcade.run()
