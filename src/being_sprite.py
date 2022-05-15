import arcade

BEING_SCALE = 0.1


class BeingSprite(arcade.Sprite):
    def __init__(self):
        super().__init__()

        self.scale = BEING_SCALE
        self.texture = arcade.load_texture(":resources:images/enemies/slimeBlue.png")
