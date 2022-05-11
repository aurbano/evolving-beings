class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = 'NONE'
        self.content = None

    def update(self, type, content=None):
        self.type = type
        self.content = content

    def color(self):
        if self.type == 'NONE':
            return 0

        if self.type == 'BEING':
            return self.content.color()