import numpy as np
import math
import Position2D as p


class Vector2D():

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 0
        self.angle = 0

    def from_angle_size(self, angle, size):
        self.x = size * math.cos(angle)
        self.y = size * math.sin(angle)

    def from_position(self, head, tail):
        self.x = tail.x - head.x
        self.y = tail.y - head.y

    def __add__(self, position):
        return p.Position2D(position.x + self.x, position.y + self.y)
