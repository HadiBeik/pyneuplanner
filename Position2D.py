import numpy as np
import math


class Position2D():
    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def __add__(self, other):
        return Position2D(self.x + other.x, self.y + other.y)
