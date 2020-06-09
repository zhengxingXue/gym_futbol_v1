"""
Player Module.
"""
from .object import Object


class Player(Object):
    """
    Player Class.
    """

    def __init__(self, space, x, y, mass=20, radius=1.5, max_velocity=10,
                 elasticity=0.2, color=(1, 0, 0, 1), side="left"):
        super().__init__(space, x, y, mass, radius, max_velocity, elasticity, color)
        self.side = side

    def apply_force_to_player(self, fx, fy):
        self.apply_force_to_object(fx, fy)
