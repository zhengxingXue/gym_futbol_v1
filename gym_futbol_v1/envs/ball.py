import pymunk
from pymunk.vec2d import Vec2d
from .object import Object


class Ball(Object):

    def __init__(self, space, x, y, mass=10, radius=1, max_velocity=20,
                 elasticity=0.2, color=(0, 1, 0, 1)):
        super().__init__(space, x, y, mass, radius, max_velocity, elasticity, color)

    def has_contact_with(self, player):
        """
        check if the ball and the player has contact.
        """
        return self.shape.shapes_collide(player.shape).points != []

    def apply_force_to_ball(self, fx, fy):
        self.apply_force_to_object(fx, fy)
