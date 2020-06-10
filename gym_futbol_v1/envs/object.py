"""
Contains object class used in the futbol environment.
"""
import pymunk
from pymunk.vec2d import Vec2d


class Object():

    def __init__(self, space, x, y, mass, radius, max_velocity, elasticity, color):
        self.space = space
        self.max_velocity = max_velocity
        self.color = color

        self.body, self.shape = self._setup_object(
            space, x, y, mass, radius, elasticity)

    def get_position(self):
        x, y = self.body.position
        return [x, y]

    def get_velocity(self):
        vx, vy = self.body.velocity
        return [vx, vy]

    def get_observation(self):
        return self.get_position() + self.get_velocity()

    def set_position(self, x, y):
        self.body.position = x, y

    def apply_force_to_object(self, fx, fy):
        # self.body.apply_force_at_local_point((fx,fy), point=(0, 0))
        self.body.apply_impulse_at_local_point((fx, fy), point=(0, 0))

    def _setup_object(self, space, x, y, mass, radius, elasticity):
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, moment)
        body.position = x, y
        body.start_position = Vec2d(body.position)

        def limit_velocity(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, damping, dt)
            l = body.velocity.length
            if l > self.max_velocity:
                scale = self.max_velocity / l
                body.velocity = body.velocity * scale

        body.velocity_func = limit_velocity
        shape = pymunk.Circle(body, radius)
        shape.color = self.color  # green, (R,G,B,A)
        shape.elasticity = elasticity
        self.space.add(body, shape)

        return body, shape


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
