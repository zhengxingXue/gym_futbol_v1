"""
contain helper function for futbol_env
"""
import math
import enum
import random
import pymunk
from pymunk.vec2d import Vec2d
import numpy as np


class Side(enum.Enum):
    left = 'left'
    right = 'right'
    NoSide = 'NoSide'


class Object:
    """
    Pymunk object Class.
    """

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
                 elasticity=0.2, color=(1, 0, 0, 1), side=Side.left):
        super().__init__(space, x, y, mass, radius, max_velocity, elasticity, color)
        self.side = side

    def apply_force_to_player(self, fx, fy):
        self.apply_force_to_object(fx, fy)


class Ball(Object):
    """
    Ball Class.
    """

    def __init__(self, space, x, y, mass=10, radius=1, max_velocity=20,
                 elasticity=0.2, color=(0, 1, 0, 1)):
        super().__init__(space, x, y, mass, radius, max_velocity, elasticity, color)
        self.owner_side = Side.left
        self.last_owner_side = Side.NoSide

    def has_contact_with(self, player):
        """
        check if the ball and the player has contact.
        """
        return self.shape.shapes_collide(player.shape).points != []

    def apply_force_to_ball(self, fx, fy):
        self.apply_force_to_object(fx, fy)

    def change_owner_side(self, side):
        """
        change the owner side to the opposite team of the current team.
        """
        if side is not Side.NoSide:
            self.last_owner_side = self.owner_side
        self.owner_side = side


def get_vec(coor_t, coor_o):
    """
    get the vector pointing from [coor2] to [coor1] and its magnitude
    """
    vec = [coor_t[0] - coor_o[0], coor_t[1] - coor_o[1]]
    vec_mag = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return vec, vec_mag


def setup_walls(space, width, height, goal_size):
    """
    Create walls.
    """
    static = [
        pymunk.Segment(
            space.static_body,
            (0, 0), (0, height / 2 - goal_size / 2), 1),
        pymunk.Segment(
            space.static_body,
            (0, height / 2 + goal_size / 2), (0, height), 1),
        pymunk.Segment(
            space.static_body,
            (0, height), (width, height), 1),
        pymunk.Segment(
            space.static_body,
            (width, 0), (width, height / 2 - goal_size / 2), 1),
        pymunk.Segment(
            space.static_body,
            (width, height / 2 + goal_size / 2), (width, height), 1),
        pymunk.Segment(
            space.static_body,
            (0, 0), (width, 0), 1)
    ]

    static_goal = [
        pymunk.Segment(
            space.static_body,
            (-2, height / 2 - goal_size / 2), (-2, height / 2 + goal_size / 2), 1),
        pymunk.Segment(
            space.static_body,
            (-2, height / 2 - goal_size / 2), (0, height / 2 - goal_size / 2), 1),
        pymunk.Segment(
            space.static_body,
            (-2, height / 2 + goal_size / 2), (0, height / 2 + goal_size / 2), 1),
        pymunk.Segment(
            space.static_body,
            (width + 2, height / 2 - goal_size / 2), (width + 2, height / 2 + goal_size / 2), 1),
        pymunk.Segment(
            space.static_body,
            (width, height / 2 - goal_size / 2), (width + 2, height / 2 - goal_size / 2), 1),
        pymunk.Segment(
            space.static_body,
            (width, height / 2 + goal_size / 2), (width + 2, height / 2 + goal_size / 2), 1)
    ]

    for s in static + static_goal:
        s.friction = 1.
        s.group = 1
        s.collision_type = 1

    space.add(static)
    space.add(static_goal)

    return space, static, static_goal


def normalize_array(array, array_max, array_min, n=1):
    """normalize array"""
    array_avg = np.tile((array_max + array_min) / 2, n)
    array_range = np.tile((array_max - array_min) / 2, n)
    array = (array - array_avg) / array_range
    return array


def ball_move_with_player(ball, player):
    """
    if player has contact with ball and move, let the ball move with the player.
    """
    if ball.has_contact_with(player):
        ball.body.velocity = player.body.velocity
    else:
        pass


def ball_contact_wall(ball, static):
    """
    return true and wall index if the ball is in contact with the walls
    """
    wall_index, i = -1, 0
    for wall in static:
        if ball.shape.shapes_collide(wall).points:
            wall_index = i
            return True, wall_index
        i += 1
    return False, wall_index


def ball_contact_goal(ball, static_goal):
    """
    return true if score
    """
    goal = False
    for goal_wall in static_goal:
        goal = goal or ball.shape.shapes_collide(
            goal_wall).points != []
    return goal


def ball_outside_wall(ball, width, height, goal_size):
    """return whether the ball is outside the wall"""
    bx, by = ball.get_position()
    if 0 < by < height:
        if height / 2 - goal_size / 2 <= by <= height / 2 + goal_size / 2:
            if bx <= -2 or bx >= width + 2:
                return True
            else:
                return False
        else:
            if bx <= 0 or bx >= width:
                return True
            else:
                return False
    else:
        return True


def check_and_fix_out_bounds(ball, static, team_A, team_B):
    # TODO: Fix Out bug
    """
    check if the ball contact the walls.
    if contact: change ball owner, fix, and return true
    else: return false
    """
    out, wall_index = ball_contact_wall(ball, static)
    if out:
        bx, by = ball.get_position()
        dbx, dby, dpx, dpy = 0, 0, 0, 0

        if wall_index == 1 or wall_index == 0:  # left bound
            dbx, dpx = 3.5, 1
        elif wall_index == 3 or wall_index == 4:
            dbx, dpx = -3.5, -1
        elif wall_index == 2:
            dby, dpy = -3.5, -1
        else:
            dby, dpy = 3.5, 1

        ball.set_position(bx + dbx, by + dby)
        ball.body.velocity = 0, 0

        last_owner_side = ball.last_owner_side

        new_owner_side = Side.left if last_owner_side == Side.right else Side.right

        ball.change_owner_side(new_owner_side)

        if ball.owner_side == team_B.side:
            get_ball_player = random.choice(team_B.player_array)
        else:
            get_ball_player = random.choice(team_A.player_array)

        get_ball_player.set_position(bx + dpx, by + dpy)
        get_ball_player.body.velocity = 0, 0
    else:
        pass
    return out
