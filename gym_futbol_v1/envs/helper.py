"""
contain helper function for futbol_env
"""
import math
import pymunk
import numpy as np


def get_vec(coor_t, coor_o):
    """
    get the vector pointing from [coor2] to [coor1] and its magnitude
    """
    vec = [coor_t[0] - coor_o[0], coor_t[1] - coor_o[1]]
    vec_mag = math.sqrt(vec[0]**2 + vec[1]**2)
    return vec, vec_mag


def setup_walls(space, width, height, goal_size):
    """
    Create walls.
    """
    static = [
        pymunk.Segment(
            space.static_body,
            (0, 0), (0, height/2 - goal_size/2), 1),
        pymunk.Segment(
            space.static_body,
            (0, height/2 + goal_size/2), (0, height), 1),
        pymunk.Segment(
            space.static_body,
            (0, height), (width, height), 1),
        pymunk.Segment(
            space.static_body,
            (width, 0), (width, height/2 - goal_size/2), 1),
        pymunk.Segment(
            space.static_body,
            (width, height/2 + goal_size/2), (width, height), 1),
        pymunk.Segment(
            space.static_body,
            (0, 0), (width, 0), 1)
    ]

    static_goal = [
        pymunk.Segment(
            space.static_body,
            (-2, height/2 - goal_size/2), (-2, height/2 + goal_size/2), 1),
        pymunk.Segment(
            space.static_body,
            (-2, height/2 - goal_size/2), (0, height/2 - goal_size/2), 1),
        pymunk.Segment(
            space.static_body,
            (-2, height/2 + goal_size/2), (0, height/2 + goal_size/2), 1),
        pymunk.Segment(
            space.static_body,
            (width+2, height/2 - goal_size/2), (width+2, height/2 + goal_size/2), 1),
        pymunk.Segment(
            space.static_body,
            (width, height/2 - goal_size/2), (width+2, height/2 - goal_size/2), 1),
        pymunk.Segment(
            space.static_body,
            (width, height/2 + goal_size/2), (width+2, height/2 + goal_size/2), 1)
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
