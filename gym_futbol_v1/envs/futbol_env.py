"""
Env Module.
"""
import random
import math
import io
import cv2
import gym
from gym import spaces
import numpy as np
import pymunk
import pymunk.matplotlib_util
import matplotlib.pyplot as plt
from .team import Team
from .helper import get_vec, setup_walls, normalize_array, Side, Ball, check_and_fix_out_bounds, ball_contact_goal
from .action import process_action


class Futbol(gym.Env):

    """
    Futbol is a 2D simulation of real world soccer.
    Soccer is a team sport played with a spherical ball between two teams of 11 players,
    including a goalkeeper. The goal is to get the ball into the net. For this simulation,
    user can specify the player number and the goalkeeper is not implemented for now.

    **STATE:**
    The state consists of position and velocity of each player and the ball.
    The state is flattened and normalized for better RL performance.

    **ACTIONS:**
    The action consists of arrow key and action key, inspired by PES2020.
    Arrow key controls the player action direction.
    Action key controls the action one player take at the moment.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    WIDTH = 105   # [m]
    HEIGHT = 68   # [m]
    GOAL_SIZE = 20  # [m]

    TOTAL_TIME = 30  # [s]
    TIME_STEP = 0.1  # [s]

    BALL_MAX_VELOCITY = 25  # [m/s]
    PLAYER_MAX_VELOCITY = 10  # [m/s]

    BALL_WEIGHT = 10
    PLAYER_WEIGHT = 20

    PLAYER_FORCE_LIMIT = 40
    BALL_FORCE_LIMIT = 120

    def __init__(self, debug=False, number_of_player=2, normalize_obs=True, goal_end=False):

        self.debug = debug
        self.number_of_player = number_of_player
        self.normalize_obs = normalize_obs
        self.goal_end = goal_end

        # action space
        # 1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        # 2) Action Keys: Discrete 5  - noop[0], dash[1], shoot[2], press[3], pass[4] - params: min: 0, max: 4
        self.action_space = spaces.MultiDiscrete(
            [5, 5] * self.number_of_player)

        # observation space (normalized)
        # [0] x position
        # [1] y position
        # [2] x velocity
        # [3] y velocity
        if self.normalize_obs:
            self.observation_space = spaces.Box(
                low=np.array([-1., -1., -1., -1.] *
                             (1+self.number_of_player * 2), dtype=np.float32),
                high=np.array([1., 1., 1., 1.] *
                              (1+self.number_of_player * 2), dtype=np.float32),
                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                low=np.array([-3, 0, -self.BALL_MAX_VELOCITY, -self.BALL_MAX_VELOCITY] +
                             [-3, 0, -self.PLAYER_MAX_VELOCITY, -self.PLAYER_MAX_VELOCITY] *
                             (self.number_of_player * 2), dtype=np.float32),
                high=np.array([self.WIDTH + 3, self.HEIGHT, self.BALL_MAX_VELOCITY, self.BALL_MAX_VELOCITY] +
                              [self.WIDTH + 3, self.HEIGHT, self.PLAYER_MAX_VELOCITY, self.PLAYER_MAX_VELOCITY] *
                              (self.number_of_player * 2), dtype=np.float32),
                dtype=np.float32)

        # create space
        self.space = pymunk.Space()
        self.space.gravity = 0, 0

        # Amount of simple damping to apply to the space.
        # A value of 0.9 means that each body will lose 10% of its velocity per second.
        self.space.damping = 0.95

        # create walls
        self.space, self.static, self.static_goal = setup_walls(
            self.space, self.WIDTH, self.HEIGHT, self.GOAL_SIZE)

        # Teams
        self.team_A = Team(self.space, self.WIDTH, self.HEIGHT,
                           player_weight=self.PLAYER_WEIGHT,
                           player_max_velocity=self.PLAYER_MAX_VELOCITY,
                           color=(1, 0, 0, 1),  # red
                           side=Side("left"),
                           player_number=self.number_of_player)

        self.team_B = Team(self.space, self.WIDTH, self.HEIGHT,
                           player_weight=self.PLAYER_WEIGHT,
                           player_max_velocity=self.PLAYER_MAX_VELOCITY,
                           color=(0, 0, 1, 1),  # blue
                           side=Side("right"),
                           player_number=self.number_of_player)

        self.player_arr = self.team_A.player_array + self.team_B.player_array

        # Ball
        self.ball = Ball(self.space, self.WIDTH * 0.5, self.HEIGHT * 0.5,
                         mass=self.BALL_WEIGHT,
                         max_velocity=self.BALL_MAX_VELOCITY,
                         elasticity=0.2)

        self.current_time = 0
        self.observation = self.reset()

    def reset(self):
        """reset the simulation"""
        self.current_time = 0
        self._set_position_to_initial()

        # after set position, need to step the space so that the object
        # move to the target position
        self.space.step(10**-7)
        self.observation = self._get_observation()

        return self.observation

    def render(self, mode='human'):
        padding = 5
        fig = plt.figure()
        ax = fig.add_subplot()
        ax = plt.axes(xlim=(0 - padding, self.WIDTH + padding),
                      ylim=(0 - padding, self.HEIGHT + padding))
        ax.set_aspect("equal")
        o = pymunk.matplotlib_util.DrawOptions(ax)
        self.space.debug_draw(o)
        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            plt.axis('off')
            dpi = 180
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.close()
            return img

    def _get_observation(self):
        """get observation"""
        if self.normalize_obs:
            ball_max_arr = np.array(
                [self.WIDTH + 3, self.HEIGHT, self.BALL_MAX_VELOCITY, self.BALL_MAX_VELOCITY])
            ball_min_arr = np.array(
                [-3, 0, -self.BALL_MAX_VELOCITY, -self.BALL_MAX_VELOCITY])

            player_max_arr = np.array(
                [self.WIDTH + 3, self.HEIGHT, self.PLAYER_MAX_VELOCITY, self.PLAYER_MAX_VELOCITY])
            player_min_arr = np.array(
                [-3, 0, -self.PLAYER_MAX_VELOCITY, -self.PLAYER_MAX_VELOCITY])

            ball_observation = normalize_array(
                np.array(self.ball.get_observation()), ball_max_arr, ball_min_arr)

            team_A_observation = normalize_array(
                self.team_A.get_observation(), player_max_arr, player_min_arr, self.number_of_player)
            team_B_observation = normalize_array(
                self.team_B.get_observation(), player_max_arr, player_min_arr, self.number_of_player)
        else:
            ball_observation = self.ball.get_observation()
            team_A_observation = self.team_A.get_observation()
            team_B_observation = self.team_B.get_observation()
        obs = np.concatenate(
            (ball_observation, team_A_observation, team_B_observation))
        return obs

    def _set_position_to_initial(self):
        """position ball and player to the initial position and set their velocity to 0"""
        self.team_A.set_position_to_initial()
        self.team_B.set_position_to_initial()
        self.ball.set_position(self.WIDTH * 0.5, self.HEIGHT * 0.5)

        # set the ball velocity to zero
        self.ball.body.velocity = 0, 0

    def step(self, left_player_action):
        # step the pymunk space a little so that the position to initial can work
        self.space.step(10 ** -7)

        right_player_action = np.reshape(self.action_space.sample(), (-1, 2))
        left_player_action = np.reshape(left_player_action, (-1, 2))

        init_distance_arr = self._ball_to_team_distance_arr(self.team_A)

        ball_init = self.ball.get_position()

        done = False
        reward = 0

        action_arr = np.concatenate((left_player_action, right_player_action))

        for player, action in zip(self.player_arr, action_arr):
            process_action(self, player, action)
            # change ball owner if any contact
            if self.ball.has_contact_with(player):
                self.ball.owner_side = player.side
            else:
                pass

        # fix the out of bound situation
        out = check_and_fix_out_bounds(self.ball, self.static, self.team_A, self.team_B)

        # step environment using pymunk
        self.space.step(self.TIME_STEP)
        self.observation = self._get_observation()

        # get reward
        if not out:
            ball_after = self.ball.get_position()

            reward += self.get_team_reward(init_distance_arr, self.team_A)
            reward += self.get_ball_reward(ball_init, ball_after)

        if ball_contact_goal(self.ball, self.static_goal):
            bx, _ = self.ball.get_position()

            goal_reward = 1000
            reward += goal_reward if bx > self.WIDTH - 2 else -goal_reward
            self._set_position_to_initial()
            if self.goal_end:
                done = True
            else:
                pass

        self.current_time += self.TIME_STEP

        if self.current_time > self.TOTAL_TIME:
            done = True

        return self.observation, reward, done, {}

    def _ball_to_team_distance_arr(self, team):
        distance_arr = []
        bx, by = self.ball.get_position()
        for player in team.player_array:
            px, py = player.get_position()
            distance_arr.append(math.sqrt((px-bx)**2 + (py-by)**2))
        return np.array(distance_arr)

    def get_team_reward(self, init_distance_arr, team):
        after_distance_arr = self._ball_to_team_distance_arr(team)
        difference_arr = init_distance_arr - after_distance_arr
        run_to_ball_reward_coefficient = 10

        if self.number_of_player == 5:
            return np.max([difference_arr[3], difference_arr[4]]) * run_to_ball_reward_coefficient
        else:
            return np.max(difference_arr) * run_to_ball_reward_coefficient

    def get_ball_reward(self, ball_init, ball_after):
        ball_to_goal_reward_coefficient = 10

        goal = [self.WIDTH, self.HEIGHT/2]

        _, ball_a_to_goal = get_vec(ball_after, goal)
        _, ball_i_to_goal = get_vec(ball_init, goal)

        return (ball_i_to_goal - ball_a_to_goal) * ball_to_goal_reward_coefficient
