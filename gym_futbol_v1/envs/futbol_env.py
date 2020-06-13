"""
Env Module.
"""
import io
import cv2
import gym
from gym import spaces
import numpy as np
import pymunk
import pymunk.matplotlib_util
import matplotlib.pyplot as plt
from gym_futbol_v1.envs.team import Team
from gym_futbol_v1.envs.helper import setup_walls, normalize_array, Ball, check_and_fix_out_bounds, \
    ball_contact_goal
from gym_futbol_v1.envs.action import process_action
from gym_futbol_v1.envs import Reward, Side, BaseAgent


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

    WIDTH = 105  # [m]
    HEIGHT = 68  # [m]
    GOAL_SIZE = 20  # [m]

    TOTAL_TIME = 30  # [s]
    TIME_STEP = 0.1  # [s]

    BALL_MAX_VELOCITY = 25  # [m/s]
    PLAYER_MAX_VELOCITY = 10  # [m/s]

    BALL_WEIGHT = 10
    PLAYER_WEIGHT = 20

    PLAYER_FORCE_LIMIT = 40
    BALL_FORCE_LIMIT = 120

    def __init__(self, debug=False, number_of_player=2, goal_end=False):

        self.debug = debug
        self.number_of_player = number_of_player
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
        self.obs_low = np.array([-3, 0, -self.BALL_MAX_VELOCITY, -self.BALL_MAX_VELOCITY] +
                                [-3, 0, -self.PLAYER_MAX_VELOCITY, -self.PLAYER_MAX_VELOCITY] *
                                (self.number_of_player * 2), dtype=np.float32)
        self.obs_high = np.array([self.WIDTH + 3, self.HEIGHT, self.BALL_MAX_VELOCITY, self.BALL_MAX_VELOCITY] +
                                 [self.WIDTH + 3, self.HEIGHT, self.PLAYER_MAX_VELOCITY, self.PLAYER_MAX_VELOCITY] *
                                 (self.number_of_player * 2), dtype=np.float32)

        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

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
                           side=Side.left,
                           player_number=self.number_of_player)

        self.team_B = Team(self.space, self.WIDTH, self.HEIGHT,
                           player_weight=self.PLAYER_WEIGHT,
                           player_max_velocity=self.PLAYER_MAX_VELOCITY,
                           color=(0, 0, 1, 1),  # blue
                           side=Side.right,
                           player_number=self.number_of_player)

        # Agents
        self.team_A_agent = BaseAgent(self, Side.left)
        self.team_B_agent = BaseAgent(self, Side.right)

        self.player_arr = self.team_A.player_array + self.team_B.player_array

        # Ball
        self.ball = Ball(self.space, self.WIDTH * 0.5, self.HEIGHT * 0.5,
                         mass=self.BALL_WEIGHT,
                         max_velocity=self.BALL_MAX_VELOCITY,
                         elasticity=0.2)

        self.current_time = 0
        self.observation = self.reset()
        self.reward_class = Reward(self)

    def reset(self):
        """reset the simulation"""
        self.current_time = 0
        self._set_position_to_initial()

        # after set position, need to step the space so that the object
        # move to the target position
        self.space.step(10 ** -7)
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

    def step(self, team_action, team_side=Side.left):
        # left = team_A
        # right = team_B
        # need further modification
        if team_side == Side.left:
            left_player_action = team_action
            team = self.team_A
            right_player_action, _ = self.team_B_agent.predict(self.observation)
        else:
            right_player_action = team_action
            team = self.team_B
            left_player_action, _ = self.team_A_agent.predict(self.observation)

        init_distance_arr = self.reward_class.ball_to_team_distance_arr(team)

        ball_init = self.ball.get_position()

        done = False
        reward = 0

        action_arr = np.concatenate((left_player_action, right_player_action)).reshape((-1, 2))

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

            reward += self.reward_class.get_team_reward(init_distance_arr, team)
            reward += self.reward_class.get_ball_to_goal_reward(team_side, ball_init, ball_after)

        if ball_contact_goal(self.ball, self.static_goal):
            reward += self.reward_class.get_goal_reward(team_side)
            self._set_position_to_initial()
            if self.goal_end:
                done = True
            else:
                pass

        self.current_time += self.TIME_STEP

        if self.current_time > self.TOTAL_TIME:
            done = True

        return self.observation, reward, done, {}

    def _get_observation(self):
        """get observation"""
        ball_observation = self.ball.get_observation()
        team_A_observation = self.team_A.get_observation()
        team_B_observation = self.team_B.get_observation()
        # flatten obs
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
