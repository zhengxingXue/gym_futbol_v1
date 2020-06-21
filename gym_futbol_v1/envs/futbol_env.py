"""
Env Module.
"""
import io
import random
import cv2
import gym
from gym import spaces
import numpy as np
import pymunk
import pymunk.matplotlib_util
import matplotlib.pyplot as plt
from matplotlib import patches
from gym_futbol_v1.envs.team import Team
from gym_futbol_v1.envs.helper import setup_walls, Ball, check_and_fix_out_bounds, ball_contact_goal, draw_box
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

    WIDTH = 120  # [m]
    HEIGHT = 81  # [m]
    GOAL_SIZE = 16  # [m]

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
        self.team_left = Team(self.space, self.WIDTH, self.HEIGHT,
                              player_weight=self.PLAYER_WEIGHT,
                              player_max_velocity=self.PLAYER_MAX_VELOCITY,
                              color=(1, 0, 0, 1),  # red
                              side=Side.left,
                              player_number=self.number_of_player)

        self.team_right = Team(self.space, self.WIDTH, self.HEIGHT,
                               player_weight=self.PLAYER_WEIGHT,
                               player_max_velocity=self.PLAYER_MAX_VELOCITY,
                               color=(0, 0, 1, 1),  # blue
                               side=Side.right,
                               player_number=self.number_of_player)

        # Agents
        self.team_left_agent = BaseAgent(self, Side.left)
        self.team_right_agent = BaseAgent(self, Side.right)

        self.player_arr = self.team_left.player_array + self.team_right.player_array

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

        # reset team score
        self.team_left.reset_score()
        self.team_right.reset_score()

        return self.observation

    def render(self, mode='human', axis=False, label=False, notebook=False):
        padding = 5
        if notebook:
            pass  # for notebook render
        else:
            fig = plt.figure()
            ax = fig.add_subplot()
        ax = plt.axes(xlim=(0 - padding, self.WIDTH + padding),
                      ylim=(0 - padding, self.HEIGHT + padding))
        ax.set_aspect("equal")

        color = '0.9'
        # center line and circle
        circle = plt.Circle((self.WIDTH / 2, self.HEIGHT / 2), radius=9,
                            fill=False, color=color, linewidth=1, zorder=0)
        ax.add_artist(circle)
        plt.plot([self.WIDTH / 2, self.WIDTH / 2], [0, self.HEIGHT],
                 color=color, linestyle='-', linewidth=1, zorder=0)

        # left box
        draw_box(0, self.HEIGHT / 2, 16.5, 20, color)
        draw_box(0, self.HEIGHT / 2, 5.5, self.GOAL_SIZE / 2 + 2, color)
        left_arc = patches.Arc((11, self.HEIGHT / 2), 18, 18,
                               angle=0, theta1=-50, theta2=50, color=color, zorder=0)
        ax.add_patch(left_arc)

        # right box
        draw_box(self.WIDTH, self.HEIGHT / 2, -16.5, 20, color)
        draw_box(self.WIDTH, self.HEIGHT / 2, -5.5, self.GOAL_SIZE / 2 + 2, color)
        right_arc = patches.Arc((self.WIDTH - 11, self.HEIGHT / 2), 18, 18,
                                angle=180, theta1=-50, theta2=50, color=color, zorder=0)
        ax.add_patch(right_arc)

        if label:
            self.team_left.label_player()
            self.team_right.label_player()

        # draw player and ball
        o = pymunk.matplotlib_util.DrawOptions(ax)
        o.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        o.shape_outline_color = (0, 0, 0, 0.3)
        self.space.debug_draw(o)

        if not axis:
            plt.axis('off')

        if mode == 'human':
            if notebook:
                pass
            else:
                plt.show()
        elif mode == 'rgb_array':
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

    def step(self, team_action, team_side=Side.left, opponent_state=None):
        done = False
        reward = 0

        # left = team_left
        # right = team_right
        # need further modification
        if team_side == Side.left:
            left_player_action = team_action
            team = self.team_left
            n_env = self.team_right_agent.n_envs
            stacked_obs = np.tile(self.observation, (n_env, 1))
            right_player_action, opponent_state = self.team_right_agent.predict(stacked_obs, state=opponent_state)
            right_player_action = right_player_action[0]
        else:
            right_player_action = team_action
            team = self.team_right
            n_env = self.team_left_agent.n_envs
            stacked_obs = np.tile(self.observation, (n_env, 1))
            left_player_action, opponent_state = self.team_left_agent.predict(stacked_obs, state=opponent_state)
            left_player_action = left_player_action[0]

        # TODO: ADD actions to info
        info = {'opponent state': opponent_state}

        init_distance_arr = self.reward_class.ball_to_team_distance_arr(team)

        ball_init = self.ball.get_position()

        action_arr = np.concatenate((left_player_action, right_player_action)).reshape((-1, 2))

        ball_has_contact = False
        # random shuffle the player and action pair to make the game more fair
        player_action_list = list(zip(self.player_arr, action_arr))
        random.shuffle(player_action_list)
        for player, action in player_action_list:
            process_action(self, player, action)
            # change ball owner if any contact
            if self.ball.has_contact_with(player):
                self.ball.change_owner_side(player.side)
                ball_has_contact = True
            else:
                pass

        if not ball_has_contact:
            self.ball.change_owner_side(Side.NoSide)

        # fix the out of bound situation
        out = check_and_fix_out_bounds(self.ball, self.static, self.team_left, self.team_right)

        # step environment using pymunk
        self.space.step(self.TIME_STEP)
        self.observation = self._get_observation()

        # get reward
        reward += self.reward_class.get_contact_ball_reward(side=team_side)
        if not out:
            ball_after = self.ball.get_position()

            reward += self.reward_class.get_team_reward(init_distance_arr, team)
            reward += self.reward_class.get_ball_to_goal_reward(team_side, ball_init, ball_after)

        if ball_contact_goal(self.ball, self.static_goal):
            reward += self.reward_class.get_goal_reward(team_side)
            bx, _ = self.ball.get_position()
            if bx > self.WIDTH - 2:
                self.team_left.add_score()
            else:
                self.team_right.add_score()

            # reset the player position and step a little
            # self.observation is already set above
            # might cause problem as the actual position of the goal frame is the initial position
            # while the returned observation is before the initialization
            self._set_position_to_initial()
            if self.goal_end:
                done = True
            else:
                pass

        self.current_time += self.TIME_STEP

        if self.current_time >= self.TOTAL_TIME:
            done = True

        return self.observation, reward, done, info

    def _get_observation(self):
        """get observation"""
        ball_observation = self.ball.get_observation()
        team_left_observation = self.team_left.get_observation()
        team_right_observation = self.team_right.get_observation()
        # flatten obs
        obs = np.concatenate(
            (ball_observation, team_left_observation, team_right_observation))
        return obs

    def _set_position_to_initial(self):
        """position ball and player to the initial position and set their velocity to 0"""
        self.team_left.set_position_to_initial()
        self.team_right.set_position_to_initial()
        self.ball.set_position(self.WIDTH * 0.5, self.HEIGHT * 0.5)

        # owner side reset
        self.ball.owner_side = Side.NoSide
        self.ball.last_owner_side = Side.NoSide

        # set the ball velocity to zero
        self.ball.body.velocity = 0, 0

        self.space.step(self.TIME_STEP)

    def get_score(self):
        """
        :return: score string
        """
        return "Team left : Team right = " + str(self.team_left.score) + " : " + str(self.team_right.score)


if __name__ == "__main__":
    env = Futbol(number_of_player=3)
    env.render(label=True)
