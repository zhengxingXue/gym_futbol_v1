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
from .object import Ball
from .team import Team
from .helper import get_vec, setup_walls, normalize_array


class Futbol(gym.Env):

    """
    Futbol is a 2D simulation of real world soccer.
    Soccer is a team sport played with a spherical ball between two teams of 11 players,
    including a goalkeeper. The goal is to get the ball into the net. For this simulation,
    user can specifiy the player number and the goalkeeper is not implemented for now.

    **STATE:**
    The state consists of position and velocity of each player and the ball.
    The state is flattened and normalized for better RL performance.

    **ACTIONS:**
    The action consists of arrow key and action key, inspired by PES2020.
    Arrow key controls the player action direction.
    Action key contorls the action one player take at the moment.
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

    def __init__(self, debug=False, number_of_player=2, normalize_obs=True):

        self.debug = debug
        self.number_of_player = number_of_player
        self.normalize_obs = normalize_obs

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
                           side="left",
                           player_number=self.number_of_player)

        self.team_B = Team(self.space, self.WIDTH, self.HEIGHT,
                           player_weight=self.PLAYER_WEIGHT,
                           player_max_velocity=self.PLAYER_MAX_VELOCITY,
                           color=(0, 0, 1, 1),  # blue
                           side="right",
                           player_number=self.number_of_player)

        self.player_arr = self.team_A.player_array + self.team_B.player_array

        # Ball
        self.ball = Ball(self.space, self.WIDTH * 0.5, self.HEIGHT * 0.5,
                         mass=self.BALL_WEIGHT,
                         max_velocity=self.BALL_MAX_VELOCITY,
                         elasticity=0.2)

        self.observation = self.reset()

    def reset(self):
        """reset the simulation"""
        self.current_time = 0
        self.ball_owner_side = random.choice(["left", "right"])
        self._position_to_initial()
        return self._get_observation()

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

    def _position_to_initial(self):
        """position ball and player to the initial position and set their velocity to 0"""
        self.team_A.set_position_to_initial()
        self.team_B.set_position_to_initial()
        self.ball.set_position(self.WIDTH * 0.5, self.HEIGHT * 0.5)

        # set the ball velocity to zero
        self.ball.body.velocity = 0, 0

        # after set position, need to step the space so that the object
        # move to the target position
        self.space.step(0.0001)
        self.observation = self._get_observation()

    def ball_contact_wall(self):
        """
        return true and wall index if the ball is in contact with the walls
        """
        wall_index, i = -1, 0
        for wall in self.static:
            if self.ball.shape.shapes_collide(wall).points != []:
                wall_index = i
                return True, wall_index
            i += 1
        return False, wall_index

    def check_and_fix_out_bounds(self):
        out, wall_index = self.ball_contact_wall()
        if out:
            bx, by = self.ball.get_position()
            dbx, dby, dpx, dpy = 0, 0, 0, 0

            if wall_index == 1 or wall_index == 0:  # left bound
                dbx, dpx = 3.5, 1
            elif wall_index == 3 or wall_index == 4:
                dbx, dpx = -3.5, -1
            elif wall_index == 2:
                dby, dpy = -3.5, -1
            else:
                dby, dpy = 3.5, 1

            self.ball.set_position(bx + dbx, by + dby)
            self.ball.body.velocity = 0, 0

            if self.ball_owner_side == "right":
                get_ball_player = random.choice(self.team_A.player_array)
                self.ball_owner_side = "left"
            elif self.ball_owner_side == "left":
                get_ball_player = random.choice(self.team_B.player_array)
                self.ball_owner_side = "right"
            else:
                print("invalid side")

            get_ball_player.set_position(bx + dpx, by + dpy)
            get_ball_player.body.velocity = 0, 0
        else:
            pass
        return out

    def ball_contact_goal(self):
        """
        return true if score
        """
        goal = False
        for goal_wall in self.static_goal:
            goal = goal or self.ball.shape.shapes_collide(
                goal_wall).points != []
        return goal

    def _ball_move_with_player(self, player):
        """
        if player has contact with ball and move, let the ball move with the player.
        """
        if self.ball.has_contact_with(player):
            self.ball.body.velocity = player.body.velocity
        else:
            pass

    def random_action(self): return self.action_space.sample()

    def _process_action(self, player, action):
        # Arrow Keys: NOOP
        if action[0] == 0:
            force_x, force_y = 0, 0
        # Arrow Keys: UP
        elif action[0] == 1:
            force_x, force_y = 0, 1
        # Arrow Keys: RIGHT
        elif action[0] == 2:
            force_x, force_y = 1, 0
        # Arrow Keys: DOWN
        elif action[0] == 3:
            force_x, force_y = 0, -1
        # Arrow Keys: LEFT
        elif action[0] == 4:
            force_x, force_y = -1, 0
        else:
            print("invalid arrow keys")

        # Action keys
        # noop [0]
        if action[1] == 0:
            player.apply_force_to_player(self.PLAYER_WEIGHT * force_x,
                                         self.PLAYER_WEIGHT * force_y)

            self._ball_move_with_player(player)

        # dash [1]
        elif action[1] == 1:
            player.apply_force_to_player(self.PLAYER_FORCE_LIMIT * force_x,
                                         self.PLAYER_FORCE_LIMIT * force_y)
            self._ball_move_with_player(player)

        # shoot [2]
        elif action[1] == 2:
            if self.ball.has_contact_with(player):
                if player.side == "left":
                    goal = [self.WIDTH, self.HEIGHT/2]
                elif player.side == "right":
                    goal = [0, self.HEIGHT/2]
                else:
                    print("invalid side")

                ball_pos = self.ball.get_position()
                ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                    goal, ball_pos)

                ball_force_x = self.BALL_FORCE_LIMIT * \
                    ball_to_goal_vec[0] / ball_to_goal_vec_mag
                ball_force_y = self.BALL_FORCE_LIMIT * \
                    ball_to_goal_vec[1] / ball_to_goal_vec_mag

                # decrease the velocity influence on shoot
                self.ball.body.velocity /= 2

                self.ball_owner_side = player.side
                self.ball.apply_force_to_ball(ball_force_x, ball_force_y)
            else:
                pass

        # press [3]
        elif action[1] == 3:
            # cannot press with ball
            if self.ball.has_contact_with(player):
                pass
            # no ball, no arrow keys, run to ball (press)
            elif action[0] == 0:
                ball_pos = self.ball.get_position()
                player_pos = player.get_position()

                player_to_ball_vec, player_to_ball_vec_mag = get_vec(
                    ball_pos, player_pos)

                player_force_x = self.PLAYER_FORCE_LIMIT * \
                    player_to_ball_vec[0] / player_to_ball_vec_mag
                player_force_y = self.PLAYER_FORCE_LIMIT * \
                    player_to_ball_vec[1] / player_to_ball_vec_mag

                player.apply_force_to_player(player_force_x, player_force_y)
            # no ball, arrow keys pressed, run as the arrow key
            else:
                pass

        # pass [4]
        elif action[1] == 4:
            if self.ball.has_contact_with(player):
                team = self.team_A if player.side == "left" else self.team_B

                target_player = team.get_pass_target_teammate(
                    player, arrow_keys=action[0])

                goal = target_player.get_position()

                ball_pos = self.ball.get_position()
                ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                    goal, ball_pos)

                ball_force_x = (self.BALL_FORCE_LIMIT - 20) * \
                    ball_to_goal_vec[0] / ball_to_goal_vec_mag
                ball_force_y = (self.BALL_FORCE_LIMIT - 20) * \
                    ball_to_goal_vec[1] / ball_to_goal_vec_mag

                # decrease the velocity influence on pass
                self.ball.body.velocity /= 10

                self.ball_owner_side = player.side
                self.ball.apply_force_to_ball(ball_force_x, ball_force_y)
            # cannot pass ball without ball
            else:
                pass

        else:
            print("invalid action key")

    def step(self, left_player_action):
        right_player_action = np.reshape(self.random_action(), (-1, 2))
        left_player_action = np.reshape(left_player_action, (-1, 2))

        init_distance_arr = self._ball_to_team_distance_arr(self.team_A)

        ball_init = self.ball.get_position()

        done = False
        reward = 0

        action_arr = np.concatenate((left_player_action, right_player_action))

        for player, action in zip(self.player_arr, action_arr):
            self._process_action(player, action)
            # change ball owner if any contact
            if self.ball.has_contact_with(player):
                self.ball_owner_side = player.side
            else:
                pass

        # fix the out of bound situation
        out = self.check_and_fix_out_bounds()

        # step environment using pymunk
        self.space.step(self.TIME_STEP)
        self.observation = self._get_observation()

        # get reward
        if not out:
            ball_after = self.ball.get_position()

            reward += self.get_team_reward(init_distance_arr, self.team_A)
            reward += self.get_ball_reward(ball_init, ball_after)

        if self.ball_contact_goal():
            bx, _ = self.ball.get_position()

            goal_reward = 1000
            reward += goal_reward if bx > self.WIDTH - 2 else -goal_reward
            self._position_to_initial()
            self.ball_owner_side = random.choice(["left", "right"])
            # done = True

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
