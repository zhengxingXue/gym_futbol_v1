import math
import numpy as np
from gym_futbol_v1.envs.helper import Side, get_vec


class Reward:
    def __init__(self, env, run_to_ball_reward_coefficient=10,  ball_to_goal_reward_coefficient=10):
        self.run_to_ball_reward_coefficient = run_to_ball_reward_coefficient
        self.ball_to_goal_reward_coefficient = ball_to_goal_reward_coefficient
        self.env = env

    def get_ball_to_goal_reward(self, side, ball_init, ball_after):
        """
        get the reward for ball move to the goal position (based on side)
        """
        if side == Side.left:
            goal = [self.env.WIDTH, self.env.HEIGHT/2]
        else:
            goal = [0, self.env.HEIGHT/2]

        _, ball_a_to_goal = get_vec(ball_after, goal)
        _, ball_i_to_goal = get_vec(ball_init, goal)

        return (ball_i_to_goal - ball_a_to_goal) * self.ball_to_goal_reward_coefficient

    def ball_to_team_distance_arr(self, team):
        distance_arr = []
        bx, by = self.env.ball.get_position()
        for player in team.player_array:
            px, py = player.get_position()
            distance_arr.append(math.sqrt((px-bx)**2 + (py-by)**2))
        return np.array(distance_arr)

    def get_team_reward(self, init_distance_arr, team):
        after_distance_arr = self.ball_to_team_distance_arr(team)
        difference_arr = init_distance_arr - after_distance_arr

        if self.env.number_of_player == 5:
            return np.max([difference_arr[3], difference_arr[4]]) * self.run_to_ball_reward_coefficient
        else:
            return np.max(difference_arr) * self.run_to_ball_reward_coefficient
