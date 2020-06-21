import numpy as np
from gym_futbol_v1.envs.helper import Side
from gym_futbol_v1.envs.team import PlayerType

action_dict = {
    'noop': [0, 0],
    'press': [0, 3],
    'shoot': [0, 2],
    'up': [1, 0], 'right': [2, 0], 'down': [3, 0], 'left': [4, 0],
    'up-dash': [1, 1], 'right-dash': [2, 1], 'down-dash': [3, 1], 'left-dash': [4, 1],
    'up-pass': [1, 4], 'right-pass': [2, 4], 'down-pass': [3, 4], 'left-pass': [4, 4]
}


class BaseAgent:
    def __init__(self, env, side):
        self.env = env
        self.side = side
        self.n_envs = 1
        self.n_player = env.number_of_player
        self.player_array = env.team_left.player_array if side == Side.left else env.team_right.player_array

    def predict(self, obs, state=None):
        """
        Generate random action
        based on PPO2.predict()
        Return [action], next state(unimplemented)
        """
        return [self.env.action_space.sample()], None


class SimpleAgent(BaseAgent):
    def __init__(self, env, side):
        super().__init__(env, side)

    def predict(self, obs, state=None):

        observation_array = obs[0].reshape((-1, 4))

        ball_obs = observation_array[0]
        left_obs = observation_array[1:1 + self.n_player]
        right_obs = observation_array[1 + self.n_player:]

        action = []

        for player in self.player_array:
            if player.type == PlayerType.forward:
                if player == self.env.has_ball_player:
                    x, _ = player.get_position()
                    if self.side == Side.right:
                        if x <= self.env.WIDTH * 0.2:
                            action_key_str = 'shoot'
                        else:
                            action_key_str = 'left-dash'
                    else:
                        if x >= self.env.WIDTH * 0.8:
                            action_key_str = 'shoot'
                        else:
                            action_key_str = 'right-dash'
                else:
                    action_key_str = 'press'
            else:
                action_key_str = 'noop'
            action += action_dict[action_key_str]

        # print(action)

        return [np.array(action)], None
