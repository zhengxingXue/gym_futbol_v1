import random
from gym_futbol_v1.envs import Side


class BaseAgent:
    def __init__(self, env, side):
        self.env = env
        self.side = side
        self.n_envs = 1

    def predict(self, obs, state=None):
        """
        Generate random action
        based on PPO2.predict()
        Return [action], next state(unimplemented)
        """
        return [self.env.action_space.sample()], None
