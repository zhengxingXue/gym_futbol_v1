import random
from gym_futbol_v1.envs import Side


class BaseAgent:
    def __init__(self, env, side):
        self.env = env
        self.side = side

    def predict(self, obs):
        """
        Generate random action
        based on PPO2.predict()
        Return action, next state(unimplemented)
        """
        return self.env.action_space.sample(), None
