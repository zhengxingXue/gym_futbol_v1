import numpy as np


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


class SimpleAgent(BaseAgent):
    def __init__(self, env, side):
        super().__init__(env, side)

    def predict(self, obs, state=None):
        pass
