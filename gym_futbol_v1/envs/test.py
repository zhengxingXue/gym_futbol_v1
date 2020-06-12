from gym_futbol_v1.envs.futbol_env import Futbol
import numpy as np

env = Futbol(normalize_obs=False)
env.render()
obs = env.observation
reshape_obs = obs.reshape((-1, 4))
print(reshape_obs)
