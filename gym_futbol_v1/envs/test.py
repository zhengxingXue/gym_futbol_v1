from gym_futbol_v1.envs.futbol_env import Futbol
import numpy as np

env = Futbol(normalize_obs=False)
env.render()
obs = env.observation
reshape_obs = obs.reshape((-1, 4))
print(0)
print(reshape_obs)

for i in range(60):
    obs, reward, _, _ = env.step([0, 3, 0, 0])
    reshape_obs = obs.reshape((-1, 4))
    print(i+1)
    print(reshape_obs)
    # env.render()

for i in range(30):
    obs, reward, _, _ = env.step([0, 2, 0, 0])
    reshape_obs = obs.reshape((-1, 4))
    print(i+1)
    print(reshape_obs)
    env.render()
