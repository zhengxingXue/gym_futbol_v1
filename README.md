# gym_futbol_v1

## Overview
This project contiunes the work on gym-futbol(https://github.com/yc2454/gym-futbol).

## Quick start

To install the Python module:
```
$ git clone https://github.com/zhengxingXue/gym_futbol_v1.git
$ cd gym_futbol_v1
$ pip install -e .
```

To try the environment with PPO2 agent (without any training):
```
$ python main.py
```
The simulation is recorded and showed in the pop up window. The video file is saved in ``videos/`` directory.

## Env

To create the ``futbol`` environment:
```
import gym, gym_futbol_v1
env = gym.make('futbol-v1')  # 2v2 settings
env.render()
```
