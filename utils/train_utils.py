import time
import os
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import EvalCallback
import gym
import numpy as np
from gym import spaces
from gym_futbol_v1.envs import Side
from stable_baselines import PPO2
import tensorflow
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class CustomPolicy2v2(FeedForwardPolicy):
    """
    Custom MLP policy for 2v2
    """

    def __init__(self, *args, **kwargs):
        super(CustomPolicy2v2, self).__init__(*args, **kwargs,
                                              net_arch=[256, 256, dict(pi=[128, 128],
                                                                       vf=[128, 128])],
                                              feature_extraction="mlp")


def create_custom_policy(net_arch):
    """
    function to create custom policy
    eg. net_arch = [64, 64]
    eg. net_arch = [64, dict(pi=[64],vf=[64])]
    """

    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=net_arch, feature_extraction="mlp")

    return CustomPolicy


def create_n_env(env, log_dir='./temp', num_envs=8):
    """
    :param env: (str or gym.env) if env is string use gym.make() else directly use env
    :param log_dir: log directory
    :param num_envs: number of environment
    :return: DummyVecEnv for training
    """
    if isinstance(env, str):
        env = gym.make(env)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env] * num_envs)
    return env


def create_eval_callback(env_id, save_dir='./logs', eval_freq=1000, n_eval_episodes=10):
    """
    :param env_id: 'futbol-v1'
    :param save_dir: the directory to save the best model
    :param eval_freq: the frequency of the evaluation callback
    :param n_eval_episodes: the number  of evaluation of each callback
    :return: EvalCallback for training
    """
    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir,
                                 log_path=save_dir,
                                 eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
                                 deterministic=False, render=False)
    return eval_callback


def ppo2_train(policy, policy_name, env_id='futbol-v1', env_n=8, time_step=10**5,
               save_dir_prefix='./training/logs', log_dir_prefix='./training/tmp',
               enable_call_back=True, verbose=1):
    """
    :param policy: stable-baseline policy
    :param policy_name: string of policy name
    :param env_id: id of the environment
    :param env_n: number of env for the PPO2 model
    :param time_step: total time step for training
    :param save_dir_prefix: prefix of directory to save trained model and best model
    :param log_dir_prefix: prefix of log directory
    :param enable_call_back: whether to use the eval call back
    :param verbose: whether to show training info
    :return: (PPO2, str)the trained ppo2 model, and the save directory
    """
    time_str = "{}".format(int(time.time()))
    log_dir = log_dir_prefix + "/" + policy_name + '-' + time_str
    os.makedirs(log_dir, exist_ok=True)
    env = create_n_env(env_id, log_dir, env_n)

    save_dir = save_dir_prefix + '/' + policy_name + '-' + time_str
    os.makedirs(save_dir, exist_ok=True)

    model = PPO2(policy, env, verbose=verbose)
    if enable_call_back:
        eval_callback = create_eval_callback(env_id, save_dir=save_dir)
        model.learn(total_timesteps=time_step, callback=eval_callback)
    else:
        model.learn(total_timesteps=time_step)

    model_path = save_dir + '/' + policy_name
    model.save(model_path)
    print("model saved to " + model_path + '.zip')
    return model, save_dir


class NormalizeObservationWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        env.observation_space = spaces.Box(
            low=np.array([-1., -1., -1., -1.] *
                         (1 + env.number_of_player * 2), dtype=np.float32),
            high=np.array([1., 1., 1., 1.] *
                          (1 + env.number_of_player * 2), dtype=np.float32),
            dtype=np.float32)
        # Call the parent constructor, so we can access self.env later
        super(NormalizeObservationWrapper, self).__init__(env)

        ball_max_arr = np.array(
            [env.WIDTH + 3, env.HEIGHT, env.BALL_MAX_VELOCITY, env.BALL_MAX_VELOCITY])
        ball_min_arr = np.array(
            [-3, 0, -env.BALL_MAX_VELOCITY, -env.BALL_MAX_VELOCITY])

        player_max_arr = np.array(
            [env.WIDTH + 3, env.HEIGHT, env.PLAYER_MAX_VELOCITY, env.PLAYER_MAX_VELOCITY])
        player_min_arr = np.array(
            [-3, 0, -env.PLAYER_MAX_VELOCITY, -env.PLAYER_MAX_VELOCITY])

        max_arr = np.concatenate((ball_max_arr, np.tile(player_max_arr, self.number_of_player * 2)))
        min_arr = np.concatenate((ball_min_arr, np.tile(player_min_arr, self.number_of_player * 2)))

        self.avg_arr = (max_arr + min_arr) / 2
        self.range_arr = (min_arr - max_arr) / 2

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        obs = (obs - self.avg_arr) / self.range_arr
        return obs

    def step(self, action, team_side=Side.left):
        """
        :param action: ([float] or int) Action taken by the agent
        :param team_side: team side for the action
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action, team_side)
        obs = (obs - self.avg_arr) / self.range_arr
        # reward /= 1000
        return obs, reward, done, info


class AddHeightWidthObservationWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        self.width_height_array = np.array([env.WIDTH, env.HEIGHT])
        low = np.concatenate((env.obs_low, self.width_height_array))
        high = np.concatenate((env.obs_high, self.width_height_array))
        env.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Call the parent constructor, so we can access self.env later
        super(AddHeightWidthObservationWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        obs = np.concatenate((obs, self.width_height_array))
        return obs

    def step(self, action, team_side=Side.left):
        """
        :param action: ([float] or int) Action taken by the agent
        :param team_side: team side for the action
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action, team_side)
        obs = np.concatenate((obs, self.width_height_array))

        return obs, reward, done, info
