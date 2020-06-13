from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import EvalCallback
import gym
import numpy as np
from gym import spaces
from gym_futbol_v1.envs import Side


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


def create_n_env(env_id, log_dir='./temp', num_envs=8):
    """
    :param env_id: 'futbol-v1'
    :param log_dir: log directory
    :param num_envs: number of environment
    :return: DummyVecEnv for training
    """
    env = gym.make(env_id)
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


class NormalizeObservationWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        env.observation_space = env.get_normalized_observation_space()
        # Call the parent constructor, so we can access self.env later
        super(NormalizeObservationWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        obs = self.env.normalize_observation_array(obs)
        return obs

    def step(self, action, team_side=Side.left):
        """
        :param action: ([float] or int) Action taken by the agent
        :param team_side: team side for the action
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action, team_side)
        obs = self.env.normalize_observation_array(obs)
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
