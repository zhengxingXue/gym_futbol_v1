import argparse

from stable_baselines.common.policies import MlpPolicy
from utils import ppo2_train
import tensorflow
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def ppo2_mlp_policy_train(env_id='futbol-v1', env_n=8, time_step=10**4, save_dir_prefix='./training/logs',
                          verbose=1):
    """
    :param env_id: id of the environment
    :param env_n: number of env for the PPO2 model
    :param time_step: total time step for training
    :param save_dir_prefix: prefix of directory to save trained model and best model
    :param verbose: whether to show training info
    :return: (PPO2, str)the trained ppo2 model, and the save directory
    """
    policy = MlpPolicy
    policy_name = 'ppo2-mlp'
    model, save_dir = ppo2_train(policy, policy_name, env_id, env_n, time_step, save_dir_prefix, verbose=verbose)
    return model, save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ppo2 with MlpPolicy training against random opponents.')
    parser.add_argument('--timestep', help='set time step of the training process', type=int, default=10 ** 5)
    args = parser.parse_args()
    _ = ppo2_mlp_policy_train(time_step=args.timestep)
