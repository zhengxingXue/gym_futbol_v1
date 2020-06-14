import argparse
from utils import ppo2_multi_agent_train
from stable_baselines.common.policies import MlpPolicy
import tensorflow
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)


def ppo2_mlp_multi_agent_train(env_id='futbol-v1', env_n=8, time_step=10 ** 4,
                               num_turn=10, save_dir_prefix='./training/logs',
                               verbose=0):
    """
    :param env_id: id of the environment
    :param env_n: number of env for the PPO2 model
    :param time_step: total time step for training
    :param num_turn: number of time to turn
    :param save_dir_prefix: prefix of directory to save trained model and best model
    :param verbose: whether to show training info
    :return: (PPO2, str)the trained ppo2 model, and the save directory
    """
    policy = MlpPolicy
    policy_name = 'ppo2-mlp'
    model_left, model_right, save_dir = ppo2_multi_agent_train(policy, policy_name, env_id=env_id, env_n=env_n,
                                                               time_step=time_step, num_turn=num_turn,
                                                               save_dir_prefix=save_dir_prefix, verbose=verbose)
    return model_left, model_right, save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='multi agent training with PPO2(MLP policy).')
    parser.add_argument('--timestep', help='set time step of each training process', type=int, default=10 ** 4)
    parser.add_argument('--turn', help='set turn number of training process', type=int, default=10)
    args = parser.parse_args()
    _ = ppo2_mlp_multi_agent_train(time_step=args.timestep, num_turn=args.turn)
