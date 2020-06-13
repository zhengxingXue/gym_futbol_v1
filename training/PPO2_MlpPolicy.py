import time
import os
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from utils.train_utils import create_n_env, create_eval_callback
import tensorflow
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def ppo2_training_mlp(env_id='futbol-v1', env_n=8, time_step=10**5, save_dir_prefix='./training/logs/MlpPolicy'):
    """
    :param env_id: id of the environment
    :param env_n: number of env for the PPO2 model
    :param time_step: total time step for training
    :param save_dir_prefix: prefix of directory to save trained model and best model
    :return: (PPO2, str)the trained ppo2 model, and the save directory
    """
    time_str = "{}".format(int(time.time()))
    log_dir = "./training/tmp/MlpPolicy-" + time_str
    os.makedirs(log_dir, exist_ok=True)
    env = create_n_env(env_id, log_dir, env_n)

    save_dir = save_dir_prefix + '-' + time_str
    eval_callback = create_eval_callback(env_id, save_dir=save_dir)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=time_step, callback=eval_callback)
    model.save(save_dir + '/' + "ppo2_mlp")
    return model, save_dir


if __name__ == "__main__":
    _ = ppo2_training_mlp()
