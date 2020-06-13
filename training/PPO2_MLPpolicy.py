import time
import os
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from utils.train_utils import create_n_env, create_eval_callback


def main(time_step=10**5):
    log_dir = "./tmp/MlpPolicy/{}".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)
    env = create_n_env('futbol-v1', log_dir, 8)
    eval_callback = create_eval_callback('futbol-v1', save_dir='./logs/MlpPolicy')

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=time_step, callback=eval_callback)


if __name__ == "__main__":
    main()
