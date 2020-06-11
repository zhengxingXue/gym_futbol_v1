import gym
import gym_futbol_v1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
from utils.video_utils import show_video, record_video, record_gif


def main(video=True):
    env = gym.make("futbol-v1")
    check_env(env, warn=True)
    model = PPO2(MlpPolicy, env, verbose=1)
    prefix = 'ppo2-futbol-pre'
    record_length = 300
    if video:
        record_video('futbol-v1', model, video_length=record_length, prefix=prefix)
        show_video('videos/' + prefix + '-step-0-to-step-' + str(record_length) + '.mp4')
    else:
        record_gif('futbol-v1', model, video_length=record_length, prefix=prefix)


if __name__ == "__main__":
    main()
    # show_video('videos/ppo2-futbol-pre-step-0-to-step-50.mp4')
