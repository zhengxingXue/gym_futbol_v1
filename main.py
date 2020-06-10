import gym
import gym_futbol_v1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
from utils.video_utils import show_video, record_video


def main():
    env = gym.make("futbol-v1")
    check_env(env, warn=True)
    model = PPO2(MlpPolicy, env, verbose=1)
    prefix = 'ppo2-futbol-pre'
    video_length = 50
    record_video('futbol-v1', model, video_length=video_length, prefix=prefix)
    show_video('videos/' + prefix + '-step-0-to-step-' + str(video_length) + '.mp4')


if __name__ == "__main__":
    # main()
    show_video('videos/ppo2-futbol-pre-step-0-to-step-50.mp4')


