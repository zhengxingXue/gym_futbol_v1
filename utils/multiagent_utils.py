import os
import time
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
import gym
from gym_futbol_v1.envs import Side


class MultiAgentWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, side=Side.left):
        # action side
        self.side = side
        # Call the parent constructor, so we can access self.env later
        super(MultiAgentWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        return obs

    def step(self, action, team_side=None):
        """
        :param action: ([float] or int) Action taken by the agent
        :param team_side: used to be compatibal with written function
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action, self.side)
        return obs, reward, done, info

    def set_agent(self, model, team_side):
        """
        :param model: model to set
        :param team_side: which side of team to set
        """
        if team_side == Side.left:
            self.env.team_left_agent = model
        else:
            self.env.team_right_agent = model


def create_multi_agent_env(env, side, log_dir='./temp', num_envs=8, set_model=False, model=None, model_side=None):
    """
        :param env: (str or gym.env) if env is string use gym.make() else directly use env
        :param side: player side for the env
        :param log_dir: log directory
        :param num_envs: number of environment
        :param set_model: whether to set model for the env
        :param model: the model to set
        :param model_side: the model's side
        :return: DummyVecEnv for self play training
        """
    if isinstance(env, str):
        env = gym.make(env)

    env = MultiAgentWrapper(env, side)
    if set_model:
        env.set_agent(model, model_side)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env] * num_envs)
    return env


def ppo2_multi_agent_train(policy, policy_name, env_id='futbol-v1', env_n=8, time_step=10 ** 5, num_turn=10,
                           save_dir_prefix='./training/logs', verbose=0):
    """
    :param policy: stable-baseline policy
    :param policy_name: string of policy name
    :param env_id: id of the environment
    :param env_n: number of env for the PPO2 model
    :param time_step: time step for one step of multi agent training
    :param num_turn: number of time to turn
    :param save_dir_prefix: prefix of directory to save trained model and best model
    :param verbose: PPO2 training verbose or not
    :return: (PPO2, PPO2, str)the 2 trained ppo2 model, and the save directory
    """
    time_str = "{}".format(int(time.time()))
    log_dir = "./training/tmp/" + policy_name + 'multi-agent-' + time_str
    os.makedirs(log_dir, exist_ok=True)

    # initial left env & model, with no model set
    env_left = create_multi_agent_env(env_id, Side.left, log_dir=log_dir, num_envs=env_n,
                                      set_model=False, model=None, model_side=None)
    model_left = PPO2(policy, env_left, verbose=verbose)

    # initial right env & model, with no model set
    env_right = create_multi_agent_env(env_id, Side.right, log_dir=log_dir, num_envs=env_n,
                                       set_model=False, model=None, model_side=None)
    model_right = PPO2(policy, env_right, verbose=verbose)

    save_dir = save_dir_prefix + '/' + policy_name + 'multi-agent-' + time_str
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_turn):
        model_path = save_dir + '/' + policy_name + '-' + str(i)
        # start with left
        env_left = create_multi_agent_env(env_id, Side.left, log_dir=log_dir, num_envs=env_n,
                                          set_model=True, model=model_right,
                                          model_side=Side.right)  # add right model to env
        model_left.set_env(env_left)  # set the new env to model
        model_left.learn(total_timesteps=time_step, reset_num_timesteps=False)

        model_left_path = model_path + '-left'
        model_left.save(model_left_path)
        print("left__model saved to " + model_left_path + '.zip')

        # switch to right
        env_right = create_multi_agent_env(env_id, Side.right, log_dir=log_dir, num_envs=env_n,
                                           set_model=True, model=model_left,
                                           model_side=Side.left)  # add left model to env
        model_right.set_env(env_right)  # set the new env to model
        model_right.learn(total_timesteps=time_step, reset_num_timesteps=False)
        model_right_path = model_path + '-right'
        model_left.save(model_right_path)
        print("right_model saved to " + model_right_path + '.zip')

    return model_left, model_right, save_dir
