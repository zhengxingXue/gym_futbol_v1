import os
import time
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
import gym
from gym_futbol_v1.envs import Side
from utils import notebook_render_mlp, notebook_render_lstm


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
                           save_dir_prefix='./training/logs', log_dir_prefix='./training/tmp', verbose=0):
    """
    :param policy: stable-baseline policy
    :param policy_name: string of policy name
    :param env_id: id of the environment
    :param env_n: number of env for the PPO2 model
    :param time_step: time step for one step of multi agent training
    :param num_turn: number of time to turn
    :param save_dir_prefix: prefix of directory to save trained model and best model
    :param log_dir_prefix: prefix of log directory
    :param verbose: PPO2 training verbose or not
    :return: (PPO2, PPO2, str)the 2 trained ppo2 model, and the save directory
    """
    time_str = "{}".format(int(time.time()))
    log_dir = log_dir_prefix + "/" + policy_name + '-multi-agent-' + time_str
    os.makedirs(log_dir, exist_ok=True)

    # initial left env & model, with no model set
    env_left = create_multi_agent_env(env_id, Side.left, log_dir=log_dir, num_envs=env_n,
                                      set_model=False, model=None, model_side=None)
    model_left = PPO2(policy, env_left, verbose=verbose)

    # initial right env & model, with no model set
    env_right = create_multi_agent_env(env_id, Side.right, log_dir=log_dir, num_envs=env_n,
                                       set_model=False, model=None, model_side=None)
    model_right = PPO2(policy, env_right, verbose=verbose)

    save_dir = save_dir_prefix + '/' + policy_name + '-multi-agent-' + time_str
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
        model_right.save(model_right_path)
        print("right_model saved to " + model_right_path + '.zip')

    return model_left, model_right, save_dir


class MultiAgentTrain:

    def __init__(self, env_id='futbol-v1', env_num=8, rl_agent=PPO2, policy=MlpPolicy, policy_name='ppo2-mlp'):
        self.env_id = env_id
        self.env_num = env_num
        self.rl_agent = rl_agent
        self.policy = policy
        self.policy_name = policy_name
        self.verbose = 0
        self.env_left = gym.make(env_id)
        self.env_right = gym.make(env_id)
        self.log_dir = './'
        self.save_dir = './'
        self.set_up_dir()
        self._set_up_envs()
        self._set_up_models()
        self.turn = 0
        self.total_time_step = 0
        self.left_time_step = 0
        self.right_time_step = 0

    def get_info(self):
        print('Training Turn : {}'.format(self.turn))
        print('Total Time Step : {}'.format(self.total_time_step))
        print('Model Left Time Step : {}'.format(self.left_time_step))
        print('Model Right Time Step : {}'.format(self.right_time_step))
        print('Model Save Directory : {}'.format(self.save_dir))
        print('Model Log Directory : {}'.format(self.log_dir))
        print('Verbose : ' + str(self.verbose))

    def set_up_dir(self, save_dir_prefix='./training/logs', log_dir_prefix='./training/tmp'):
        time_str = "{}".format(int(time.time()))
        self.log_dir = log_dir_prefix + "/" + self.policy_name + '-multi-agent-' + time_str
        os.makedirs(self.log_dir, exist_ok=True)
        self.save_dir = save_dir_prefix + '/' + self.policy_name + '-multi-agent-' + time_str
        os.makedirs(self.save_dir, exist_ok=True)

    def _set_up_envs(self):
        self.set_up_env(Side.left, random_op=True, set_model=False)
        self.set_up_env(Side.right, random_op=True, set_model=False)

    def _set_up_models(self):
        self.model_left = self.rl_agent(self.policy, self.env_left, verbose=self.verbose)
        self.model_right = self.rl_agent(self.policy, self.env_right, verbose=self.verbose)

    def set_up_env(self, side, random_op=False, set_model=True):
        """
        :param side: which side of env to set up
        :param random_op: whether the env's opponents' actions are random.
        """
        if side == Side.left:
            if random_op:
                self.env_left = create_multi_agent_env(self.env_id, Side.left, log_dir=self.log_dir,
                                                       num_envs=self.env_num,
                                                       set_model=False, model=None, model_side=None)
            else:
                self.env_left = create_multi_agent_env(self.env_id, Side.left, log_dir=self.log_dir,
                                                       num_envs=self.env_num,
                                                       set_model=True, model=self.model_right,
                                                       model_side=Side.right)  # add right model to env
            if set_model:
                self.model_left.set_env(self.env_left)
        elif side == Side.right:
            if random_op:
                self.env_right = create_multi_agent_env(self.env_id, Side.right, log_dir=self.log_dir,
                                                        num_envs=self.env_num,
                                                        set_model=False, model=None, model_side=None)
            else:
                self.env_right = create_multi_agent_env(self.env_id, Side.right, log_dir=self.log_dir,
                                                        num_envs=self.env_num,
                                                        set_model=True, model=self.model_left,
                                                        model_side=Side.left)  # add left model to env
            if set_model:
                self.model_right.set_env(self.env_right)

    def train(self, num_turn=10, time_step=10 ** 4, save=True, verbose=0):
        self.change_verbose(verbose)
        current_turn = self.turn
        for i in range(num_turn):
            model_path = self.save_dir + '/' + self.policy_name + '-' + str(i+current_turn)

            model_left_path = model_path + '-left'
            self.train_left(time_step=time_step, save=save, save_path=model_left_path, random_right=False)

            model_right_path = model_path + '-right'
            self.train_right(time_step=time_step, save=save, save_path=model_right_path, random_left=False)

            self.turn += 1

    def train_left(self, time_step=10 ** 4, save=False, save_path='./', random_right=False):
        self.set_up_env(Side.left, random_op=random_right)
        self.model_left.learn(total_timesteps=time_step, reset_num_timesteps=False)
        self.left_time_step += time_step
        self.total_time_step += time_step
        if save:
            self.model_left.save(save_path)
            print("left__model saved to " + save_path + '.zip')

    def train_right(self, time_step=10 ** 4, save=False, save_path='./', random_left=False):
        self.set_up_env(Side.right, random_op=random_left)
        self.model_right.learn(total_timesteps=time_step, reset_num_timesteps=False)
        self.right_time_step += time_step
        self.total_time_step += time_step
        if save:
            self.model_right.save(save_path)
            print("right_model saved to " + save_path + '.zip')

    def change_verbose(self, verbose):
        """
        :param verbose: 0 for no train info, 1 for train info
        """
        self.verbose = verbose
        self.model_left.verbose = verbose
        self.model_right.verbose = verbose

    def notebook_render_left(self, random_right=False):
        if random_right:
            self.env_left = gym.make(self.env_id)
            notebook_render_mlp(self.env_left, self.model_left, side=Side.left)
        else:
            self.env_left = MultiAgentWrapper(gym.make(self.env_id), Side.left)
            self.env_left.set_agent(self.model_right, Side.right)
            _ = notebook_render_mlp(self.env_left, self.model_left, length=300, side=Side.left)

    def notebook_render_right(self, random_left=False):
        if random_left:
            self.env_right = gym.make(self.env_id)
            notebook_render_mlp(self.env_right, self.model_right, side=Side.right)
        else:
            self.env_right = MultiAgentWrapper(gym.make(self.env_id), Side.right)
            self.env_right.set_agent(self.model_left, Side.left)
            _ = notebook_render_mlp(self.env_right, self.model_right, length=300, side=Side.right)

    def save_models(self):
        self.model_left.save(self.save_dir + '/' + self.policy_name + '-left')
        self.model_right.save(self.save_dir + '/' + self.policy_name + '-right')
