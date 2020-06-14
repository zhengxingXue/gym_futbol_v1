import time
import io
import os
import imageio
import gym
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
import cv2
import pymunk
import numpy as np
import matplotlib.pyplot as plt
from gym_futbol_v1.envs.action import action_key_string, arrow_key_string
from IPython import display
from gym_futbol_v1.envs import Side


def show_video(name):
    cap = cv2.VideoCapture(name)
    while cap.isOpened():
        ret, frame = cap.read()
        # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        if ret:
            cv2.imshow("Image", frame)
            time.sleep(0.1)
        else:
            print('video ended')
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def record_video(env_id, model, video_length=300, prefix='', video_folder='videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    # Start the video at step=0 and record 300 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


def record_gif(env_id, model, video_length=300, prefix='env', video_folder='videos/'):
    images = []
    env = gym.make(env_id)
    obs = env.reset()
    img = env.render(mode='rgb_array')
    for _ in range(video_length):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        img = env.render(mode='rgb_array')

    fps = env.metadata['video.frames_per_second']
    imageio.mimsave(video_folder + prefix + '.gif', [np.array(img) for img in images], fps=fps)


def get_title_str(env, total_reward, reward, n_player, action):
    title_str = "total reward : " + '{:6.1f}'.format(total_reward)
    title_str += "\ncurrent reward : " + '{:4.1f}'.format(reward)
    title_str += "\ncurrent time : " + '{:4.1f}'.format(env.current_time)
    title_str += "\n" + env.get_score()

    for i in range(n_player):
        title_str += "\nplayer " + str(i) + " action : " + '{:6}'.format(action_key_string(
            action[2 * i + 1])) + " arrow : " + '{:6}'.format(arrow_key_string(action[2 * i]))

    return title_str


def render_helper(env, action, total_reward, reward):
    # plot the current state
    padding = 5
    fig = plt.figure()
    ax = fig.add_subplot()
    ax = plt.axes(xlim=(0 - padding, env.WIDTH + padding),
                  ylim=(0 - padding, env.HEIGHT + padding))
    ax.set_aspect("equal")
    o = pymunk.matplotlib_util.DrawOptions(ax)
    env.space.debug_draw(o)

    total_reward += reward
    title_str = get_title_str(env, total_reward, reward, env.number_of_player, action)
    plt.title(title_str, loc='left')
    plt.axis('off')
    plt.tight_layout()

    # convert the plt figure to RGB array
    dpi = 180
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    plt.close()
    # plt.show()

    return img, total_reward


def record_video_with_title(env_id, model, prefix='test', video_folder='videos/'):
    env = gym.make(env_id)
    obs = env.reset()
    img, _ = render_helper(env, [0, 0] * env.number_of_player, 0, 0)

    video_folder = os.path.abspath(video_folder)
    # Create output folder if needed
    os.makedirs(video_folder, exist_ok=True)

    height, width, _ = np.shape(img)
    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = env.metadata['video.frames_per_second']
    video_filename = video_folder + '/' + prefix + '.mp4'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    done = False
    total_reward = 0

    while not done:
        # for _ in range(50):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        img, total_reward = render_helper(env, action, total_reward, reward)

        out.write(img)

    out.release()


def notebook_render_helper(env, total_reward, reward, action):
    plt.clf()
    title_str = get_title_str(env, total_reward, reward, env.number_of_player, action)
    padding = 5
    ax = plt.axes(xlim=(0 - padding, env.WIDTH + padding), ylim=(0 - padding, env.HEIGHT + padding))
    ax.set_aspect("equal")
    o = pymunk.matplotlib_util.DrawOptions(ax)
    env.space.debug_draw(o)
    plt.title(title_str, loc='left')
    display.display(plt.gcf())
    display.clear_output(wait=True)


def notebook_render_simple(env, length=300, random=True, action=np.array([0, 0, 0, 0]), side=Side.left):
    """
    :param env: environment
    :param length: render length
    :param random: whether the side act randomly
    :param action: if the random is false, the action user want the side to act
    :param side: action side
    :return: total reward
    """
    total_reward = 0
    for _ in range(length):
        if random:
            action = np.reshape(env.action_space.sample(), -1)
        else:
            action = action
        ob, reward, _, _ = env.step(action, team_side=side)
        total_reward += reward

        notebook_render_helper(env, total_reward, reward, action)

    return total_reward


def notebook_render_mlp(env_id, model, length=300, side=Side.left):
    """
    :param env_id: environment id
    :param model: model for rendering
    :param length: render length
    :param side: action side
    :return: total reward
    """
    env = gym.make(env_id)
    done = False
    total_reward = 0
    obs = env.reset()
    i = 0
    while not done:
        action, _ = model.predict(obs)
        ob, reward, _, _ = env.step(action, team_side=side)
        total_reward += reward

        notebook_render_helper(env, total_reward, reward, action)

        i += 1
        if i > length:
            break
    return total_reward


def notebook_render_lstm(env_id, model, length=300, side=Side.left):
    """
    :param env_id: environment id
    :param model: model for rendering
    :param length: render length
    :param side: action side
    :return: total reward
    """
    env = gym.make(env_id)
    done = False
    total_reward = 0
    obs = env.reset()
    n_env = model.n_envs
    _, state = model.predict(np.tile(obs, (n_env, 1)), deterministic=False)
    i = 0
    while not done:
        action, state = model.predict(np.tile(obs, (n_env, 1)), state=state, deterministic=False)
        action = action[0]
        obs, reward, done, info = env.step(action, team_side=side)

        total_reward += reward
        notebook_render_helper(env, total_reward, reward, action)

        i += 1
        if i > length:
            break
    return total_reward
