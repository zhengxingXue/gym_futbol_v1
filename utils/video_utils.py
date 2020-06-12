import time
import io
import imageio
import gym
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
import cv2
import pymunk
import numpy as np
import matplotlib.pyplot as plt
from gym_futbol_v1.envs.action import action_key_string, arrow_key_string


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
    title_str = "total reward : " + str(total_reward)
    title_str += "\ncurrent reward : " + str(reward)

    for i in range(env.number_of_player):
        title_str += "\nplayer " + str(i) + " action : " + action_key_string(
            action[2 * i + 1]) + " arrow : " + arrow_key_string(action[2 * i])
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
    img, _ = render_helper(env, [0, 0]*env.number_of_player, 0, 0)

    height, width, _ = np.shape(img)
    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = env.metadata['video.frames_per_second']
    video_filename = video_folder + prefix + '.mp4'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    done = False
    total_reward = 0

    while not done:
    # for _ in range(1):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        img, total_reward = render_helper(env, action, total_reward, reward)

        out.write(img)

    out.release()
