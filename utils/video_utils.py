import time
import gym
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
import cv2


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
