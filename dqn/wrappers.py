import typing as tt
import gymnasium as gym
from gymnasium import spaces
import collections
import numpy as np
from stable_baselines3.common import atari_wrappers


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Wrapper to convert images to tensors.
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(), shape=new_shape, dtype=obs.dtype
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    """
    Replay buffer wrapper.
    """

    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0),
            obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype,
        )
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def reset(
        self,
        *,
        seed: tt.Optional[int] = None,
        options: tt.Optional[dict[str, tt.Any]] = None
    ):
        for _ in range(self.buffer.maxlen - 1):
            self.buffer.append(self.env.observation_space.low)
        obs, extra = self.env.reset()
        return self.observation(obs), extra

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)


def make_env(
    env_name: str, record_video: bool = False, video_folder: str = "videos/", **kwargs
):
    """
    Factory method for creating a new gym.Env instance wrapped around custom wrappers.
    :param env_name: Gym environment name.
    :param record_video: Whether to record gameplay video.
    :param video_folder: Directory where video will be saved.
    :param kwargs: Extra args for gym.make.
    :return: Wrapped gym.Env instance.
    """
    env = gym.make(env_name, **kwargs)

    if record_video:
        from gymnasium.wrappers import RecordVideo

        env = RecordVideo(
            env, video_folder=video_folder, episode_trigger=lambda ep: ep % 50 == 0
        )

    env = atari_wrappers.AtariWrapper(env, clip_reward=False, noop_max=0)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)

    return env
