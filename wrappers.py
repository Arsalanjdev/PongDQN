from typing import Any

import gymnasium as gym
import numpy as np
import collections
from gymnasium.core import WrapperObsType
from stable_baselines3.common import atari_wrappers

class ImageToPyTorchWrapper(gym.ObservationWrapper):
    """
    Converts images to PyTorch Tensor.
    """
    def __init__(self, env) -> None:
        super().__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1],obs.shape[0],obs.shape[1]) # Channel, Height, Weight format
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(),
            shape=new_shape, dtype=obs.dtype)

        def observation(self, observation):
            return np.transpose(observation, 2,0)


class BufferWrapper(gym.ObservationWrapper):
    """
    Stacks n_steps consecutive observations into a single observation
    """

    def __init__(self, env, n_steps) -> None:
        super().__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0), obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype)
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        for i in range(self.buffer.maxlen - 1):
            self.buffer.append(self.env.observation_space.low)
        obs, extra = self.env.reset()
        return self.observation(obs), extra

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer) #flattening the buffer


def make_env(env_name: str, **kwargs):
    """
    Factory method for creating an environment.
    :param env_name:
    :param kwargs:
    :return:
    """
    env = gym.make(env_name, **kwargs)
    env = atari_wrappers.AtariWrapper(env, clip_reward=False, noop_max=0)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    return env