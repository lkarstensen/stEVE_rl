import gymnasium as gym
import numpy as np


class DummyEnv(gym.Env):
    def __init__(  # pylint: disable=super-init-not-called
        self, *args, **kwds  # pylint: disable=unused-argument
    ) -> None:
        ...

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=np.empty((1,)), high=np.empty((1,)))

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=np.empty((1,)), high=np.empty((1,)))

    def step(self, action: np.ndarray) -> None:
        ...

    def reset(self, *args, **kwds) -> None:  # pylint: disable=unused-argument
        ...

    def render(self) -> None:
        ...

    def close(self) -> None:
        ...
