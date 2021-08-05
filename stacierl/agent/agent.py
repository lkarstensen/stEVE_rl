from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
from math import inf


def dict_state_to_flat_np_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    keys = sorted(state.keys())

    flat_state = []
    for key in keys:
        flat_state.append(state[key].flatten())
    flat_state = np.array(flat_state).flatten()
    return flat_state


class Agent(ABC):
    def update(self, steps, batch_size):
        return self._update(steps, batch_size)

    @abstractmethod
    def _update(self, steps):
        ...

    def heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        if steps is None and episodes is None:
            raise ValueError("One of the two (steps or episodes) needs to be given")
        steps = steps or inf
        episodes = episodes or inf
        return self._heatup(steps, episodes)

    @abstractmethod
    def _heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        ...

    def explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        if steps is None and episodes is None:
            raise ValueError("One of the two (steps or episodes) needs to be given")
        steps = steps or inf
        episodes = episodes or inf
        return self._explore(steps, episodes)

    @abstractmethod
    def _explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        ...

    def evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        if steps is None and episodes is None:
            raise ValueError("One of the two (steps or episodes) needs to be given")
        steps = steps or inf
        episodes = episodes or inf
        return self._evaluate(steps, episodes)

    @abstractmethod
    def _evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        ...
