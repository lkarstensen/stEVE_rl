from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
import torch.multiprocessing as mp

from stacierl.replaybuffer.replaybuffer import Episode


@dataclass
class EpisodeCounter:
    heatup: int = 0
    exploration: int = 0
    evaluation: int = 0
    lock = mp.Lock()

    def __iadd__(self, other):
        self.heatup += other.heatup
        self.exploration += other.exploration
        self.evaluation += other.evaluation
        return self


@dataclass
class StepCounter:
    heatup: int = 0
    exploration: int = 0
    evaluation: int = 0
    update: int = 0
    lock = mp.Lock()

    def __iadd__(self, other):
        self.heatup += other.heatup
        self.exploration += other.exploration
        self.evaluation += other.evaluation
        self.update += other.update
        return self


class StepCounterShared(StepCounter):
    def __init__(self):
        self._heatup: mp.Value = mp.Value("i", 0)
        self._exploration: mp.Value = mp.Value("i", 0)
        self._evaluation: mp.Value = mp.Value("i", 0)
        self._update: mp.Value = mp.Value("i", 0)

    @property
    def heatup(self) -> int:
        return self._heatup.value

    @heatup.setter
    def heatup(self, value: int) -> int:
        self._heatup.value = value

    @property
    def exploration(self) -> int:
        return self._exploration.value

    @exploration.setter
    def exploration(self, value: int) -> int:
        self._exploration.value = value

    @property
    def evaluation(self) -> int:
        return self._evaluation.value

    @evaluation.setter
    def evaluation(self, value: int) -> int:
        self._evaluation.value = value

    @property
    def update(self) -> int:
        return self._update.value

    @update.setter
    def update(self, value: int) -> int:
        self._update.value = value

    def __iadd__(self, other):
        self._heatup.value = self._heatup.value + other.heatup
        self._exploration.value = self._exploration.value + other.exploration
        self._evaluation.value = self._evaluation.value + other.evaluation
        self._update.value = self._update.value + other.update
        return self


class EpisodeCounterShared(EpisodeCounter):
    def __init__(self):
        self._heatup: mp.Value = mp.Value("i", 0)
        self._exploration: mp.Value = mp.Value("i", 0)
        self._evaluation: mp.Value = mp.Value("i", 0)

    @property
    def heatup(self) -> int:
        return self._heatup.value

    @heatup.setter
    def heatup(self, value: int) -> int:
        self._heatup.value = value

    @property
    def exploration(self) -> int:
        return self._exploration.value

    @exploration.setter
    def exploration(self, value: int) -> int:
        self._exploration.value = value

    @property
    def evaluation(self) -> int:
        return self._evaluation.value

    @evaluation.setter
    def evaluation(self, value: int) -> int:
        self._evaluation.value = value

    def __iadd__(self, other):
        self._heatup.value = self._heatup.value + other.heatup
        self._exploration.value = self._exploration.value + other.exploration
        self._evaluation.value = self._evaluation.value + other.evaluation
        return self


class Agent(ABC):
    @abstractmethod
    def heatup(self, steps: int = None, episodes: int = None) -> List[Episode]:
        ...

    @abstractmethod
    def explore(self, steps: int = None, episodes: int = None) -> List[Episode]:
        ...

    @abstractmethod
    def update(self, steps) -> List[List[float]]:
        ...

    @abstractmethod
    def evaluate(self, steps: int = None, episodes: int = None) -> List[Episode]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    @abstractmethod
    def step_counter(self) -> StepCounter:
        ...

    @property
    @abstractmethod
    def episode_counter(self) -> EpisodeCounter:
        ...

    @abstractmethod
    def save_checkpoint(self, directory: str, name: str) -> None:
        ...

    @abstractmethod
    def load_checkpoint(self, directory: str, name: str) -> None:
        ...

    @abstractmethod
    def copy(self):
        ...
