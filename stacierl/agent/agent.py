from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class EpisodeCounter:
    heatup: int = 0
    exploration: int = 0
    evaluation: int = 0

    def __iadd__(self, other):
        self.exploration += other.exploration
        self.evaluation += other.evaluation
        return self


@dataclass
class StepCounter:
    heatup: int = 0
    exploration: int = 0
    evaluation: int = 0
    update: int = 0

    def __iadd__(self, other):
        self.exploration += other.exploration
        self.evaluation += other.evaluation
        self.update += other.update
        return self


class Agent(ABC):
    @abstractmethod
    def heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        ...

    @abstractmethod
    def explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        ...

    @abstractmethod
    def update(self, steps) -> List[float]:
        ...

    @abstractmethod
    def evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
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
