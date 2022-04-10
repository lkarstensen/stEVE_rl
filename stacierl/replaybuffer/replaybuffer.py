from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Union
import numpy as np

import torch

from ..util import StacieRLUserObject


class Episode:
    def __init__(self, reset_state: Dict[str, np.ndarray], reset_flat_state: np.ndarray) -> None:
        self.states: List[Dict[str, np.ndarray]] = [reset_state]
        self.flat_states: List[np.ndarray] = [reset_flat_state]
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.infos: List[Dict[str, np.ndarray]] = []
        self.successes: List[float] = []
        self.episode_reward: float = 0.0

    @property
    def episode_success(self) -> float:
        return self.successes[-1]

    def add_transition(
        self,
        state: Dict[str, np.ndarray],
        flat_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        info: Dict[str, np.ndarray],
        success: float,
    ):
        self.states.append(state)
        self.flat_states.append(flat_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        self.successes.append(success)
        self.episode_reward += reward

    def to_replay(self):
        return EpisodeReplay(self.flat_states, self.actions, self.rewards, self.dones)

    def __len__(self):
        return len(self.actions)


@dataclass
class EpisodeReplay:
    flat_states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]

    def __len__(self):
        return len(self.actions)


class Batch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    padding_mask: torch.Tensor = None


class ReplayBuffer(StacieRLUserObject, ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int:
        ...

    @abstractmethod
    def push(self, episode: Union[Episode, EpisodeReplay]) -> None:
        ...

    @abstractmethod
    def sample(self) -> Batch:
        ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
