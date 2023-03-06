from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Union
import numpy as np

import torch


class Episode:
    def __init__(
        self, reset_state: Dict[str, np.ndarray], reset_flat_state: np.ndarray
    ) -> None:
        self.states: List[Dict[str, np.ndarray]] = [reset_state]
        self.flat_states: List[np.ndarray] = [reset_flat_state]
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminals: List[bool] = []
        self.truncations: List[bool] = []
        self.infos: List[Dict[str, np.ndarray]] = []
        self.episode_reward: float = 0.0

    def add_transition(
        self,
        state: Dict[str, np.ndarray],
        flat_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        truncation: bool,
        info: Dict[str, np.ndarray],
    ):
        self.states.append(state)
        self.flat_states.append(flat_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.truncations.append(truncation)
        self.infos.append(info)
        self.episode_reward += reward

    def to_replay(self):
        return EpisodeReplay(
            self.flat_states, self.actions, self.rewards, self.terminals
        )

    def __len__(self):
        return len(self.actions)


@dataclass
class EpisodeReplay:
    flat_states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    terminals: List[bool]

    def __len__(self):
        return len(self.actions)


class Batch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    padding_mask: torch.Tensor = None


class ReplayBuffer(ABC):
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
