from abc import ABC, abstractmethod
from typing import List, NamedTuple, Union
import numpy as np
from torch.nn.utils.rnn import PackedSequence

import torch

from dataclasses import dataclass


@dataclass
class EpisodeNumpy:

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    def __len__(self):
        return len(self.states)


class Episode:
    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(np.array([reward]))
        self.next_states.append(next_state)
        self.dones.append(np.array([done]))

    def __len__(self):
        return len(self.states)

    def to_numpy(self) -> EpisodeNumpy:
        return EpisodeNumpy(
            states=np.array(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            next_states=np.array(self.next_states),
            dones=np.array(self.dones),
        )


class Batch(NamedTuple):
    states: PackedSequence
    actions: PackedSequence
    rewards: PackedSequence
    next_states: PackedSequence
    dones: PackedSequence


class ReplayBuffer(ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int:
        ...

    @abstractmethod
    def push(self, episode: Episode) -> None:
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
