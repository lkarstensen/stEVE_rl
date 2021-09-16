from abc import ABC, abstractmethod
from typing import List, NamedTuple
import numpy as np
from dataclasses import dataclass


class Episode:
    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.hidden_states: List[np.ndarray] = []
        self.next_hidden_states: List[np.ndarray] = []

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        hidden_state: np.ndarray = None,
        next_hidden_state: np.ndarray = None,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        if hidden_state:
            self.hidden_states.append(hidden_state)
            self.next_hidden_states.append(next_hidden_state)

    def __len__(self):
        return len(self.states)


class Batch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    hidden_states: np.ndarray = None
    hidden_next_states: np.ndarray = None


class ReplayBuffer(ABC):
    @abstractmethod
    def push(self, episode: Episode) -> None:
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> Batch:
        ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def close(self):
        ...
