from abc import ABC, abstractmethod
from typing import NamedTuple
import numpy as np


class Episode:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.hidden_states = []

    def add_transition(self, state, action, reward, next_state, done, hidden_state=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        if hidden_state:
            self.hidden_states.append(hidden_state)

    def __len__(self):
        return len(self.states)


class Batch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: float
    next_states: np.ndarray
    dones: bool
    hidden_states: np.ndarray = None


class ReplayBuffer(ABC):
    @abstractmethod
    def push(self, episode: Episode) -> None:
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> Batch:
        ...
