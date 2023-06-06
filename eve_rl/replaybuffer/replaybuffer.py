from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union
import numpy as np

import torch

from ..util import EveRLObject


class Episode:
    def __init__(
        self,
        reset_obs: Dict[str, np.ndarray],
        reset_flat_obs: np.ndarray,
        flat_obs_to_obs: Optional[Union[List, Dict]] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.obs: List[Dict[str, np.ndarray]] = [reset_obs]
        self.flat_obs: List[np.ndarray] = [reset_flat_obs]
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminals: List[bool] = []
        self.truncations: List[bool] = []
        self.infos: List[Dict[str, np.ndarray]] = []
        self.episode_reward: float = 0.0
        self.flat_state_to_state = flat_obs_to_obs
        self.seed = seed
        self.options = options

    def add_transition(
        self,
        obs: Dict[str, np.ndarray],
        flat_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        truncation: bool,
        info: Dict[str, np.ndarray],
    ):
        self.obs.append(obs)
        self.flat_obs.append(flat_obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.truncations.append(truncation)
        self.infos.append(info)
        self.episode_reward += reward

    def to_replay(self):
        return EpisodeReplay(self.flat_obs, self.actions, self.rewards, self.terminals)

    def __len__(self):
        return len(self.actions)


@dataclass
class EpisodeReplay:
    flat_obs: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    terminals: List[bool]

    def __len__(self):
        return len(self.actions)


class Batch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    padding_mask: torch.Tensor = None

    def to(self, device: torch.device, non_blocking=False):
        obs = self.obs.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        actions = self.actions.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        rewards = self.rewards.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        terminals = self.terminals.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        if self.padding_mask is not None:
            padding_mask = self.padding_mask.to(
                device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ).share_memory_()
        else:
            padding_mask = None
        return Batch(obs, actions, rewards, terminals, padding_mask)


class ReplayBuffer(EveRLObject, ABC):
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
