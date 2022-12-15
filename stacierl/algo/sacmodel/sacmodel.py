from abc import abstractmethod, ABC
from typing import Tuple
import numpy as np
from ..model import Model
import torch

from ..model import NetworkStatesContainer, OptimizerStatesContainer


class SACModel(Model, ABC):
    @abstractmethod
    def get_play_action(self, flat_state: np.ndarray, evaluation: bool) -> np.ndarray:
        ...

    @abstractmethod
    def get_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def get_target_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    # epsilon makes sure that log(0) does not occur
    @abstractmethod
    def get_update_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def q1_update_zero_grad(self):
        ...

    @abstractmethod
    def q2_update_zero_grad(self):
        ...

    @abstractmethod
    def policy_update_zero_grad(self):
        ...

    @abstractmethod
    def alpha_update_zero_grad(self):
        ...

    @abstractmethod
    def q1_update_step(self):
        ...

    @abstractmethod
    def q2_update_step(self):
        ...

    @abstractmethod
    def policy_update_step(self):
        ...

    @abstractmethod
    def alpha_update_step(self):
        ...

    @abstractmethod
    def q1_scheduler_step(self):
        ...

    @abstractmethod
    def q2_scheduler_step(self):
        ...

    @abstractmethod
    def policy_scheduler_step(self):
        ...

    @abstractmethod
    def update_target_q(self, tau):
        ...
