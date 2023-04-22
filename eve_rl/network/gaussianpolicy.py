from typing import Optional, Tuple
import logging
import torch

from .component import Component, ComponentDummy
from .network import Network


class GaussianPolicy(Network):
    def __init__(
        self,
        body: Component,
        n_observations: int,
        n_actions: int,
        head: Optional[Component] = None,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__module__)

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.body = body
        self.head = head or ComponentDummy()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.head.n_inputs = n_observations
        self.body.n_inputs = self.head.n_outputs
        self.body.output_layer_size = [n_actions, n_actions]

    @property
    def device(self) -> torch.device:
        return self.body.device

    def forward(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_out = self.head(obs_batch)

        mean, log_std = self.body.forward(head_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def forward_play(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_out = self.head.forward_play(obs_batch)
        mean, log_std = self.body.forward_play(head_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def reset(self) -> None:
        ...
