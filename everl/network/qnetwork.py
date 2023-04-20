# pylint: disable=no-member
# pylint: disable=arguments-differ

from typing import Optional
import torch

from .components import Component, ComponentDummy
from .network import Network


class QNetwork(Network):
    def __init__(
        self,
        body: Component,
        n_observations: int,
        n_actions: int,
        head: Optional[Component] = None,
    ):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.body = body
        self.head = head or ComponentDummy()

        self.head.n_inputs = n_observations
        self.body.n_inputs = self.head.n_outputs + n_actions
        self.body.output_layer_size = 1

    @property
    def device(self) -> torch.device:
        return self.body.device

    def forward(
        self, obs_batch: torch.Tensor, action_batch: torch.Tensor, *args, **kwds
    ) -> torch.Tensor:
        head_out = self.head(obs_batch)
        body_in = torch.dstack([head_out, action_batch])
        q_value_batch = self.body(body_in)
        return q_value_batch

    def forward_play(
        self, obs_batch: torch.Tensor, action_batch: torch.Tensor, *args, **kwds
    ) -> torch.Tensor:
        head_out = self.head.forward_play(obs_batch)
        body_in = torch.dstack([head_out, action_batch])
        q_value_batch = self.body.forward_play(body_in)
        return q_value_batch

    def reset(self) -> None:
        ...
