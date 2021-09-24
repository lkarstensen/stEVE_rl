from typing import Dict, Iterator, Tuple
import numpy as np
from torch.distributions.normal import Normal

from .sacmodel import SACModel, ModelStateDicts
from ... import network
import torch.optim as optim
import torch
from dataclasses import dataclass
from copy import deepcopy

from ...environment import ObservationSpace, ActionSpace


@dataclass
class SACStateDicts(ModelStateDicts):
    q1: Dict[str, torch.Tensor]
    q2: Dict[str, torch.Tensor]
    target_q1: Dict[str, torch.Tensor]
    target_q2: Dict[str, torch.Tensor]
    policy: Dict[str, torch.Tensor]

    def __iter__(self):
        return iter([self.q1, self.q2, self.target_q1, self.target_q2, self.policy])

    def copy(self):
        return SACStateDicts(
            deepcopy(self.q1),
            deepcopy(self.q2),
            deepcopy(self.target_q1),
            deepcopy(self.target_q2),
            deepcopy(self.policy),
        )


class Vanilla(SACModel):
    def __init__(
        self,
        q1: network.QNetwork,
        q2: network.QNetwork,
        policy: network.GaussianPolicy,
        learning_rate: float,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
    ) -> None:
        self.learning_rate = learning_rate
        self.obs_space = obs_space
        self.action_space = action_space

        self.q1 = q1
        self.q2 = q2
        self.target_q1 = q1.copy()
        self.target_q2 = q2.copy()
        self.policy = policy
        self.log_alpha = torch.zeros(1, requires_grad=True)

        n_actions = 1
        for dim in self.action_space.shape:
            n_actions *= dim
        n_observations = 0
        for state_shape in self.obs_space.shape.values():
            n_state_observations = 1
            for dim in state_shape:
                n_state_observations *= dim
            n_observations += n_state_observations
        self.q1.set_input(n_observations, n_actions)
        self.q2.set_input(n_observations, n_actions)
        self.target_q1.set_input(n_observations, n_actions)
        self.target_q2.set_input(n_observations, n_actions)
        self.policy.set_input(n_observations)
        self._init_optimizer()

    def _init_optimizer(self):
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

    def get_play_action(self, flat_state: np.ndarray = None) -> np.ndarray:
        with torch.no_grad():
            flat_state = [
                torch.as_tensor(flat_state, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            ]
            mean, log_std = self.policy.forward(flat_state)
            std = log_std.exp()

            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
            action = action.cpu().detach().squeeze(0).squeeze(0).numpy()
            return action

    def get_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset()
        q1 = self.q1.forward(states, actions)
        self.reset()
        q2 = self.q2.forward(states, actions)
        return q1, q2

    def get_target_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.reset()
            q1 = self.target_q1.forward(states, actions)
            self.reset()
            q2 = self.target_q2.forward(states, actions)
            return q1, q2

    # epsilon makes sure that log(0) does not occur
    def get_update_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset()
        mean_batch, log_std = self.policy.forward(state_batch)

        std_batch = log_std.exp()
        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = normal.log_prob(z) - torch.log(1 - action_batch.pow(2) + epsilon)
        log_pi_batch = log_pi_batch.sum(-1, keepdim=True)

        return action_batch, log_pi_batch

    def q1_update_zero_grad(self):
        self.q1_optimizer.zero_grad()

    def q2_update_zero_grad(self):
        self.q2_optimizer.zero_grad()

    def policy_update_zero_grad(self):
        self.policy_optimizer.zero_grad()

    def alpha_update_zero_grad(self):
        self.alpha_optimizer.zero_grad()

    def q1_update_step(self):
        self.q1_optimizer.step()

    def q2_update_step(self):
        self.q2_optimizer.step()

    def policy_update_step(self):
        self.policy_optimizer.step()

    def alpha_update_step(self):
        self.alpha_optimizer.step()

    def to(self, device: torch.device):
        self.device = device
        self.q1.to(device)
        self.q2.to(device)
        self.target_q1.to(device)
        self.target_q2.to(device)
        self.policy.to(device)
        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        self._init_optimizer()

    def update_target_q(self, tau):
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def copy(self):
        copy = self.__class__(
            self.q1.copy(),
            self.q2.copy(),
            self.policy.copy(),
            self.learning_rate,
            self.obs_space,
            self.action_space,
        )

        return copy

    def copy_shared_memory(self):

        copy = self.__class__(
            self.q1.copy(),
            self.q2.copy(),
            self.policy.copy(),
            self.learning_rate,
            self.obs_space,
            self.action_space,
        )
        self.q1.share_memory()
        self.q2.share_memory()
        self.target_q1.share_memory()
        self.target_q2.share_memory()
        self.policy.share_memory()
        copy.q1 = self.q1
        copy.q2 = self.q2
        copy.policy = self.policy
        copy.target_q1 = self.target_q1
        copy.target_q2 = self.target_q2
        copy._init_optimizer()

        return copy

    def load_state_dicts(self, sac_model_state_dicts: SACStateDicts):
        self.q1.load_state_dict(sac_model_state_dicts.q1)
        self.q2.load_state_dict(sac_model_state_dicts.q2)
        self.target_q1.load_state_dict(sac_model_state_dicts.target_q1)
        self.target_q2.load_state_dict(sac_model_state_dicts.target_q2)
        self.policy.load_state_dict(sac_model_state_dicts.policy)

    @property
    def state_dicts(self) -> SACStateDicts:
        state_dicts = SACStateDicts(
            self.q1.state_dict(),
            self.q2.state_dict(),
            self.target_q1.state_dict(),
            self.target_q2.state_dict(),
            self.policy.state_dict(),
        )
        return state_dicts

    def reset(self) -> None:
        for net in self:
            net.reset()

    def __iter__(self) -> Iterator[network.Network]:
        return iter([self.q1, self.q2, self.target_q1, self.target_q2, self.policy])