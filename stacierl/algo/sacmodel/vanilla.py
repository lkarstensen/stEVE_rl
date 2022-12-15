from typing import Dict, Iterator, Tuple
import numpy as np
from torch.distributions.normal import Normal

from .sacmodel import SACModel, NetworkStatesContainer, OptimizerStatesContainer
from ... import network
from ...optimizer import Optimizer
import torch.optim as optim
import torch
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class SACNetworkStateContainer(NetworkStatesContainer):
    q1: Dict[str, torch.Tensor]
    q2: Dict[str, torch.Tensor]
    target_q1: Dict[str, torch.Tensor]
    target_q2: Dict[str, torch.Tensor]
    policy: Dict[str, torch.Tensor]
    log_alpha: Dict[str, torch.Tensor]

    def __iter__(self):
        return iter(
            [
                self.q1,
                self.q2,
                self.target_q1,
                self.target_q2,
                self.policy,
                self.log_alpha,
            ]
        )

    def copy(self):
        return SACNetworkStateContainer(
            deepcopy(self.q1),
            deepcopy(self.q2),
            deepcopy(self.target_q1),
            deepcopy(self.target_q2),
            deepcopy(self.policy),
            deepcopy(self.log_alpha),
        )

    def to_dict(self) -> Dict:
        model_state_dict = {
            "q1": self.q1,
            "q2": self.q2,
            "target_q1": self.target_q1,
            "target_q2": self.target_q2,
            "policy": self.policy,
            "log_alpha": self.log_alpha,
        }

        return model_state_dict

    def from_dict(self, model_state_dict: Dict):
        self.q1 = model_state_dict["q1"]
        self.q2 = model_state_dict["q2"]
        self.target_q1 = model_state_dict["target_q1"]
        self.target_q2 = model_state_dict["target_q2"]
        self.policy = model_state_dict["policy"]
        self.log_alpha = model_state_dict["log_alpha"]


@dataclass
class SACOptimizerStateContainer(OptimizerStatesContainer):
    q1: Dict[str, torch.Tensor]
    q2: Dict[str, torch.Tensor]
    policy: Dict[str, torch.Tensor]
    alpha: Dict[str, torch.Tensor]

    def __iter__(self):
        return iter([self.q1, self.q2, self.policy, self.alpha])

    def copy(self):
        return SACOptimizerStateContainer(
            deepcopy(self.q1),
            deepcopy(self.q2),
            deepcopy(self.policy),
            deepcopy(self.alpha),
        )

    def to_dict(self) -> Dict:
        optimizer_state_dict = {
            "q1": self.q1,
            "q2": self.q2,
            "policy": self.policy,
            "alpha": self.alpha,
        }

        return optimizer_state_dict

    def from_dict(self, optimizer_state_dict: Dict):
        self.q1 = optimizer_state_dict["q1"]
        self.q2 = optimizer_state_dict["q2"]
        self.policy = optimizer_state_dict["policy"]
        self.alpha = optimizer_state_dict["alpha"]


class Vanilla(SACModel):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        lr_alpha: float,
        q1: network.QNetwork,
        q2: network.QNetwork,
        policy: network.GaussianPolicy,
        q1_optimizer: Optimizer,
        q2_optimizer: Optimizer,
        policy_optimizer: Optimizer,
        q1_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        q2_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        policy_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> None:
        self.lr_alpha = lr_alpha
        self.n_observations = n_observations
        self.n_actions = n_actions

        self.q1 = q1
        self.q2 = q2
        self.policy = policy

        self.q1_optimizer = q1_optimizer
        self.q2_optimizer = q2_optimizer
        self.policy_optimizer = policy_optimizer

        self.q1_scheduler = q1_scheduler
        self.q2_scheduler = q2_scheduler
        self.policy_scheduler = policy_scheduler

        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

    def get_play_action(
        self, flat_state: np.ndarray = None, evaluation=False
    ) -> np.ndarray:
        with torch.no_grad():
            flat_state = (
                torch.as_tensor(flat_state, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )

            mean, log_std = self.policy.forward(flat_state)
            std = log_std.exp()

            if evaluation:
                action = torch.tanh(mean)
                rescaled_action = action.cpu().detach().squeeze(0).squeeze(0).numpy()
                return rescaled_action
            else:
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

        # log_pi_batch = torch.sum(normal.log_prob(z), dim=-1, keepdim=True) - torch.sum(
        #        torch.log(1 - action_batch.pow(2) + epsilon), dim=-1, keepdim=True)

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

    def q1_scheduler_step(self):
        if self.q1_scheduler is not None:
            self.q1_scheduler.step()

    def q2_scheduler_step(self):
        if self.q2_scheduler is not None:
            self.q2_scheduler.step()

    def policy_scheduler_step(self):
        if self.policy_scheduler is not None:
            self.policy_scheduler.step()

    def to(self, device: torch.device):
        self.device = device

        self.q1.to(device)
        self.target_q1.to(device)
        self.q1_optimizer.param_groups = []
        self.q1_optimizer.add_param_group({"params": self.q1.parameters()})

        self.q2.to(device)
        self.target_q2.to(device)
        self.q2_optimizer.param_groups = []
        self.q2_optimizer.add_param_group({"params": self.q2.parameters()})

        self.policy.to(device)
        self.policy_optimizer.param_groups = []
        self.policy_optimizer.add_param_group({"params": self.policy.parameters()})

        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        self.alpha_optimizer.param_groups = []
        self.alpha_optimizer.add_param_group({"params": [self.log_alpha]})

    def update_target_q(self, tau):
        for target_param, param in zip(
            self.target_q1.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

        for target_param, param in zip(
            self.target_q2.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def copy(self):
        q1 = self.q1.copy()
        q1_optimizer = self.q1_optimizer.__class__(
            q1,
            **self.q1_optimizer.defaults,
        )
        q1_optimizer.load_state_dict(self.q1_optimizer.state_dict())
        q1_scheduler = deepcopy(self.q1_scheduler)
        if q1_scheduler is not None:
            q1_scheduler.optimizer = q1_optimizer

        q2 = self.q2.copy()
        q2_optimizer = self.q2_optimizer.__class__(
            q2,
            **self.q2_optimizer.defaults,
        )
        q2_optimizer.load_state_dict(self.q2_optimizer.state_dict())
        q2_scheduler = deepcopy(self.q2_scheduler)
        if q2_scheduler is not None:
            q2_scheduler.optimizer = q2_optimizer

        policy = self.policy.copy()
        policy_optimizer = self.policy_optimizer.__class__(
            policy,
            **self.policy_optimizer.defaults,
        )
        policy_optimizer.load_state_dict(self.policy_optimizer.state_dict())
        policy_scheduler = deepcopy(self.policy_scheduler)
        if policy_scheduler is not None:
            policy_scheduler.optimizer = policy_optimizer

        copy = self.__class__(
            self.n_observations,
            self.n_actions,
            self.lr_alpha,
            q1,
            q2,
            policy,
            q1_optimizer,
            q2_optimizer,
            policy_optimizer,
            q1_scheduler,
            q2_scheduler,
            policy_scheduler,
        )

        return copy

    def copy_shared_memory(self):

        self.q1.share_memory()
        self.q2.share_memory()
        self.target_q1.share_memory()
        self.target_q2.share_memory()
        self.policy.share_memory()

        q1 = self.q1
        q1_optimizer = self.q1_optimizer.__class__(
            q1,
            **self.q1_optimizer.defaults,
        )
        q1_optimizer.load_state_dict(self.q1_optimizer.state_dict())
        q1_scheduler = deepcopy(self.q1_scheduler)
        if q1_scheduler is not None:
            q1_scheduler.optimizer = q1_optimizer

        q2 = self.q2
        q2_optimizer = self.q2_optimizer.__class__(
            q2,
            **self.q2_optimizer.defaults,
        )
        q2_optimizer.load_state_dict(self.q2_optimizer.state_dict())
        q2_scheduler = deepcopy(self.q2_scheduler)
        if q2_scheduler is not None:
            q2_scheduler.optimizer = q2_optimizer

        policy = self.policy
        policy_optimizer = self.policy_optimizer.__class__(
            policy,
            **self.policy_optimizer.defaults,
        )
        policy_optimizer.load_state_dict(self.policy_optimizer.state_dict())
        policy_scheduler = deepcopy(self.policy_scheduler)
        if policy_scheduler is not None:
            policy_scheduler.optimizer = policy_optimizer

        copy = self.__class__(
            self.n_observations,
            self.n_actions,
            self.lr_alpha,
            q1,
            q2,
            policy,
            q1_optimizer,
            q2_optimizer,
            policy_optimizer,
            q1_scheduler,
            q2_scheduler,
            policy_scheduler,
        )

        return copy

    def set_network_states(self, network_states_container: SACNetworkStateContainer):
        self.q1.load_state_dict(network_states_container.q1)
        self.q2.load_state_dict(network_states_container.q2)
        self.target_q1.load_state_dict(network_states_container.target_q1)
        self.target_q2.load_state_dict(network_states_container.target_q2)
        self.policy.load_state_dict(network_states_container.policy)
        self.log_alpha.data.copy_(network_states_container.log_alpha["log_alpha"])

    @property
    def network_states_container(self) -> SACNetworkStateContainer:
        network_states_container = SACNetworkStateContainer(
            self.q1.state_dict(),
            self.q2.state_dict(),
            self.target_q1.state_dict(),
            self.target_q2.state_dict(),
            self.policy.state_dict(),
            {"log_alpha": self.log_alpha.detach()},
        )
        return network_states_container

    @property
    def optimizer_states_container(self) -> SACOptimizerStateContainer:
        optimizer_states_container = SACOptimizerStateContainer(
            self.q1_optimizer.state_dict(),
            self.q2_optimizer.state_dict(),
            self.policy_optimizer.state_dict(),
            self.alpha_optimizer.state_dict(),
        )

        return optimizer_states_container

    def set_optimizer_states(
        self, optimizer_states_container: SACOptimizerStateContainer
    ):
        self.q1_optimizer.load_state_dict(optimizer_states_container.q1)
        self.q2_optimizer.load_state_dict(optimizer_states_container.q2)
        self.policy_optimizer.load_state_dict(optimizer_states_container.policy)
        self.alpha_optimizer.load_state_dict(optimizer_states_container.alpha)

    def reset(self) -> None:
        for net in self:
            net.reset()

    def __iter__(self) -> Iterator[network.Network]:
        return iter([self.q1, self.q2, self.target_q1, self.target_q2, self.policy])

    def close(self):
        del self.q1
        del self.q1_optimizer
        del self.q2
        del self.q2_optimizer
        del self.policy
        del self.policy_optimizer
        del self.alpha_optimizer
